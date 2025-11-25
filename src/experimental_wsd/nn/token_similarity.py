import inspect
import logging
import math
from collections import OrderedDict

import lightning as L
import torch
from torchmetrics.classification import BinaryAccuracy
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutput

from experimental_wsd.nn.scalar_mix import ScalarMix
from experimental_wsd.nn.utils import (
    get_linear_schedule_with_warmup,
    tiny_value_of_dtype,
)

logger = logging.getLogger(__name__)


class TokenSimilarityVariableNegatives(L.LightningModule):
    def get_total_number_steps(self) -> int:
        batch_size_step = (
            self.trainer.datamodule.batch_size * self.trainer.accumulate_grad_batches
        )
        number_training_samples = len(self.trainer.datamodule.train)
        total_number_training_samples = (
            number_training_samples * self.trainer.max_epochs
        )
        return math.ceil(total_number_training_samples / batch_size_step)

    @staticmethod
    def _get_base_model(base_model_name: str) -> AutoModel:
        """
        Checks if a pooling layer would be added when loaded and if so the
        pooling layer is removed which will remove the number of parameters in
        the model and the number of parameters used.
        """
        base_model = AutoModel.from_pretrained(base_model_name)
        base_model_type = type(base_model)
        if "add_pooling_layer" in inspect.getfullargspec(base_model_type.__init__).args:
            return AutoModel.from_pretrained(base_model_name, add_pooling_layer=False)
        return base_model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        total_number_steps = self.get_total_number_steps()
        total_epoch_steps = total_number_steps / self.trainer.max_epochs
        warm_up_steps = math.ceil(total_number_steps * self.fraction_of_warm_up_steps)
        logger.info(f"Total number of training steps: {total_number_steps}")
        logger.info(f"Number steps per epoch: {total_epoch_steps}")
        logger.info(f"Total number warmup steps: {warm_up_steps}")
        if self.use_scheduler:
            linear_warmup_with_decay = get_linear_schedule_with_warmup(
                optimizer, warm_up_steps, total_number_steps
            )
            return [optimizer], [
                {
                    "scheduler": linear_warmup_with_decay,
                    "interval": "step",
                    "frequency": 1,
                }
            ]
        else:
            return optimizer

    def __init__(
        self,
        base_model_name: str,
        freeze_base_model: bool,
        number_transformer_encoder_layers: int,
        add_scalar_mixer: bool = True,
        scalar_mix_layer_norm: bool = True,
        transformer_encoder_hidden_dim: int = 512,
        transformer_encoder_num_heads: int = 8,
        batch_first: bool = True,
        learning_rate: float = 2e-5,
        weight_decay: float = 1e-2,
        use_scheduler: bool = True,
        fraction_of_warm_up_steps: float = 0.1,
        scheduler_frequency: int = 1,
    ) -> None:
        """
        Args:
            base_model_name (str): The name of the HuggingFace base model
                to use, e.g. FacebookAI/roberta-base
            freeze_base_model (bool): Whether the base model should not be
                trained (the model weights are frozen).
            number_transformer_encoder_layers (int): The number of transformer
                encoder layers to add to the base model. Can be 0.
            scalar_mix_layer_norm (bool): Whether the scalar mixer should normalise
                each transformer hidden layer before weighting or not.
            batch_first (bool): If the batch should be first dimension,
                (batch, seq, feature) else False will be (seq, batch, feature)
        """

        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.scheduler_frequency = scheduler_frequency
        self.fraction_of_warm_up_steps = fraction_of_warm_up_steps
        self.base_model_name = base_model_name
        self.base_model = self._get_base_model(self.base_model_name)
        self.base_model_hidden_size = self.base_model.config.hidden_size
        self.freeze_base_model = freeze_base_model
        logger.info(f"Base model: {self.base_model_name} loaded")
        # Add 1 for the embedding layer
        self.base_model_number_hidden_layers = (
            self.base_model.config.num_hidden_layers + 1
        )
        logger.info(
            "Number of hidden layers in base model: "
            f"{self.base_model_number_hidden_layers}"
        )
        if self.freeze_base_model:
            self.base_model.train(False)
            logger.info("Freezing base model parameters")
            for base_model_parameter in self.base_model.parameters():
                base_model_parameter.requires_grad = False
        else:
            self.base_model.train(True)
            logger.info("Training base model parameters")
            for base_model_parameter in self.base_model.parameters():
                base_model_parameter.requires_grad = True

        self.scalar_mix = add_scalar_mixer
        self.scalar_mix_layer_norm = scalar_mix_layer_norm
        if add_scalar_mixer:
            self.scalar_mix = ScalarMix(
                self.base_model_number_hidden_layers,
                do_layer_norm=self.scalar_mix_layer_norm,
            )

        # Optional list of layers to further encode the tokens after embedding
        # from the base transformer model.
        token_model_layers_list: list[tuple[str, torch.nn.Module]] = []
        self.token_model_layers: torch.nn.Sequential | None = None

        self.batch_first = batch_first
        self.number_transformer_encoder_layers = number_transformer_encoder_layers
        self.transformer_encoder_hidden_dim = transformer_encoder_hidden_dim
        self.transformer_encoder_num_heads = transformer_encoder_num_heads
        self.linear_bridge: torch.nn.Linear | None = None
        self.transformer: torch.nn.TransformerEncoder | None = None

        if self.number_transformer_encoder_layers:
            logger.info(
                f"Adding {self.number_transformer_encoder_layers} "
                "transformer encoder layers to the base model."
            )
            if self.transformer_encoder_hidden_dim != self.base_model_hidden_size:
                logger.info(
                    "Base model hidden dimension "
                    f"{self.base_model_hidden_size} does not match "
                    "Transformer encoder model hidden dimension "
                    f"{self.transformer_encoder_hidden_dim}"
                    "creating a linear layer bridge between the two."
                )
                self.linear_bridge = torch.nn.Linear(
                    self.base_model_hidden_size, self.transformer_encoder_hidden_dim
                )
                token_model_layers_list.append(("Linear Bridge", self.linear_bridge))

            transformer_encoder_layer = torch.nn.TransformerEncoderLayer(
                self.transformer_encoder_hidden_dim,
                self.transformer_encoder_num_heads,
                batch_first=self.batch_first,
            )
            self.token_transformer = torch.nn.TransformerEncoder(
                transformer_encoder_layer, num_layers=number_transformer_encoder_layers
            )
            token_model_layers_list.append(
                ("Token Transformer", self.token_transformer)
            )
            self.token_model_layers = torch.nn.Sequential(
                OrderedDict(token_model_layers_list)
            )

        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=None, reduction="mean", ignore_index=-100
        )

        # This allows for more than one metric if we want more than one metric
        split_names = ["train", "validation", "test"]
        standard_metric_kwargs = {"ignore_index": -100, "multidim_average": "global"}
        self.metric_names = ["accuracy"]
        all_metric_class_args = [(BinaryAccuracy, standard_metric_kwargs)]
        for split_name in split_names:
            for metric_name, metric_class_args in zip(
                self.metric_names, all_metric_class_args
            ):
                metric_class, metric_args = metric_class_args
                metric = metric_class(**metric_args)
                setattr(self, f"{split_name}_{metric_name}", metric)

        self.save_hyperparameters()

    def _token_encoding(
        self, token_input_ids: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Given the token IDs and the attention masks for those token ID's it
        returns a contextualised embedding for each token using the following
        applicable layers:
        1. self.base_model - Typically a pre-train language Model.
        2. self.scalar_mix - The average embedding for that token based on a learnt
            weighting of all of the base model layers.
        3. self.token_model_layers - to be learnt transformer layers.

        Args:
            token_input_ids (torch.Tensor): The token IDs to embed. torch.Long.
                Shape (Batch, Sequence Length).
            token_attention_mask (torch.Tensor): The attention mask associated
                with the token IDs. torch.Long. Shape (Batch, Sequence Length).
        Returns:
            torch.Tensor: A contextualised embedding for each token. torch.Float.
                Shape (Batch, Sequence Length, Embedding Dimension)
        """

        base_model_output: BaseModelOutput = self.base_model(
            token_input_ids, token_attention_mask, output_hidden_states=True
        )

        # self.base_model_number_hidden_layers of hidden layers of
        # (BATCH, SEQUENCE, self.base_model_hidden_size)

        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        token_model_embedding = base_model_output.last_hidden_state
        if self.scalar_mix:
            base_model_hidden_layers = base_model_output.hidden_states
            token_model_embedding = self.scalar_mix(
                base_model_hidden_layers, token_attention_mask
            )

        # Further token encoding through the token model layers
        if self.token_model_layers:
            token_model_embedding = self.token_model_layers(token_model_embedding)

        return token_model_embedding

    @staticmethod
    def _average_token_embedding_pooling(
        token_embeddings: torch.Tensor, token_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        The token embeddings average pooled over the sequence length taking
        into account the attention mask.

        Written with Mistral Codestral.

        Args:
            token_embeddings (torch.Tensor): The embeddings/encodings for a
                batch of tokens whereby the batch can be of various shapes.
                torch.Float. Shape (..., Sequence Length, Embedding Dimension).
            token_attention_mask (torch.Tensor): The attention mask for the
                token embeddings. The mask is expected to be a Binary mask (1 or 0).
                torch.Long. Shape (..., Sequence Length).
        Returns:
            torch.Tensor: The average token embeddings pooled over each sequence
                in the batch taking into account the attention mask. torch.Float.
                Shape (..., Embedding Dimension).
        Raises:
            ValueError: If token_embeddings doesn't have at least 3 dimensions,
                or if token_attention_mask's dimension
                doesn't match the expected shape (one less than token_embeddings).
        """

        # Validate input shapes
        if token_embeddings.dim() < 3:
            raise ValueError(
                "token_embeddings must have at least 3 dimensions "
                "(... , Sequence Length, Embedding Dimension), "
                f"got {token_embeddings.shape}"
            )
        if (
            token_attention_mask.dim() < 2
            or token_attention_mask.dim() != token_embeddings.dim() - 1
        ):
            raise ValueError(
                "token_attention_mask must have one less dimension "
                "than token_embeddings (... , Sequence Length), "
                f"got {token_attention_mask.shape} for the attention mask and "
                f"{token_embeddings.shape} for the embeddings."
            )

        # Float Tensor, shape (..., Sequence Length, 1)
        broadcast_token_attention_mask = token_attention_mask.unsqueeze(-1).to(
            token_embeddings
        )
        # Float tensor, shape (..., Embedding Dimension)
        masked_embedding_sum = torch.mul(
            token_embeddings, broadcast_token_attention_mask
        ).sum(dim=-2)
        # Float tensor of (..., 1) represents the number of tokens that make up the given word.
        number_token_vectors = broadcast_token_attention_mask.sum(-2)
        # Stops dividing by zero which causes nan values
        tiny_value_to_stop_nan = tiny_value_of_dtype(number_token_vectors.dtype)
        number_token_vectors = torch.clamp(
            number_token_vectors, min=tiny_value_to_stop_nan
        )

        # Float tensor, shape (..., Embedding Dimension)
        average_token_embeddings = masked_embedding_sum / number_token_vectors
        return average_token_embeddings

    def label_definition_encoding(
        self,
        label_definitions_input_ids: torch.Tensor,
        label_definitions_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        BATCH_SIZE, S, ST = label_definitions_input_ids.shape

        # Encoding the label definition sequences, these need to be reshaped
        # so that they can be processed by the token encoding model/layers
        definition_input_ids_encoding = label_definitions_input_ids.view(-1, ST)
        definition_attention_mask_encoding = label_definitions_attention_mask.view(
            -1, ST
        )
        definition_token_embedding = self._token_encoding(
            definition_input_ids_encoding, definition_attention_mask_encoding
        )
        average_definition_token_embeddings = self._average_token_embedding_pooling(
            definition_token_embedding, definition_attention_mask_encoding
        )
        # View the embeddings back to shape:
        # (Batch, S, Embedding Dimension)
        average_definition_token_embeddings = average_definition_token_embeddings.view(
            BATCH_SIZE, S, -1
        )
        return average_definition_token_embeddings

    def text_encoding(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Encoding the text sequence
        # Shape (B, D)
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        return text_encoding

    def token_encoding_using_text_encoding(
        self, text_encoding: torch.Tensor, text_word_ids_mask: torch.Tensor
    ) -> torch.Tensor:
        # Expanded so that we have a text embedding per positive sample
        # Current Shape (Batch, Sequence Length, Dimension)
        # New Shape (B, M, T, D)
        # expanded_text_encoding = text_encoding.unsqueeze(1).expand(-1, S, -1, -1)
        # Shape (B, D)
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding

    def token_text_encoding(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_word_ids_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Encoding the text sequence
        # Shape (B, D)
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        # Expanded so that we have a text embedding per positive sample
        # Current Shape (Batch, Sequence Length, Dimension)
        # New Shape (B, M, T, D)
        # expanded_text_encoding = text_encoding.unsqueeze(1).expand(-1, S, -1, -1)
        # Shape (B, D)
        average_text_encoding = self._average_token_embedding_pooling(
            text_encoding, text_word_ids_mask
        )
        return average_text_encoding

    def token_label_similarity(
        self,
        label_definition_embedding: torch.Tensor,
        token_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # Expand the text encoding so that we can get token similarity for each
        # label definition
        expanded_average_text_encoding = token_text_embedding.unsqueeze(-1)
        expanded_similarity_score = torch.matmul(
            label_definition_embedding, expanded_average_text_encoding
        )
        similarity_score = expanded_similarity_score.squeeze(-1)
        return similarity_score

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_word_ids_mask: torch.Tensor,
        label_definitions_input_ids: torch.Tensor,
        label_definitions_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """

        B represents the batch size. This is the number of text sequences.
        S represents the largest number of similarity sentences within one text
            sequence within the batch.
        T represents the largest token length for the text sample.
        ST represents the largest token length for the label definitions sentences.

        Args:
            text_input_ids (torch.Tensor): Tokenized text sample
                which contains all of the tokens a single set of tokens will be
                encoded and matched against the label definitions to determine
                which definition is the most similar.
                torch.Long Shape (B, T).
            text_attention_mask (torch.Tensor): 1 or 0 attention mask for the
                text samples. torch.Long tensor. 1 represents a token to
                attend to, 0 a token to ignore. Shape (B, T).
            text_word_ids_mask (torch.Tensor): A token mask for the
                single set of tokens used to average the encoded text inputs.
                torch.Long. Shape (B, T).
            label_definitions_input_ids (torch.Tensor): The input ids for the
                sentences to match the token encoding against to determine which
                is the most similar. torch.Long. Shape (B, S, ST).
            label_definitions_attention_mask (torch.Tensor): The attention
                mask (1 or 0) for the `label_definitions_input_ids`. torch.Long.
                Shape (B, S, ST)

        Returns:
            torch.Tensor: A floating point tensor of shape (B, S).


        This type checking library might be useful in the future:
        https://docs.kidger.site/jaxtyping/
        """
        average_definition_token_embeddings = self.label_definition_encoding(
            label_definitions_input_ids, label_definitions_attention_mask
        )

        average_text_encoding = self.token_text_encoding(
            text_input_ids, text_attention_mask, text_word_ids_mask
        )

        similarity_score = self.token_label_similarity(
            average_definition_token_embeddings, average_text_encoding
        )

        return similarity_score

    def _forward_with_loss_and_metric(
        self, batch: dict[str, torch.Tensor], split: str, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        logits = self(
            batch["text_input_ids"],
            batch["text_attention_mask"],
            batch["text_word_ids_mask"],
            batch["label_definitions_input_ids"],
            batch["label_definitions_attention_mask"],
        )

        labels = batch["label_ids"]
        loss = self.loss_fn(logits, labels)

        pred_labels = torch.argmax(logits, dim=-1)
        metric_pred_labels = torch.zeros_like(labels)
        metric_pred_labels[pred_labels == labels] = 1
        metric_labels = torch.ones_like(labels)

        for metric_name in self.metric_names:
            metric = getattr(self, f"{split}_{metric_name}")
            metric(metric_pred_labels, metric_labels)
            self.log(
                f"{split}_{metric_name}",
                metric,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
            )

        number_labels = labels.shape[0]
        self.log(
            f"{split}_loss",
            loss,
            batch_size=number_labels,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "train", batch_idx)

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "validation", batch_idx)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "test", batch_idx)
