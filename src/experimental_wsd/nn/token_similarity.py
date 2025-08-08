import inspect
import logging
from collections import OrderedDict

import lightning as L
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from experimental_wsd.nn.scalar_mix import ScalarMix
from experimental_wsd.nn.utils import tiny_value_of_dtype

logger = logging.getLogger(__name__)


class TokenSimilarityVariableNegatives(L.LightningModule):
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

    def __init__(
        self,
        base_model_name: str,
        freeze_base_model: bool,
        number_transformer_encoder_layers: int,
        scalar_mix_layer_norm: bool = True,
        transformer_encoder_hidden_dim: int = 512,
        transformer_encoder_num_heads: int = 8,
        batch_first: bool = True,
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
            for base_model_parameter in self.base_model.parameters():
                base_model_parameter.requires_grad = True

        self.scalar_mix_layer_norm = scalar_mix_layer_norm
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

        split_names = ["train", "validation", "test"]
        standard_metric_kwargs = {"ignore_index": -100, "multidim_average": "global"}
        self.metric_names = ["accuracy", "micro_f1"]
        all_metric_class_args = [
            (BinaryAccuracy, standard_metric_kwargs),
            (BinaryF1Score, standard_metric_kwargs),
        ]
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

        base_model_output: BaseModelOutputWithPoolingAndCrossAttentions = (
            self.base_model(
                token_input_ids, token_attention_mask, output_hidden_states=True
            )
        )

        # self.base_model_number_hidden_layers of hidden layers of
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        base_model_embedding_layers = base_model_output.hidden_states
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        token_model_embedding = self.scalar_mix(
            base_model_embedding_layers, token_attention_mask
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

    def forward(
        self,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_attention_mask: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        text_word_ids_mask: torch.Tensor,
        random_label_reindexing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """

        B represents the batch size. This is the number of text sequences.
        M represents the largest number of positive samples within one text
        sequence for the entire batch.
        N represents the largest number of negative samples for a positive sample
        within the batch of sequences (B).
        T represents the largest token length for the text sample.
        PT represents the largest token length for the positive text samples.
        NT represents the largest token length for the negative text samples.

        Args:
            positive_input_ids (torch.Tensor): Tokenized positive text sample
                for a token (MWE). torch.Long Shape (B, M, PT).
            positive_attention_mask (torch.Tensor): 1 or 0 attention mask for the
                positive text samples. torch.Long tensor. 1 represents a token to
                attend to, 0 a token to ignore. Shape (B, M, PT).
            negative_input_ids (torch.Tensor): Tokenized negative text samples
                for a token (MWE). torch.Long Shape (B, M, N, NT).
            negative_attention_mask (torch.Tensor): 1 or 0 attention mask for the
                negative text samples. torch.Long tensor. 1 represents a token to
                attend to, 0 a token to ignore. Shape (B, M, N, NT).
            text_input_ids (torch.Tensor): Tokenized text sample
                which contains all of the tokens whereby some will have positive
                and negative samples which the model learns from.
                torch.Long Shape (B, T).
            text_attention_mask (torch.Tensor): 1 or 0 attention mask for the
                text samples. torch.Long tensor. 1 represents a token to
                attend to, 0 a token to ignore. Shape (B, T).
            text_word_ids_mask (torch.Tensor): A token mask for each token that
                has a positive and negative samples to learn from. torch.Long.
                Shape (B, M, T).
        Returns:
            torch.Tensor: A floating point tensor of shape (B, M, 1 + N).


        This type checking library might be useful in the future:
        https://docs.kidger.site/jaxtyping/
        """

        BATCH_SIZE = text_input_ids.shape[0]
        M = positive_input_ids.shape[1]
        PT = positive_input_ids.shape[2]
        N = negative_input_ids.shape[2]
        NT = negative_input_ids.shape[3]

        # Encoding the Positive text sequences
        # Need to reshape the positive text sequences so that they can be
        # processed by the token encoding model/layers
        positive_input_ids_encoding = positive_input_ids.view(-1, PT)
        positive_attention_mask_encoding = positive_attention_mask.view(-1, PT)
        positive_token_embedding = self._token_encoding(
            positive_input_ids_encoding, positive_attention_mask_encoding
        )
        average_positive_token_embeddings = self._average_token_embedding_pooling(
            positive_token_embedding, positive_attention_mask_encoding
        )
        # View the embeddings back to shape:
        # (Batch, M, Embedding Dimension)
        average_positive_token_embeddings = average_positive_token_embeddings.view(
            BATCH_SIZE, M, -1
        )

        # Encoding the Negative text sequences
        negative_input_ids_encoding = negative_input_ids.view(-1, NT)
        negative_attention_mask_encoding = negative_attention_mask.view(-1, NT)
        negative_token_embedding = self._token_encoding(
            negative_input_ids_encoding, negative_attention_mask_encoding
        )
        average_negative_token_embeddings = self._average_token_embedding_pooling(
            negative_token_embedding, negative_attention_mask_encoding
        )
        # View the embeddings back to shape:
        # (Batch, M, N, Embedding Dimension)
        average_negative_token_embeddings = average_negative_token_embeddings.view(
            BATCH_SIZE, M, N, -1
        )

        # Combine the positive and negative embeddings, so that the positive embeddings
        # Are always the first embedding in the sequence
        average_positive_token_embeddings = average_positive_token_embeddings.unsqueeze(
            -2
        )
        # Shape (Batch, M, N+1, Dimension)
        average_positive_negative_embeddings = torch.cat(
            (average_positive_token_embeddings, average_negative_token_embeddings),
            dim=-2,
        )
        if random_label_reindexing is not None:
            average_positive_negative_embeddings = average_positive_negative_embeddings[
                :, :, random_label_reindexing, :
            ]

        # Encoding the text sequence
        text_encoding = self._token_encoding(text_input_ids, text_attention_mask)
        # Expanded so that we have a text embedding per positive sample
        # Current Shape (Batch, Sequence Length, Dimension)
        # New Shape (B, M, T, D)
        expanded_text_encoding = text_encoding.unsqueeze(1).expand(-1, M, -1, -1)
        # Shape (B, M, D)
        average_expanded_text_encoding = self._average_token_embedding_pooling(
            expanded_text_encoding, text_word_ids_mask
        )
        # Shape (B, M, D, 1)
        average_expanded_text_encoding = average_expanded_text_encoding.unsqueeze(-1)
        # Generate the similarity score through dot product.
        # Shape (B, M, N + 1, 1)
        similarity_score = torch.matmul(
            average_positive_negative_embeddings, average_expanded_text_encoding
        )
        # Shape (B, M, N + 1)
        similarity_score = similarity_score.squeeze(-1)
        return similarity_score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer

    def _forward_with_loss_and_metric(
        self, batch: dict[str, torch.Tensor], split: str, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        # Float (Batch, M, N + 1)
        number_of_possible_classes = batch["negative_attention_mask"].shape[-2] + 1
        # Long (N + 1)
        random_label_reindexing = torch.randperm(number_of_possible_classes).to(
            self.device
        )

        logits = self(
            batch["positive_input_ids"],
            batch["positive_attention_mask"],
            batch["negative_input_ids"],
            batch["negative_attention_mask"],
            batch["text_input_ids"],
            batch["text_attention_mask"],
            batch["text_word_ids_mask"],
            random_label_reindexing=random_label_reindexing,
        )
        # This is in essence N + 1
        number_of_negative_samples_with_positive = logits.shape[-1]
        # Float (Batch * M, N + 1)
        logits_compressed = logits.view(-1, number_of_negative_samples_with_positive)
        # Shape (Batch, M) Should all be 0's as this is the correct index, except
        # for samples that should be ignored, they will contain the `label_pad_id`
        # which is likely to be -100
        labels = batch["labels"]
        correct_index = (random_label_reindexing == 0).nonzero().item()
        # labels that are correct are always zero therefore need to replace 0
        # values with correct index value.

        labels[labels == 0] = correct_index

        # For the loss the labels need to be of shape (Batch * M)
        compressed_labels = labels.view(-1)
        loss = self.loss_fn(logits_compressed, compressed_labels)

        number_labels = (compressed_labels != -100).sum()

        # Correct index is always correct_index, shape (Batch * M)
        pred_labels = torch.argmax(logits_compressed, dim=-1)
        pred_labels[pred_labels != correct_index] = 0
        pred_labels[pred_labels == correct_index] = 1

        true_labels = compressed_labels.clone()
        true_labels[true_labels == correct_index] = 1

        for metric_name in self.metric_names:
            metric = getattr(self, f"{split}_{metric_name}")
            metric(pred_labels, true_labels)
            self.log(
                f"{split}_{metric_name}",
                metric,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )

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
