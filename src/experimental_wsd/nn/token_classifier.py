import logging
from collections import OrderedDict
import inspect

import lightning as L
import torch
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.classification.stat_scores import MulticlassStatScores

from experimental_wsd.nn.scalar_mix import ScalarMix
from experimental_wsd.nn.utils import tiny_value_of_dtype

logger = logging.getLogger(__name__)


class TokenClassifier(L.LightningModule):

    @staticmethod
    def _get_base_model(base_model_name: str) -> AutoModel:
        """
        Checks if a pooling layer would be added when loaded and if so the 
        pooling layer is removed which will remove the number of parameters in 
        the model and the number of parameters used.
        """
        base_model = AutoModel.from_pretrained(
            base_model_name
        )
        base_model_type = type(base_model)
        if 'add_pooling_layer' in inspect.getfullargspec(base_model_type.__init__).args:
            return AutoModel.from_pretrained(
            base_model_name, add_pooling_layer=False
            )
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
        number_classes: int = 2,
        classifier_dropout: float | None = 0.1,
        label_weights: list[int] | None = None
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
        self.number_classes = number_classes
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
            logger.info("Freezing base model parameters")
            for base_model_parameter in self.base_model.parameters():
                base_model_parameter.requires_grad = False

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
        self.output_linear_layer: torch.nn.Linear | None = None
        self.classifier_dropout: torch.nn.Dropout | None = None
        if classifier_dropout:
            self.classifier_dropout = torch.nn.Dropout(classifier_dropout)
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
            self.output_linear_layer = torch.nn.Linear(
                self.transformer_encoder_hidden_dim, self.number_classes
            )
        else:
            self.output_linear_layer = torch.nn.Linear(
                self.base_model_hidden_size, self.number_classes
            )
        # We could add a positive weight tensor that needs to be of shape C
        # Where C is the number of classes in this case 1 class.
        self.label_weights = label_weights
        if self.label_weights:
            self.label_weights = torch.tensor(label_weights, dtype=torch.float, device=self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=self.label_weights, reduction="mean", ignore_index=-100)

        
        split_names = ['train', 'validation', 'test']
        standard_metric_kwargs = {
            "num_classes": self.number_classes,
            "ignore_index": -100,
            "multidim_average": "global",
            "average": "macro"
        }
        self.metric_names = ["macro_accuracy", "macro_f1"]
        all_metric_class_args = [
            (MulticlassAccuracy, standard_metric_kwargs),
            (MulticlassF1Score, standard_metric_kwargs)
        ]
        for split_name in split_names:
            for metric_name, metric_class_args in zip(self.metric_names, all_metric_class_args):
                metric_class, metric_args = metric_class_args
                metric = metric_class(**metric_args)
                setattr(self, f'{split_name}_{metric_name}', metric)

        self.save_hyperparameters()


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            input_ids (torch.Tensor): Tokenized input ids, torch.Long.
                Shape (Batch, Sequence Length)
            attention_mask (torch.Tensor): 1 or 0 torch.Long tensor. 1 represents
                a token to attend to, 0 a token to ignore.
                Shape (Batch, Sequence Length).
            word_ids (torch.Tensor): Indexes of the words that the token's represent.
                Any word id that is less than 0 is assumed to not be a word, e.g.
                a special token symbol like [CLS] or [SEP].
                Shape (Batch, Sequence Length).
        Returns:
            torch.Tensor: A floating point tensor of shape (Batch, Maximum number of words, Number Classes).
                The number of classes is determined by self.number_classes.
                The Maximum number of words is determined by the argument `word_ids` whereby
                we assume any word id equal to or greater than 0 is a unique word. Therefore
                we are ignoring all special word ids.


        This type checking library might be useful in the future:
        https://docs.kidger.site/jaxtyping/
        """
        BATCH_SIZE = input_ids.shape[0]

        base_model_output: BaseModelOutputWithPoolingAndCrossAttentions = (
            self.base_model(input_ids, attention_mask, output_hidden_states=True)
        )
        # self.base_model_number_hidden_layers of hidden layers of
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        base_model_embedding_layers = base_model_output.hidden_states
        # (BATCH, SEQUENCE, self.base_model_hidden_size)
        token_model_embedding = self.scalar_mix(
            base_model_embedding_layers, attention_mask
        )
        token_embedding_dim = self.base_model_hidden_size

        # Further token encoding through the token model layers
        if self.token_model_layers:
            token_model_embedding = self.token_model_layers(token_model_embedding)
            token_embedding_dim = self.transformer_encoder_hidden_dim
        
        # Whether to apply dropout to the encoded sequence
        if self.classifier_dropout:
            token_model_embedding = self.classifier_dropout(token_model_embedding)

        largest_word_id = torch.max(word_ids)
        # NOTE DIFFERENCE: float of BATCH, UNIQUE_WORD_IDS, token_embedding_dim
        batch_average_word_vectors = torch.zeros(
            (BATCH_SIZE, largest_word_id + 1, token_embedding_dim),
            device=self.device
        )
        for word_id in torch.arange(0, largest_word_id + 1):
            # Bool of BATCH, SEQUENCE
            word_id_mask = word_ids == word_id
            # Float of BATCH, SEQUENCE, 1
            broadcast_word_id_mask = word_id_mask.unsqueeze(-1).to(dtype=torch.float)
            # Float of BATCH, token_embedding_dim, represents the sum of
            # all the token vectors that make up the single word.
            word_vectors = torch.mul(token_model_embedding, broadcast_word_id_mask).sum(
                -2
            )
            # Long of (Batch, 1), e.g. [[1.0], [2.0], [1.0], [3.0], [0.0]] which 
            # represent the number of tokens that make up the given word.
            number_token_vectors = word_id_mask.sum(-1).to(word_vectors).unsqueeze(-1)
            # Stops dividing by zero which causes nan values 
            tiny_value_to_stop_nan = tiny_value_of_dtype(number_token_vectors.dtype)
            number_token_vectors = number_token_vectors + tiny_value_to_stop_nan
            # Average them as the sum could contain more than 1 token embedding
            # Float Batch, self.base_model_hidden_size
            average_word_vectors = word_vectors / number_token_vectors
            batch_average_word_vectors[:, word_id, :] = average_word_vectors
        # Float BATCH, UNIQUE_WORD_IDS, self.number_classes
        output_vectors = self.output_linear_layer(batch_average_word_vectors)
        # Note that some values will be `nan` as the word_ids are likely to have
        # ids to ignore which are values associated with -100.
        return output_vectors
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        return optimizer
    
    def _forward_with_loss_and_metric(self,
                                      batch: dict[str, torch.Tensor],
                                      split: str,
                                      batch_idx: int
                                      ) -> dict[str, torch.Tensor]:
        # Float (BATCH, SEQUENCE, NUMBER CLASSES)
        logits = self(batch['input_ids'], batch['attention_mask'], batch['word_ids'])
        labels = batch['labels']
        labels_mask = (labels != -100).to(logits)

        # Float (BATCH * SEQUENCE, NUMBER CLASSES)
        loss_logits = logits.view(-1, self.number_classes)
        # Long (BATCH * SEQUENCE)
        loss_labels = labels.view(-1)
        
        loss = self.loss_fn(loss_logits, loss_labels)

        number_words = labels_mask.sum()
        
        # torch metrics does not work with logits when using ignore index
        # Float (BATCH * SEQUENCE)
        pred_classes = torch.argmax(loss_logits, dim=-1)
        for metric_name in self.metric_names:
            metric = getattr(self, f'{split}_{metric_name}')
            metric(pred_classes, loss_labels)
            self.log(f'{split}_{metric_name}', metric, on_epoch=True, on_step=True, prog_bar=True) 

        self.log(f'{split}_loss', loss, batch_size=number_words, on_epoch=True, on_step=True, prog_bar=True)
        return {"loss": loss}

    def training_step(self,
                      batch: dict[str, torch.Tensor],
                      batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "train", batch_idx)
        
    
    def validation_step(self,
                  batch: dict[str, torch.Tensor],
                  batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "validation", batch_idx)
    
    def test_step(self,
                  batch: dict[str, torch.Tensor],
                  batch_idx: int):
        return self._forward_with_loss_and_metric(batch, "test", batch_idx)