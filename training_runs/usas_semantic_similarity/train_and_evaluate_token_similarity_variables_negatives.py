from pathlib import Path
from collections import defaultdict
import json
import logging
import statistics

import torch

from dotenv import load_dotenv
load_dotenv()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser


from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives
from experimental_wsd.data_processing.lightning_data_modules.mosaico_usas import VariableMosaicoUSASTraining

logger = logging.getLogger(__name__)

class TokenSimilarityVariableNegativesCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("tracking_metric")
        parser.add_argument("tracking_metric_mode")
        parser.link_arguments("model.base_model_name", "data.base_model_name")

def model_cli() -> None:

    trainer_default_dict = {
        "gradient_clip_algorithm": "norm",
        "detect_anomaly": False,
        "max_epochs": 20,
        "val_check_interval": 1/5
    }
    torch.set_float32_matmul_precision('high')
    cli = TokenSimilarityVariableNegativesCLI(
        model_class=TokenSimilarityVariableNegatives,
        datamodule_class=VariableMosaicoUSASTraining,
        run=True,
        seed_everything_default=False,
        trainer_defaults=trainer_default_dict,
        parser_kwargs={"parser_mode": "omegaconf"}
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cli()