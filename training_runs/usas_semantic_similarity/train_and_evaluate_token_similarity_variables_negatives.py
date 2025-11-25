import logging

import torch
from dotenv import load_dotenv

load_dotenv()

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI  # noqa: E402

from experimental_wsd.data_processing.lightning_data_modules.mosaico_usas import (  # noqa: E402
    VariableMosaicoUSASTraining,
)
from experimental_wsd.nn.token_similarity import (  # noqa: E402
    TokenSimilarityVariableNegatives,
)

logger = logging.getLogger(__name__)


class TokenSimilarityVariableNegativesCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument("tracking_metric")
        parser.add_argument("tracking_metric_mode")
        parser.link_arguments("model.base_model_name", "data.base_model_name")


def model_cli() -> None:
    torch.set_float32_matmul_precision("high")
    cli = TokenSimilarityVariableNegativesCLI(  # noqa: F841
        model_class=TokenSimilarityVariableNegatives,
        datamodule_class=VariableMosaicoUSASTraining,
        run=True,
        seed_everything_default=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cli()
