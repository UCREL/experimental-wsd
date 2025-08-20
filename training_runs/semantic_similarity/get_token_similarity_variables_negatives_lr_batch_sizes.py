import logging

from dotenv import load_dotenv
load_dotenv()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.tuner.tuning import Tuner

from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives
from experimental_wsd.data_processing.lightning_data_modules.semcor import VariableSemCorTraining

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
        "val_check_interval": 1/3
    }
    cli = TokenSimilarityVariableNegativesCLI(
        model_class=TokenSimilarityVariableNegatives,
        datamodule_class=VariableSemCorTraining,
        run=False,
        seed_everything_default=False,
        trainer_defaults=trainer_default_dict,
        parser_kwargs={"parser_mode": "omegaconf"}
    )
    tuner = Tuner(cli.trainer)
    lr_results = tuner.lr_find(cli.model, datamodule=cli.datamodule, attr_name="learning_rate")
    lr_suggestion = lr_results.suggestion()
    lr_suggestion_loss = None
    learning_rate_results = lr_results.results
    sorted_learning_rates = sorted(zip(learning_rate_results['lr'],
                                       learning_rate_results['loss']),
                                   key=lambda x: x[1])
    for learning_rate, loss in sorted_learning_rates:
        logging.info(f"LR: {learning_rate}    Loss: {loss}")
        if learning_rate == lr_suggestion:
            lr_suggestion_loss = loss
    
    if lr_suggestion_loss:
        logging.info(f"Suggested LR: {lr_suggestion}, loss at this LR: {lr_suggestion_loss}")
    else:
        logging.info(f"Suggested LR: {lr_suggestion}")

    batch_size_result = tuner.scale_batch_size(cli.model, datamodule=cli.datamodule, batch_arg_name="batch_size", method="fit", steps_per_trial=50)
    logging.info(f'Largest batch size: {batch_size_result}')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cli()