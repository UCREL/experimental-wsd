from pathlib import Path
from collections import defaultdict
import json
import logging
import statistics

from dotenv import load_dotenv
load_dotenv()

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from experimental_wsd.config import RaganatoEnglish
from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives
from experimental_wsd.data_processing.lightning_data_modules.semcor import VariableSemCorTraining
from experimental_wsd.evaluation.token_similarity_variable_negatives import evaluate_on_wordnet_dataset

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
        run=True,
        seed_everything_default=False,
        trainer_defaults=trainer_default_dict,
        parser_kwargs={"parser_mode": "omegaconf"}
    )

    wordnet = cli.datamodule.wordnet
    wordnet_lexicon_name = cli.datamodule.lexicon_name
    wordnet_sense_getter = cli.datamodule.get_sense
    tokenizer = cli.datamodule.tokenizer

    best_model_path = str(Path(cli.trainer.checkpoint_callback.best_model_path).resolve())
    model = TokenSimilarityVariableNegatives.load_from_checkpoint(best_model_path)
    model.eval()
    
    validation_datasets = [
        ("SE7", RaganatoEnglish.semeval_2007)
    ]
    test_datasets = [
        ("SE2", RaganatoEnglish.senseval_2),
        ("SE3", RaganatoEnglish.senseval_3),
        ("SE13", RaganatoEnglish.semeval_2013),
        ("SE15", RaganatoEnglish.semeval_2015)
    ]
    all_datasets = [("validation", validation_datasets),
                    ("test", test_datasets)]
    data_metric_results = defaultdict(dict)
    test_results = {"all_micro_f1": [], "all_macro_f1": []}
    for dataset_split_name, datasets in all_datasets:
        for dataset_name, dataset in datasets:
            micro_f1, macro_f1 = evaluate_on_wordnet_dataset(model, dataset, wordnet_sense_getter, tokenizer, wordnet, wordnet_lexicon_name)
            data_metric_results[dataset_split_name][dataset_name] = {
                "micro_f1": micro_f1,
                "macro_f1": macro_f1
            }
            if dataset_split_name == "test":
                test_results["all_micro_f1"].append(micro_f1)
                test_results["all_macro_f1"].append(macro_f1)
            logger.info(f"{dataset_split_name} dataset: {dataset_name}: "
                        f"Micro F1: {micro_f1:.4f} Macro F1: {macro_f1:.4f}")
            
    data_metric_results["test"]["all"] = {
        "micro_f1": statistics.mean(test_results["all_micro_f1"]),
        "macro_f1": statistics.mean(test_results["all_macro_f1"])
    }

    save_data_dict = {
        "metrics": dict(data_metric_results),
        "best_model_path": best_model_path
    }
    
    save_data_filename = "data.json"
    logging_directory = None
    for ml_logger in cli.trainer.loggers:
        if isinstance(ml_logger, CSVLogger):
            version_number = ml_logger.version
            logging_directory = Path(ml_logger.root_dir, f"version_{version_number}")
            
            with Path(logging_directory, save_data_filename).open("w", encoding="utf-8") as save_fp:
                json.dump(save_data_dict, save_fp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model_cli()