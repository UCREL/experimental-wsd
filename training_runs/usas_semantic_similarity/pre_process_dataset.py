import logging

from dotenv import load_dotenv

load_dotenv()


from experimental_wsd.data_processing.lightning_data_modules.mosaico_usas import (  # noqa: E402
    VariableMosaicoUSASTraining,
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    base_model_name_help = (
        "Name of the base model to use for pre-processing the data. "
        "This should be a HuggingFace model name. "
        "This is required to download/retrieve the correct tokenizer."
    )
    dataset_folder_name_help = (
        "Name of the folder to save the pre-processed dataset to. "
        "This folder will exist under the directory "
        "`$EXPERIMENTAL_WSD_DATA_PROCESSING_DIR/machine_learning_data/`. "
        "If the `$EXPERIMENTAL_WSD_DATA_PROCESSING_DIR` environment variable is not set, "
        "this will default to `$HOME/.cache/experimental_wsd/machine_learning_data`. "
        "This folder will be created if it does not exist."
    )
    attention_pad_id_help = (
        "The Integer ID that represents the attention pad token for the attention mask. "
        "Default is 0"
    )
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model_name", type=str, help=base_model_name_help)
    parser.add_argument("dataset_folder_name", type=str, help=dataset_folder_name_help)
    parser.add_argument(
        "--num-cpus-pre-processing",
        type=int,
        default=1,
        help="Number of CPUs to use for pre-processing the data. Default is 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the pre-processed data if it already exists. Default is False.",
    )
    parser.add_argument(
        "--attention-pad-id", default=0, type=int, help=attention_pad_id_help
    )
    parser.add_argument(
        "--filter-out-labels",
        nargs="+",
        help="USAS tags to filter out of the training and evaluation datasets",
        required=False,
    )
    args = parser.parse_args()

    base_model_name = args.base_model_name
    dataset_folder_name = args.dataset_folder_name
    num_cpus_pre_processing = args.num_cpus_pre_processing
    overwrite = args.overwrite
    attention_pad_id = args.attention_pad_id
    filter_out_labels = args.filter_out_labels

    logger.info(f"Base model name: {base_model_name}")
    logger.info(
        f"Dataset folder name the final dataset will be saved too: {dataset_folder_name}"
    )
    logger.info(f"Number CPUs for pre-processing: {num_cpus_pre_processing}")
    logger.info(f"Overwrite data: {overwrite}")
    logger.info(f"Attention pad id: {attention_pad_id}")
    logger.info(
        f"Filtering out the following USAS tags from the training and evaluation datasets: {filter_out_labels}"
    )
    dataset = VariableMosaicoUSASTraining(
        base_model_name,
        dataset_folder_name=dataset_folder_name,
        tokenizer_kwargs={"add_prefix_space": True},
        batch_size=1,
        num_dataloader_cpus=1,
        num_cpus_pre_processing=num_cpus_pre_processing,
        overwrite_all_pre_processed_data=overwrite,
        attention_pad_id=attention_pad_id,
        filter_out_labels=filter_out_labels,
    )
    logger.info("Starting to pre-process data")
    dataset.prepare_data()
    logger.info("Finished pre-processing data")
