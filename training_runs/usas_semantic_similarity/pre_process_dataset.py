import logging

from dotenv import load_dotenv
load_dotenv()


from experimental_wsd.data_processing.lightning_data_modules.mosaico_usas import VariableMosaicoUSASTraining

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model_name", type=str)
    parser.add_argument("dataset_folder_name", type=str)
    parser.add_argument("--num-cpus-pre-processing", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--attention-pad-id", default=0, type=int)
    args = parser.parse_args()

    base_model_name = args.base_model_name
    dataset_folder_name = args.dataset_folder_name
    num_cpus_pre_processing = args.num_cpus_pre_processing
    overwrite = args.overwrite
    attention_pad_id = args.attention_pad_id
    
    logger.info(f"Base model name: {base_model_name}")
    logger.info(f"Dataset folder name the final dataset will be saved too: {dataset_folder_name}")
    logger.info(f"Number CPUs for pre-processing: {num_cpus_pre_processing}")
    logger.info(f"Overwrite data: {overwrite}")
    logger.info(f"Attention pad id: {attention_pad_id}")
    dataset = VariableMosaicoUSASTraining(base_model_name, dataset_folder_name=dataset_folder_name, tokenizer_kwargs={"add_prefix_space": True}, batch_size=1, num_dataloader_cpus=1, num_cpus_pre_processing=num_cpus_pre_processing, overwrite_all_pre_processed_data=overwrite, attention_pad_id=attention_pad_id)
    logger.info("Starting to pre-process data")
    dataset.prepare_data()
    logger.info("Finished pre-processing data")