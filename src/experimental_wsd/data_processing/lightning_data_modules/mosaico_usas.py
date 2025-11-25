import json
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import datasets
import lightning as L
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from experimental_wsd.config import DATA_PROCESSING_DIR, MosaicoCoreUSAS, USASMapper
from experimental_wsd.data_processing import processed_usas_utils
from experimental_wsd.data_processing.shared_token_utils import (
    filter_sequences_too_long,
    map_negative_usas_labels,
    remove_duplicate_list_of_list_entries_while_maintaining_order,
    token_word_id_mask,
    tokenize_key,
    usas_join_positive_negative_labels,
    usas_map_to_definitions,
    usas_samples_to_single_sample,
)
from experimental_wsd.training_utils import collate_variable_token_similarity_dataset

logger = logging.getLogger(__name__)


def file_exists(file_path: Path, overwrite: bool) -> bool:
    if file_path.exists() and not overwrite:
        return True
    return False


def write_to_json(data: Any, file_path: Path) -> None:
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)


def read_json(file_path: Path) -> Any:
    with file_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class VariableMosaicoUSASTraining(L.LightningDataModule):
    def __init__(
        self,
        base_model_name: str,
        tokenizer_kwargs: dict[str, Any],
        batch_size: int,
        num_cpus_pre_processing: int,
        num_dataloader_cpus: int,
        dataset_folder_name: str,
        overwrite_all_pre_processed_data: bool = False,
        attention_pad_id: int = 0,
        filter_out_labels: list[str] | None = None,
    ):
        """
        Args:
            base_model_name (str): The name of the base model to use for
                tokenization, should be a HuggingFace model ID e.g. `jhu-clsp/ettin-encoder-17m `.
            tokenizer_kwargs (dict[str, Any]): Keyword arguments to pass to
                the HuggingFace tokenizer.
            batch_size (int): The batch size to use for training.
            num_cpus_pre_processing (int): Number of CPU cores to use for
                pre-processing data.
            num_dataloader_cpus (int): Number of CPU cores to use from loading
                data from the Pytorch dataloaders, this applies to all dataloaders,
                train and validation.
            dataset_folder_name (str): The name of the folder to store the
                pre-processed data in. This folder will exist under the directory
                `$EXPERIMENTAL_WSD_DATA_PROCESSING_DIR/machine_learning_data/`.
                If the `$EXPERIMENTAL_WSD_DATA_PROCESSING_DIR` environment variable is not set,
                this will default to `$HOME/.cache/experimental_wsd/machine_learning_data`.
                This folder will be created if it does not exist.
            overwrite_all_pre_processed_data (bool): All pre-processed data is
                cached within the directory specified by the
                `EXPERIMENTAL_WSD_DATA_PROCESSING_DIR` environment variable.
                If you would like the data to be overwritten then set this to
                True, this can be useful if you need to debug something
                and you think the problem is the data. Default False.
            attention_pad_id (int): The integer ID that represents padding
                for the attention mask. Default 0.
            filter_out_labels (list[str] | None): A list of USAS labels that
                should not be in the data for training or evaluating. For example
                `[Z99]` would mean that the dataset will not contain any samples
                for that label, if None then no label filtering is applied. To
                NOTE if label filtering is applied then these filtered out labels
                will not appear as a negative training examples either.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, **tokenizer_kwargs
        )
        self.batch_size = batch_size
        self.num_cpus_pre_processing = num_cpus_pre_processing
        self.num_dataloader_cpus = num_dataloader_cpus

        self.overwrite_all_pre_processed_data = overwrite_all_pre_processed_data
        self.dataset_folder_name = dataset_folder_name

        processed_dataset_folder = Path(DATA_PROCESSING_DIR, "machine_learning_data")
        processed_dataset_folder.mkdir(exist_ok=True)
        self.processed_dataset_path = Path(
            processed_dataset_folder, self.dataset_folder_name
        )
        self.attention_pad_id = attention_pad_id
        self.filter_out_labels = filter_out_labels
        if filter_out_labels is None:
            self.filter_out_labels = set()
        self.save_hyperparameters()

    def prepare_data(self) -> None:
        if MosaicoCoreUSAS is None:
            error_message = (
                "Please follow the data download instructions and set the "
                "`MOSAICO_CORE_USAS` environment variable. "
                "If you have done that successfully then please ensure the "
                "environment variable is loaded before importing this file."
            )
            raise FileNotFoundError(error_message)
        if (
            self.processed_dataset_path.exists()
            and not self.overwrite_all_pre_processed_data
        ):
            return

        train_data_file_paths = MosaicoCoreUSAS.train
        validation_file_path = MosaicoCoreUSAS.validation
        test_file_path = MosaicoCoreUSAS.test
        all_file_paths = [*train_data_file_paths, validation_file_path, test_file_path]

        process_file_paths_arguments = [
            (
                file_path,
                DATA_PROCESSING_DIR,
                f"variable_mosaico_usas_{index}",
                self.overwrite_all_pre_processed_data,
            )
            for index, file_path in enumerate(all_file_paths)
        ]

        processed_file_paths: list[Path] | None = None
        with Pool(self.num_cpus_pre_processing) as pool:
            processed_file_paths = pool.starmap(
                processed_usas_utils.process_file,
                process_file_paths_arguments,
                chunksize=1,
            )
        processed_file_paths_str = [
            str(file_path.resolve()) for file_path in processed_file_paths
        ]
        # training_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[:8])
        training_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[0]
        )
        # validation_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[8])
        validation_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[8]
        )["train"].take(20000)
        # test_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[9])
        test_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[9]
        )["train"].take(20000)

        usas_dataset = datasets.DatasetDict(
            {
                "train": training_dataset["train"],
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )
        usas_dataset_de_duplicated = usas_dataset.map(
            remove_duplicate_list_of_list_entries_while_maintaining_order,
            fn_kwargs={"key": "usas", "tags_to_filter_out": self.filter_out_labels},
            num_proc=self.num_cpus_pre_processing,
        )

        usas_tag_to_description_mapper = processed_usas_utils.load_usas_mapper(
            USASMapper, self.filter_out_labels
        )
        training_label_statistics: dict[str, int] | None = None
        if not self.filter_out_labels:
            training_label_statistics_file_path = Path(
                DATA_PROCESSING_DIR,
                "variable_mosaico_usas_training_label_statistics.json",
            )
            if not file_exists(
                training_label_statistics_file_path,
                self.overwrite_all_pre_processed_data,
            ):
                training_label_statistics = (
                    processed_usas_utils.get_usas_label_statistics(
                        usas_dataset_de_duplicated["train"],
                        usas_tag_to_description_mapper,
                    )
                )
                write_to_json(
                    training_label_statistics, training_label_statistics_file_path
                )
            training_label_statistics = read_json(training_label_statistics_file_path)
        else:
            training_label_statistics = processed_usas_utils.get_usas_label_statistics(
                usas_dataset_de_duplicated["train"], usas_tag_to_description_mapper
            )
        training_log_inverse_label_statistics = (
            processed_usas_utils.usas_inverse_label_statistics(
                training_label_statistics, log_scaled=2
            )
        )
        training_inverse_label_statistics = (
            processed_usas_utils.usas_inverse_label_statistics(
                training_label_statistics, log_scaled=None
            )
        )

        # Here we are creating three negative samples per token:
        # The random negative sample comes from a weighted distribution that is inverse to the USAS label's frequency within the training dataset
        # Same as the first but log scaled whereby the log is to the base 2
        # The last is random sampling without any weighting
        negative_usas_dataset = usas_dataset_de_duplicated.map(
            map_negative_usas_labels,
            fn_kwargs={
                "positive_usas_key": "usas",
                "negative_usas_key": "negative_usas",
                "usas_weighting": training_log_inverse_label_statistics,
                "use_weights": True,
            },
            num_proc=self.num_cpus_pre_processing,
        )
        negative_usas_dataset = negative_usas_dataset.map(
            map_negative_usas_labels,
            fn_kwargs={
                "positive_usas_key": "usas",
                "negative_usas_key": "negative_usas",
                "usas_weighting": training_inverse_label_statistics,
                "use_weights": True,
            },
            num_proc=self.num_cpus_pre_processing,
        )
        negative_usas_dataset = negative_usas_dataset.map(
            map_negative_usas_labels,
            fn_kwargs={
                "positive_usas_key": "usas",
                "negative_usas_key": "negative_usas",
                "usas_weighting": training_inverse_label_statistics,
                "use_weights": False,
            },
            num_proc=self.num_cpus_pre_processing,
        )

        # Remove columns that are not required
        removed_negative_usas_dataset = negative_usas_dataset.remove_columns(
            ["lemmas", "pos", "is_content_token", "text"]
        )
        removed_negative_usas_dataset = removed_negative_usas_dataset.rename_column(
            "tokens", "text"
        )

        sample_per_usas = removed_negative_usas_dataset.map(
            usas_samples_to_single_sample,
            batched=True,
            batch_size=20,
            num_proc=self.num_cpus_pre_processing,
        )
        joined_usas_labels = sample_per_usas.map(
            usas_join_positive_negative_labels,
            fn_kwargs={"randomize": True},
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            remove_columns=["negative_usas", "usas"],
        )
        mapped_usas_labels = joined_usas_labels.map(
            usas_map_to_definitions,
            fn_kwargs={"usas_mapper": usas_tag_to_description_mapper},
            batched=False,
            num_proc=self.num_cpus_pre_processing,
        )

        tokenized_text = mapped_usas_labels.map(
            tokenize_key,
            batched=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_key": "text",
                "output_key_prefix": "text",
                "add_word_ids": True,
                "is_split_into_words": True,
                "truncation": False,
            },
            batch_size=1000,
            num_proc=self.num_cpus_pre_processing,
        )
        if self.tokenizer.model_max_length:
            model_max_length = self.tokenizer.model_max_length
            logger.info(
                "Filtering out text sequences that are longer than the "
                f"model's maximum sequence length: {model_max_length}"
            )
            number_filtered_samples = {
                split_name: len(split_dataset)
                for split_name, split_dataset in tokenized_text.items()
            }
            tokenized_text = tokenized_text.filter(
                filter_sequences_too_long,
                fn_kwargs={"key": "text_input_ids", "length": model_max_length},
                num_proc=self.num_cpus_pre_processing,
                batched=False,
            )
            for split_name, split_dataset in tokenized_text.items():
                samples_filtered = number_filtered_samples[split_name] - len(
                    split_dataset
                )
                logger.info(
                    f"{split_name}: Number of samples removed due to "
                    f"text sequence length: {samples_filtered:,}"
                )

        tokenized_definitions = tokenized_text.map(
            tokenize_key,
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_key": "label_definitions",
                "output_key_prefix": "label_definitions",
                "add_word_ids": False,
                "is_split_into_words": False,
                "truncation": False,
            },
        )

        text_token_masks = tokenized_definitions.map(
            token_word_id_mask,
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            fn_kwargs={
                "word_ids_key": "text_word_ids",
                "token_offsets_key": "usas_token_offsets",
                "word_id_mask_key": "text_word_ids_mask",
            },
        )

        training_keys = [
            "text_input_ids",
            "text_attention_mask",
            "text_word_ids_mask",
            "label_definitions_input_ids",
            "label_definitions_attention_mask",
            "label_ids",
        ]
        keys_to_remove = [
            column_name
            for column_name in text_token_masks.column_names["train"]
            if column_name not in training_keys
        ]
        final_dataset = text_token_masks.remove_columns(keys_to_remove)
        final_dataset.save_to_disk(self.processed_dataset_path)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            train_validation_test_dataset = datasets.load_from_disk(
                self.processed_dataset_path
            )
            self.train = train_validation_test_dataset["train"]
            self.validation = train_validation_test_dataset["validation"]
        elif stage == "test":
            train_validation_test_dataset = datasets.load_from_disk(
                self.processed_dataset_path
            )
            self.test = train_validation_test_dataset["test"]

    def train_dataloader(self) -> StatefulDataLoader:
        collate_fn = collate_variable_token_similarity_dataset(
            self.tokenizer,
            text_input_ids="text_input_ids",
            text_attention_mask="text_attention_mask",
            text_word_ids_mask="text_word_ids_mask",
            similarity_sentence_input_ids="label_definitions_input_ids",
            similarity_sentence_attention_mask="label_definitions_attention_mask",
            label_key="label_ids",
            attention_pad_id=self.attention_pad_id,
        )
        training_dataloader = StatefulDataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_dataloader_cpus,
            pin_memory=True,
        )
        return training_dataloader

    def val_dataloader(self) -> StatefulDataLoader:
        collate_fn = collate_variable_token_similarity_dataset(
            self.tokenizer,
            text_input_ids="text_input_ids",
            text_attention_mask="text_attention_mask",
            text_word_ids_mask="text_word_ids_mask",
            similarity_sentence_input_ids="label_definitions_input_ids",
            similarity_sentence_attention_mask="label_definitions_attention_mask",
            label_key="label_ids",
            attention_pad_id=self.attention_pad_id,
        )
        validation_dataloader = StatefulDataLoader(
            self.validation,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_dataloader_cpus,
            pin_memory=True,
        )
        return validation_dataloader

    def test_dataloader(self) -> StatefulDataLoader:
        collate_fn = collate_variable_token_similarity_dataset(
            self.tokenizer,
            text_input_ids="text_input_ids",
            text_attention_mask="text_attention_mask",
            text_word_ids_mask="text_word_ids_mask",
            similarity_sentence_input_ids="label_definitions_input_ids",
            similarity_sentence_attention_mask="label_definitions_attention_mask",
            label_key="label_ids",
            attention_pad_id=self.attention_pad_id,
        )
        test_dataloader = StatefulDataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_dataloader_cpus,
        )
        return test_dataloader
