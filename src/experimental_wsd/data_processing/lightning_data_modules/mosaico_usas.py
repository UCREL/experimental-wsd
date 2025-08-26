import logging
from typing import Any
from uuid import uuid4
from pathlib import Path
from multiprocessing import Pool
import json


import lightning as L
from transformers import AutoTokenizer
import datasets
import torch

from experimental_wsd.config import MosaicoCoreUSAS, DATA_PROCESSING_DIR, USASMapper
from experimental_wsd.data_processing import processed_usas_utils
from experimental_wsd.data_processing.shared_token_utils import remove_duplicate_list_of_list_entries_while_maintaining_order, map_negative_usas_labels, usas_samples_to_single_sample, usas_join_positive_negative_labels, usas_map_to_definitions, tokenize_key, token_word_id_mask
from experimental_wsd.training_utils import collate_variable_token_similarity_dataset, DescendingTokenSimilarityBatchSampler

logger = logging.getLogger(__name__)


def file_exists(file_path: Path, overwrite: bool) -> bool:
    if file_path.exists() and not overwrite:
        return True
    return False

def write_to_json(data: Any, file_path: Path) -> None:
    with file_path.open('w', encoding="utf-8") as fp:
        json.dump(data, fp)

def read_json(file_path: Path) -> Any:
    with file_path.open('r', encoding="utf-8") as fp:
        return json.load(fp)

class VariableMosaicoUSASTraining(L.LightningDataModule):

    def __init__(self,
                 base_model_name: str,
                 tokenizer_kwargs: dict[str, Any],
                 batch_size: int,
                 num_cpus_pre_processing: int,
                 num_dataloader_cpus: int,
                 overwrite_all_pre_processed_data: bool = False,
                 attention_pad_id: int = 0,
                 ):
        """
        Args:
            num_dataloader_cpus (int): Number of CPU cores to use from loading 
                data from the Pytorch dataloaders, this applies to all dataloaders, 
                train and validation.
            lexicon_name (str): The name of the Wordnet lexicon to use, the 
                name also normally contains it's version number. The relevant 
                lexicon for SemCor and generally for training/testing 
                English Word Sense Disambiguation is WordNet version 3.0 of 
                which the equivalent for this within the `goodmami/wn` 
                open source repository is `omw-en:1.4` of which other 
                lexicons can be found here: 
                https://github.com/goodmami/wn?tab=readme-ov-file#available-wordnets 
                The lexicon associated with this name should have been downloaded 
                before running this by running the following: 
                `uv run python -m wn download omw-en:1.4`
            overwrite_all_pre_processed_data (bool): All pre-processed data is 
                cached within the directory specified by the 
                `EXPERIMENTAL_WSD_DATA_PROCESSING_DIR` environment variable. 
                If you would like the data to be overwritten then set this to 
                True, this can be useful if you need to debug something 
                and you think the problem is the data. Default False.
            generate_random_senses (bool): Whether to generate a random sense 
                for non-ambiguous terms. Default False.
            attention_pad_id (int): The integer ID that represents padding 
                for the attention mask. Default 0.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name,
                                                       **tokenizer_kwargs)
        self.batch_size = batch_size
        self.num_cpus_pre_processing = num_cpus_pre_processing
        self.num_dataloader_cpus = num_dataloader_cpus

        self.overwrite_all_pre_processed_data = overwrite_all_pre_processed_data
        
        processed_dataset_folder = Path(DATA_PROCESSING_DIR, "machine_learning_data")
        processed_dataset_folder.mkdir(exist_ok=True)
        self.processed_dataset_path = Path(processed_dataset_folder, str(uuid4()))
        self.attention_pad_id = attention_pad_id
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
        train_data_file_paths = MosaicoCoreUSAS.train
        validation_file_path =  MosaicoCoreUSAS.validation
        test_file_path =  MosaicoCoreUSAS.test
        all_file_paths = [*train_data_file_paths, validation_file_path, test_file_path]

        process_file_paths_arguments = [
            (file_path, DATA_PROCESSING_DIR, f"variable_mosaico_usas_{index}", self.overwrite_all_pre_processed_data) 
            for index, file_path in enumerate(all_file_paths)
        ]

        processed_file_paths: list[Path] | None = None
        with Pool(self.num_cpus_pre_processing)as pool:
            processed_file_paths = pool.starmap(processed_usas_utils.process_file, process_file_paths_arguments, chunksize=1)
        processed_file_paths_str = [str(file_path.resolve()) for file_path in processed_file_paths]
        training_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[:8])
        validation_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[8])
        test_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[9])
        
        usas_dataset = datasets.DatasetDict({"train": training_dataset["train"],
                                             "validation": validation_dataset["train"],
                                             "test": test_dataset["train"]})
        usas_dataset_de_duplicated = usas_dataset.map(remove_duplicate_list_of_list_entries_while_maintaining_order, fn_kwargs={"key": "usas"}, num_proc=self.num_cpus_pre_processing)
        
        usas_tag_to_description_mapper = processed_usas_utils.load_usas_mapper(USASMapper)
        
        training_label_statistics_file_path = Path(DATA_PROCESSING_DIR, f"variable_mosaico_usas_training_label_statistics.json")
        if not file_exists(training_label_statistics_file_path, self.overwrite_all_pre_processed_data):
            training_label_statistics = processed_usas_utils.get_usas_label_statistics(usas_dataset_de_duplicated["train"], usas_tag_to_description_mapper)
            write_to_json(training_label_statistics, training_label_statistics_file_path)
        training_label_statistics = read_json(training_label_statistics_file_path)
        training_log_inverse_label_statistics = processed_usas_utils.usas_inverse_label_statistics(training_label_statistics, log_scaled=2)
        training_inverse_label_statistics = processed_usas_utils.usas_inverse_label_statistics(training_label_statistics, log_scaled=None)

        # Here we are creating three negative samples per token:
        # The random negative sample comes from a weighted distribution that is inverse to the USAS label's frequency within the training dataset
        # Same as the first but log scaled whereby the log is to the base 2
        # The last is random sampling without any weighting
        negative_usas_dataset = usas_dataset_de_duplicated.map(map_negative_usas_labels, fn_kwargs={"positive_usas_key": "usas", "negative_usas_key": "negative_usas", "usas_weighting": training_log_inverse_label_statistics, "use_weights": True}, num_proc=self.num_cpus_pre_processing)
        negative_usas_dataset = negative_usas_dataset.map(map_negative_usas_labels, fn_kwargs={"positive_usas_key": "usas", "negative_usas_key": "negative_usas", "usas_weighting": training_inverse_label_statistics, "use_weights": True}, num_proc=self.num_cpus_pre_processing)
        negative_usas_dataset = negative_usas_dataset.map(map_negative_usas_labels, fn_kwargs={"positive_usas_key": "usas", "negative_usas_key": "negative_usas", "usas_weighting": training_inverse_label_statistics, "use_weights": False}, num_proc=self.num_cpus_pre_processing)
        
        # Remove columns that are not required
        removed_negative_usas_dataset = negative_usas_dataset.remove_columns(["lemmas", "pos", "is_content_token", "text"])
        removed_negative_usas_dataset = removed_negative_usas_dataset.rename_column("tokens", "text")
        sample_per_usas = removed_negative_usas_dataset.map(usas_samples_to_single_sample, batched=True, batch_size=20, num_proc=self.num_cpus_pre_processing)
        joined_usas_labels = sample_per_usas.map(usas_join_positive_negative_labels, fn_kwargs={"randomize": True}, batched=False, num_proc=self.num_cpus_pre_processing, remove_columns=["negative_usas", "usas"])
        mapped_usas_labels = joined_usas_labels.map(usas_map_to_definitions, fn_kwargs={"usas_mapper": usas_tag_to_description_mapper}, batched=False, num_proc=self.num_cpus_pre_processing)

        tokenized_text = mapped_usas_labels.map(
            tokenize_key,
            batched=True,
            fn_kwargs={"tokenizer": self.tokenizer,
                       "text_key": "text",
                       "output_key_prefix": "text",
                       "add_word_ids": True,
                       "is_split_into_words": True,
                       "truncation": False},
            batch_size=1000,
            num_proc=self.num_cpus_pre_processing
        )

        tokenized_definitions = tokenized_text.map(
            tokenize_key,
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            fn_kwargs={"tokenizer": self.tokenizer,
                       "text_key": "label_definitions",
                       "output_key_prefix": "label_definitions",
                       "add_word_ids": False,
                       "is_split_into_words": False,
                       "truncation": False}
        )

        text_token_masks = tokenized_definitions.map(token_word_id_mask,
                        batched=False,
                        num_proc=self.num_cpus_pre_processing,
                        fn_kwargs={"word_ids_key":"text_word_ids",
                                   "token_offsets_key":"usas_token_offsets",
                                   "word_id_mask_key":"text_word_ids_mask"})
        
        training_keys = [
            "text_input_ids",
            "text_attention_mask",
            "text_word_ids_mask",
            "label_definitions_input_ids",
            "label_definitions_attention_mask",
            "label_ids"
        ]
        keys_to_remove = [column_name for column_name in text_token_masks.column_names["train"] if column_name not in training_keys]
        final_dataset = text_token_masks.remove_columns(keys_to_remove)
        final_dataset.save_to_disk(self.processed_dataset_path)



    def setup(self, stage: str) -> None:
        if stage == "fit" or stage == "validate":
            train_validation_test_dataset = datasets.load_from_disk(self.processed_dataset_path)
            self.train = train_validation_test_dataset["train"]
            self.validation = train_validation_test_dataset["validation"]
        elif stage == "test":
            train_validation_test_dataset = datasets.load_from_disk(self.processed_dataset_path)
            self.test = train_validation_test_dataset["test"]


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        collate_fn = collate_variable_token_similarity_dataset(self.tokenizer, text_input_ids="text_input_ids", text_attention_mask="text_attention_mask",
                                                       text_word_ids_mask="text_word_ids_mask", similarity_sentence_input_ids="label_definitions_input_ids",
                                                       similarity_sentence_attention_mask="label_definitions_attention_mask", 
                                                       label_key="label_ids", attention_pad_id=self.attention_pad_id)
        training_sampler = DescendingTokenSimilarityBatchSampler(self.train, batch_size=self.batch_size, similarity_sentence_key="label_definitions_input_ids", random=True)
        training_dataloader = torch.utils.data.DataLoader(self.train, batch_sampler=training_sampler, collate_fn=collate_fn, num_workers=self.num_dataloader_cpus)
        return training_dataloader
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        collate_fn = collate_variable_token_similarity_dataset(self.tokenizer, text_input_ids="text_input_ids", text_attention_mask="text_attention_mask",
                                                       text_word_ids_mask="text_word_ids_mask", similarity_sentence_input_ids="label_definitions_input_ids",
                                                       similarity_sentence_attention_mask="label_definitions_attention_mask", 
                                                       label_key="label_ids", attention_pad_id=self.attention_pad_id)
        validation_sampler = DescendingTokenSimilarityBatchSampler(self.validation, batch_size=self.batch_size, similarity_sentence_key="label_definitions_input_ids", random=False)
        validation_dataloader = torch.utils.data.DataLoader(self.validation, batch_sampler=validation_sampler, collate_fn=collate_fn, num_workers=self.num_dataloader_cpus)
        return validation_dataloader
    
    def test_dataloader(self):
        collate_fn = collate_variable_token_similarity_dataset(self.tokenizer, text_input_ids="text_input_ids", text_attention_mask="text_attention_mask",
                                                       text_word_ids_mask="text_word_ids_mask", similarity_sentence_input_ids="label_definitions_input_ids",
                                                       similarity_sentence_attention_mask="label_definitions_attention_mask", 
                                                       label_key="label_ids", attention_pad_id=self.attention_pad_id)
        test_sampler = DescendingTokenSimilarityBatchSampler(self.test, batch_size=self.batch_size, similarity_sentence_key="label_definitions_input_ids", random=False)
        test_dataloader = torch.utils.data.DataLoader(self.test, batch_sampler=test_sampler, collate_fn=collate_fn, num_workers=self.num_dataloader_cpus)
        return test_dataloader