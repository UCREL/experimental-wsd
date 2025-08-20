import logging
from typing import Any
from uuid import uuid4
from pathlib import Path

import wn
from wn.compat import sensekey
import lightning as L
from transformers import AutoTokenizer
import datasets
import torch

from experimental_wsd.config import RaganatoEnglish, DATA_PROCESSING_DIR
from experimental_wsd.wordnet_utils import check_lexicon_exists
from experimental_wsd.wsd import wsd_sentence_generator
from experimental_wsd.training_utils import write_to_jsonl, collate_variable_token_similarity_dataset, DescendingTokenSimilarityBatchSampler
from experimental_wsd.data_processing.shared_token_utils import map_token_sense_labels, map_negative_sense_ids, sample_to_a_sense, filter_empty_values, join_positive_negative_labels, map_to_definitions, tokenize_key, token_word_id_mask
from experimental_wsd.pos_constants import SEMCOR_TO_WORDNET

logger = logging.getLogger(__name__)

class VariableSemCorTraining(L.LightningDataModule):

    def __init__(self,
                 base_model_name: str,
                 tokenizer_kwargs: dict[str, Any],
                 batch_size: int,
                 num_cpus_pre_processing: int,
                 num_dataloader_cpus: int,
                 lexicon_name: str = "omw-en:1.4",
                 overwrite_all_pre_processed_data: bool = False,
                 generate_random_senses: bool = False,
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

        self.lexicon_name = lexicon_name
        check_lexicon_exists(self.lexicon_name)
        self.wordnet = wn.Wordnet(lexicon=self.lexicon_name, expand="")
        self.get_sense = sensekey.sense_getter(self.lexicon_name, self.wordnet)
        self.overwrite_all_pre_processed_data = overwrite_all_pre_processed_data
        self.generate_random_senses = generate_random_senses
        
        processed_dataset_folder = Path(DATA_PROCESSING_DIR, "machine_learning_data")
        processed_dataset_folder.mkdir(exist_ok=True)
        self.processed_dataset_path = Path(processed_dataset_folder, str(uuid4()))
        self.attention_pad_id = attention_pad_id
        self.save_hyperparameters()
        
    

    def prepare_data(self) -> None:
        if RaganatoEnglish is None:
            error_message = (
                "Please follow the data download instructions and set the "
                "`ENGLISH_RAGANATO` environment variable, this should have "
                "already been done for you through the download script "
                "if you have done that successfully then pleas ensure the "
                "environment variable is loaded before importing this file."
            )
            raise FileNotFoundError(error_message)
        semcor_jsonl_file_path = write_to_jsonl(
            wsd_sentence_generator(RaganatoEnglish.semcor, self.get_sense),
            DATA_PROCESSING_DIR,
            f"semcor_{self.lexicon_name}",
            overwrite=self.overwrite_all_pre_processed_data,
            )
        
        raganato_validation_jsonl_file_path = write_to_jsonl(
            wsd_sentence_generator(RaganatoEnglish.semeval_2007, self.get_sense),
            DATA_PROCESSING_DIR,
            f"raganato_semeval_2007_{self.lexicon_name}",
            overwrite=self.overwrite_all_pre_processed_data,
            )
        
        semcor_dataset = datasets.load_dataset("json",
                                      data_files=str(semcor_jsonl_file_path))
        raganato_validation_dataset = datasets.load_dataset(
            "json", data_files=str(raganato_validation_jsonl_file_path)
        )

        training_dataset = datasets.DatasetDict({
            "train": semcor_dataset["train"],
            "validation": raganato_validation_dataset["train"]
                                                })
        token_sense_labels = training_dataset.map(
            map_token_sense_labels,
            remove_columns=training_dataset["train"].column_names,
            batched=False,
            fn_kwargs={"word_net_sense_getter": self.get_sense},
            num_proc=self.num_cpus_pre_processing
        )
        token_negative_sense_labels = token_sense_labels.map(
            map_negative_sense_ids,
            batched=False,
            fn_kwargs={"word_net_lexicon": self.wordnet,
                    "sense_id_key": "sense_labels",
                    "lemma_key": "lemmas",
                    "pos_tag_key": "pos_tags",
                    "negative_sense_id_key": "negative_labels",
                    "pos_tag_mapper": SEMCOR_TO_WORDNET,
                    "normalise_mwe_lemma": True,
                    "get_random_sense": self.generate_random_senses},
            num_proc=self.num_cpus_pre_processing
        )
        flattened_sense_labels = token_negative_sense_labels.map(
            sample_to_a_sense,
            batched=True
        )
        # Remove samples that do not have a negative label
        filtered_flattened_sense_labels = flattened_sense_labels.filter(filter_empty_values, fn_kwargs={"key": "negative_labels"})
        # The negative and positive sense ids are joined as well as a key-value 
        # pair determining which sense id in the sense ids is the positive one.
        joined_sense_labels = filtered_flattened_sense_labels.map(
            join_positive_negative_labels,
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            fn_kwargs={"randomize": True},
            remove_columns=["labels", "sense_labels", "negative_labels"]
        )
    
        mapped_sense_labels = joined_sense_labels.map(
            map_to_definitions,
            batched=False,
            num_proc=self.num_cpus_pre_processing,
            fn_kwargs={"sense_key": "label_sense_ids", 
                       "word_net_lexicon": self.wordnet, 
                       "definition_key": "label_definitions"},
            remove_columns=["label_sense_ids"]
        )

        tokenized_text = mapped_sense_labels.map(
            tokenize_key,
            batched=True,
            fn_kwargs={"tokenizer": self.tokenizer,
                       "text_key": "text",
                       "output_key_prefix": "text",
                       "add_word_ids": True,
                       "is_split_into_words": True,
                       "truncation": False}
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
                                   "token_offsets_key":"token_offsets",
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
            train_validation_dataset = datasets.load_from_disk(self.processed_dataset_path)
            self.train = train_validation_dataset["train"]
            self.validation = train_validation_dataset["validation"]

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