import wn
from wn.compat import sensekey
from collections import Counter
import os
from pathlib import Path

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

from dotenv import load_dotenv
load_dotenv()
from experimental_wsd.config import RaganatoEnglish, DATA_PROCESSING_DIR
from experimental_wsd.wsd import wsd_sentence_generator
from experimental_wsd.wsl import wsl_sentence_generator
from experimental_wsd.wordnet_utils import check_lexicon_exists
from experimental_wsd.training_utils import get_prefix_suffix_special_token_indexes
from experimental_wsd.training_utils import write_to_jsonl
from experimental_wsd.data_processing_utils import map_token_text_and_is_content_labels, tokenize_pre_processing
from experimental_wsd.nn.token_classifier import TokenClassifier
from torch.utils.data import DataLoader
from experimental_wsd.training_utils import AscendingSequenceLengthBatchSampler, collate_token_classification_dataset
from experimental_wsd.data_processing_utils import get_pre_processed_label_statistics, map_token_sense_labels


def get_only_checkpoint_path(model_checkpoint_directory: Path) -> Path:
    relevant_paths: list[Path] = []
    for a_path in model_checkpoint_directory.iterdir():
        if a_path.suffix == '.ckpt':
            relevant_paths.append(a_path)
    if len(relevant_paths) != 1:
        raise FileNotFoundError("Could not find the only model checkpoint "
                                "path in the model's checkpoint directory: "
                                f"{model_checkpoint_directory} the number of "
                                "relevant paths found should be only 1 we "
                                f"found: {relevant_paths}")
    return relevant_paths[0]


base_model_name = "FacebookAI/roberta-base" # "jhu-clsp/ettin-encoder-17m"
base_model_name = "jhu-clsp/ettin-encoder-17m" # "jhu-clsp/ettin-encoder-17m"
#tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)

#print(tokenizer.decode([50281,  1056,  3806,   281, 50282, 50283, 50283, 50283, 50283, 50283,
#          50283, 50283, 50283]))

#print()

#print(tokenizer.decode([50281,   452,   347,   247,  4495, 50282]))
#print()
#import pdb
#pdb.set_trace()

#alt_tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-17m", add_prefix_space=True)




semcor_data_directory = RaganatoEnglish.semcor

EN_LEXICON = 'omw-en:1.4'
check_lexicon_exists(EN_LEXICON)
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand='')
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)

DATA_PROCESSING_DIR.mkdir(exist_ok=True)


import datasets

def get_label_statistics(
    hf_dataset: datasets.Dataset,
    label_key: str,
    label_values_to_ignore: list[str | int] | None = None
) -> dict[str | int, int]:
    """
    Given a dataset and the key associated to the label value it will return 
    a dictionary of label value to label count whereby the label value 
    can be a string or an integer.

    Args:
        hf_dataset (datasets.Dataset): The dataset that contains the labelled data.
        label_key (str): The key that represents the labelled data.
        label_values_to_ignore (list[str, int] | None): The list of label values 
            to ignore, if None no label values will be ignored. Default None.
    Returns:
        dict[str | int, int]: A dictionary of label values to counts.
    """
    label_counter = Counter()
    for sample in hf_dataset:
        label_value = sample[label_key]
        if isinstance(label_value, list):
            label_counter.update(label_value)
        else:
            label_counter.update([label_value])
    if label_values_to_ignore is not None:
        for label_value_to_ignore in label_values_to_ignore:
            if label_value_to_ignore in label_counter:
                del label_counter[label_value_to_ignore]
    return label_counter

import wn
from experimental_wsd.data_processing_utils import map_negative_sense_ids, map_to_definitions, tokenize_key, token_word_id_mask
from experimental_wsd.pos_constants import SEMCOR_TO_WORDNET
from experimental_wsd.data_processing_utils import filter_empty_values, map_empty_removal_token_values
from experimental_wsd.training_utils import collate_token_negative_examples_classification_dataset, AscendingTokenNegativeExamplesBatchSampler
from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives


_collate_function = collate_token_negative_examples_classification_dataset(tokenizer)
            





semcor_jsonl_file_path = write_to_jsonl(wsd_sentence_generator(semcor_data_directory, GET_SENSE), DATA_PROCESSING_DIR, 'semcor', overwrite=False)
semcor_dataset = load_dataset("json", data_files=str(semcor_jsonl_file_path))['train'].take(1000)
semcor_dataset_splits = semcor_dataset.train_test_split(0.1)


token_sense_labels = semcor_dataset_splits.map(map_token_sense_labels, remove_columns=semcor_dataset_splits['train'].column_names, batched=False, fn_kwargs={"word_net_sense_getter": GET_SENSE}, num_proc=8)
#t = get_label_statistics(token_sense_labels['train'], "labels")

raganato_validation_data = RaganatoEnglish.semeval_2007
raganato_validation_jsonl_file_path = write_to_jsonl(wsd_sentence_generator(raganato_validation_data, GET_SENSE), DATA_PROCESSING_DIR, 'raganato_validation', overwrite=False)
raganato_validation_dataset = load_dataset("json", data_files=str(raganato_validation_jsonl_file_path))['train']
validation_token_sense_labels = raganato_validation_dataset.map(map_token_sense_labels, remove_columns=raganato_validation_dataset.column_names, batched=False, fn_kwargs={"word_net_sense_getter": GET_SENSE}, num_proc=8)
from collections import Counter
pos_tag_counter = Counter()
for sample in validation_token_sense_labels:
        pos_tag_counter.update(sample['pos_tags'])
print(f"POS Tag counter in the validation dataset: {pos_tag_counter}")

validation_token_sense_neg_labels = validation_token_sense_labels.map(map_negative_sense_ids, num_proc=8, fn_kwargs={"sense_id_key": "sense_labels", "word_net_lexicon": ENGLISH_WN, "lemma_key": "lemmas", "pos_tag_key": "pos_tags", "negative_sense_id_key": "negative_labels", "pos_tag_mapper": SEMCOR_TO_WORDNET, "normalise_mwe_lemma": True})
validation_token_sense_neg_labels_filtered = validation_token_sense_neg_labels.map(map_empty_removal_token_values, num_proc=8,  fn_kwargs={"filter_key": "negative_labels", "aligned_keys": ["labels", "sense_labels", "lemmas", "pos_tags", "token_offsets"]})
validation_token_sense_neg_labels_filtered = validation_token_sense_neg_labels_filtered.filter(filter_empty_values, num_proc=8,  fn_kwargs={"key": "labels"})

validation_token_definitions = validation_token_sense_neg_labels_filtered.map(map_to_definitions, num_proc=8,  fn_kwargs={"word_net_lexicon": ENGLISH_WN, "sense_key": "sense_labels", "definition_key": "sense_definitions"})
validation_token_definitions = validation_token_definitions.map(map_to_definitions, num_proc=8,  fn_kwargs={"word_net_lexicon": ENGLISH_WN, "sense_key": "negative_labels", "definition_key": "negative_definitions"})


validation_tokenized_definitions = validation_token_definitions.map(tokenize_key, fn_kwargs={"tokenizer": tokenizer, "text_key": "sense_definitions", "output_key_prefix": "positive", "add_word_ids": False})
validation_tokenized_definitions = validation_tokenized_definitions.map(tokenize_key, fn_kwargs={"tokenizer": tokenizer, "text_key": "negative_definitions", "output_key_prefix": "negative", "add_word_ids": False})
validation_tokenized_definitions = validation_tokenized_definitions.map(tokenize_key, batched=True, fn_kwargs={"tokenizer": tokenizer, "text_key": "text", "output_key_prefix": "text", "add_word_ids": True, "is_split_into_words": True})
validation_tokenized_definitions = validation_tokenized_definitions.map(token_word_id_mask, batched=False, fn_kwargs={"word_ids_key": "text_word_ids", "token_offsets_key": "token_offsets", "word_id_mask_key": "text_word_ids_mask"})

test_batch_sampler = AscendingTokenNegativeExamplesBatchSampler(validation_tokenized_definitions, batch_size=1, positive_sample_key="positive_input_ids", negative_sample_key="negative_input_ids", random=False)
test_dataset_loader = DataLoader(validation_tokenized_definitions, collate_fn=_collate_function, batch_sampler=test_batch_sampler, num_workers=1)



pos_tag_counter = Counter()
splits = ['train', 'test']
for split in splits:
    for sample in token_sense_labels[split]:
        pos_tag_counter.update(sample['pos_tags'])
print(f"POS Tag counter in the training dataset: {pos_tag_counter}")



token_sense_neg_labels = token_sense_labels.map(map_negative_sense_ids, num_proc=8, fn_kwargs={"sense_id_key": "sense_labels", "word_net_lexicon": ENGLISH_WN, "lemma_key": "lemmas", "pos_tag_key": "pos_tags", "negative_sense_id_key": "negative_labels", "pos_tag_mapper": SEMCOR_TO_WORDNET, "normalise_mwe_lemma": True})
token_sense_neg_labels_filtered = token_sense_neg_labels.map(map_empty_removal_token_values, num_proc=8,  fn_kwargs={"filter_key": "negative_labels", "aligned_keys": ["labels", "sense_labels", "lemmas", "pos_tags", "token_offsets"]})
token_sense_neg_labels_filtered = token_sense_neg_labels_filtered.filter(filter_empty_values, num_proc=8,  fn_kwargs={"key": "labels"})

token_definitions = token_sense_neg_labels_filtered.map(map_to_definitions, num_proc=8,  fn_kwargs={"word_net_lexicon": ENGLISH_WN, "sense_key": "sense_labels", "definition_key": "sense_definitions"})
token_definitions = token_definitions.map(map_to_definitions, num_proc=8,  fn_kwargs={"word_net_lexicon": ENGLISH_WN, "sense_key": "negative_labels", "definition_key": "negative_definitions"})


tokenized_definitions = token_definitions.map(tokenize_key, fn_kwargs={"tokenizer": tokenizer, "text_key": "sense_definitions", "output_key_prefix": "positive", "add_word_ids": False})
tokenized_definitions = tokenized_definitions.map(tokenize_key, fn_kwargs={"tokenizer": tokenizer, "text_key": "negative_definitions", "output_key_prefix": "negative", "add_word_ids": False})
tokenized_definitions = tokenized_definitions.map(tokenize_key, batched=True, fn_kwargs={"tokenizer": tokenizer, "text_key": "text", "output_key_prefix": "text", "add_word_ids": True, "is_split_into_words": True})
tokenized_definitions = tokenized_definitions.map(token_word_id_mask, batched=False, fn_kwargs={"word_ids_key": "text_word_ids", "token_offsets_key": "token_offsets", "word_id_mask_key": "text_word_ids_mask"})

text_input_ids = "text_input_ids"
text_attention_ids = "text_attention_mask"
text_word_id_mask = "text_word_ids_mask"
positive_input_ids = "positive_input_ids"
positive_attention_ids = "positive_attention_mask"
negative_input_ids = "negative_input_ids"
negative_attention_ids = "negative_attention_mask"


columns_required = [
    text_input_ids,
    text_attention_ids,
    text_word_id_mask,
    positive_input_ids,
    positive_attention_ids,
    negative_input_ids,
    negative_attention_ids
]
columns_to_drop = [column_name for column_name in tokenized_definitions['train'].column_names if column_name not in columns_required]

learning_data = tokenized_definitions.remove_columns(columns_to_drop)

for key, value in learning_data['train'][0].items():
    print(f'{key}: {value}')
    

positive_senses_count = Counter()
negative_senses_count = Counter()
number_of_empty_negatives = 0
total_samples = 0
for split in splits:
    for sample in token_sense_neg_labels[split]:
        positive_senses_count.update(sample["labels"])
        for negative_labels in sample["negative_labels"]:
            if not negative_labels:
                number_of_empty_negatives += 1
            else:
                negative_senses_count.update(negative_labels)
            total_samples += 1

print(sum(positive_senses_count.values()))
print(sum(negative_senses_count.values()))
print(number_of_empty_negatives)
print(total_samples)
print((number_of_empty_negatives / total_samples) * 100)





train_dataset = learning_data['train']
train_batch_sampler = AscendingTokenNegativeExamplesBatchSampler(train_dataset, batch_size=1, positive_sample_key="positive_input_ids", negative_sample_key="negative_input_ids", random=True)
train_dataset_loader = DataLoader(train_dataset, collate_fn=_collate_function, batch_sampler=train_batch_sampler, num_workers=1)

validation_dataset = learning_data['test']
validation_batch_sampler = AscendingTokenNegativeExamplesBatchSampler(validation_dataset, batch_size=1, positive_sample_key="positive_input_ids", negative_sample_key="negative_input_ids", random=True)
validation_dataset_loader = DataLoader(validation_dataset, collate_fn=_collate_function, batch_sampler=validation_batch_sampler, num_workers=1)


test_classifier = TokenSimilarityVariableNegatives(base_model_name, freeze_base_model=True, number_transformer_encoder_layers=2, transformer_encoder_hidden_dim=256, transformer_encoder_num_heads=4)
tracking_metric = "validation_micro_f1"
early_stopping_callback = EarlyStopping(monitor=tracking_metric, mode="max", check_finite=True, patience=3, verbose=True, strict=True)
checkpoint_file_name = "token-sense-{epoch}-{" + f"{tracking_metric}" + ":.2f}"
model_checkpoint_callback = ModelCheckpoint(monitor=tracking_metric, save_top_k=1, mode='max', filename=checkpoint_file_name)
#trainer = L.Trainer(callbacks=[early_stopping_callback, model_checkpoint_callback],
#                    gradient_clip_algorithm="norm", accumulate_grad_batches=32, detect_anomaly=True)
#trainer.fit(model=test_classifier, train_dataloaders=train_dataset_loader, val_dataloaders=validation_dataset_loader)


#model_checkpoint_file = model_checkpoint_callback.best_model_path
#print(f"Checkpoint file: {model_checkpoint_file}")

model_checkpoint_directory = Path("/", "workspaces", "experimental-wsd", "lightning_logs", "version_122", "checkpoints")
model_checkpoint_file = get_only_checkpoint_path(model_checkpoint_directory)



best_token_model = TokenSimilarityVariableNegatives.load_from_checkpoint(model_checkpoint_file)
test_trainer = L.Trainer()
test_trainer.test(best_token_model, dataloaders=test_dataset_loader)

#labelled_content_word_dataset = is_content_word_dataset.map(tokenize_pre_processing, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=is_content_word_dataset['train'].column_names)


#train_label_counts = labelled_content_word_dataset.map(get_pre_processed_label_statistics, batched=True, fn_kwargs={"label_key": "labels", "label_value_to_ignore": -100}, remove_columns=labelled_content_word_dataset['train'].column_names)

#label_counts = Counter(train_label_counts['train']['label_counts'])
#neg_weight = label_counts[1] / label_counts[0]
#pos_weight = label_counts[0] / label_counts[1]
#label_weights = [neg_weight, pos_weight]

#print(f"Label weights: {label_weights}")



#test_classifier = TokenClassifier("FacebookAI/roberta-base", True, number_transformer_encoder_layers=2, scalar_mix_layer_norm=True, label_weights=label_weights, classifier_dropout=0.1)
#test_classifier = TokenClassifier(base_model_name, True, number_transformer_encoder_layers=0, scalar_mix_layer_norm=True, label_weights=label_weights, classifier_dropout=0.1)


#_collate_function = collate_token_classification_dataset(tokenizer)
#train_dataset = labelled_content_word_dataset['train']
#train_batch_sampler = AscendingSequenceLengthBatchSampler(train_dataset, batch_size=128, length_key='input_ids', random=True)
#train_dataset_loader = DataLoader(train_dataset, collate_fn=_collate_function, batch_sampler=train_batch_sampler, num_workers=6)


