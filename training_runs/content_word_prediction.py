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
from experimental_wsd.data_processing_utils import get_pre_processed_label_statistics


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
#alt_tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/ettin-encoder-17m", add_prefix_space=True)


semcor_data_directory = RaganatoEnglish.semcor

EN_LEXICON = 'omw-en:1.4'
check_lexicon_exists(EN_LEXICON)
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand='')
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)

DATA_PROCESSING_DIR.mkdir(exist_ok=True)



semcor_jsonl_file_path = write_to_jsonl(wsd_sentence_generator(semcor_data_directory, GET_SENSE), DATA_PROCESSING_DIR, 'semcor', overwrite=False)
semcor_dataset = load_dataset("json", data_files=str(semcor_jsonl_file_path))['train']
semcor_dataset_splits = semcor_dataset.train_test_split(0.1)

wsl_ds = load_dataset("Babelscape/wsl", token=os.getenv("HF_TOKEN"))
wsl_ds_jsonl_file_path = write_to_jsonl(wsl_sentence_generator(wsl_ds, 'validation', word_net_sense_getter=GET_SENSE), DATA_PROCESSING_DIR, 'wsl_test', overwrite=False)
wsl_dataset = load_dataset("json", data_files=str(wsl_ds_jsonl_file_path))

test_is_content_word_dataset = wsl_dataset.map(map_token_text_and_is_content_labels, remove_columns=wsl_dataset['train'].column_names, batched=False)
test_labelled_content_word_dataset = test_is_content_word_dataset.map(tokenize_pre_processing, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=test_is_content_word_dataset['train'].column_names)


is_content_word_dataset = semcor_dataset_splits.map(map_token_text_and_is_content_labels, remove_columns=semcor_dataset_splits['train'].column_names, batched=False)
labelled_content_word_dataset = is_content_word_dataset.map(tokenize_pre_processing, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=is_content_word_dataset['train'].column_names)


train_label_counts = labelled_content_word_dataset.map(get_pre_processed_label_statistics, batched=True, fn_kwargs={"label_key": "labels", "label_value_to_ignore": -100}, remove_columns=labelled_content_word_dataset['train'].column_names)

label_counts = Counter(train_label_counts['train']['label_counts'])
neg_weight = label_counts[1] / label_counts[0]
pos_weight = label_counts[0] / label_counts[1]
label_weights = [neg_weight, pos_weight]

print(f"Label weights: {label_weights}")



#test_classifier = TokenClassifier("FacebookAI/roberta-base", True, number_transformer_encoder_layers=2, scalar_mix_layer_norm=True, label_weights=label_weights, classifier_dropout=0.1)
test_classifier = TokenClassifier(base_model_name, True, number_transformer_encoder_layers=0, scalar_mix_layer_norm=True, label_weights=label_weights, classifier_dropout=0.1)


_collate_function = collate_token_classification_dataset(tokenizer)
train_dataset = labelled_content_word_dataset['train']
train_batch_sampler = AscendingSequenceLengthBatchSampler(train_dataset, batch_size=128, length_key='input_ids', random=True)
train_dataset_loader = DataLoader(train_dataset, collate_fn=_collate_function, batch_sampler=train_batch_sampler, num_workers=6)

validation_dataset = labelled_content_word_dataset['test']
validation_batch_sampler = AscendingSequenceLengthBatchSampler(validation_dataset, batch_size=128, length_key='input_ids', random=False)
validation_dataset_loader = DataLoader(validation_dataset, collate_fn=_collate_function, batch_sampler=validation_batch_sampler, num_workers=6)

test_dataset = test_labelled_content_word_dataset['train']
test_batch_sampler = AscendingSequenceLengthBatchSampler(test_dataset, batch_size=128, length_key='input_ids', random=False)
test_dataset_loader = DataLoader(test_dataset, collate_fn=_collate_function, batch_sampler=test_batch_sampler, num_workers=6)

tracking_metric = "validation_macro_f1"
early_stopping_callback = EarlyStopping(monitor=tracking_metric, mode="max", check_finite=True, patience=3, verbose=True, strict=True)
checkpoint_file_name = "content-word-{epoch}-{" + f"{tracking_metric}" + ":.2f}"
model_checkpoint_callback = ModelCheckpoint(monitor=tracking_metric, save_top_k=1, mode='max', filename=checkpoint_file_name)
trainer = L.Trainer(callbacks=[early_stopping_callback, model_checkpoint_callback],
                    gradient_clip_algorithm="norm")
trainer.fit(model=test_classifier, train_dataloaders=train_dataset_loader, val_dataloaders=validation_dataset_loader)


#model_checkpoint_directory = Path("lightning_logs", "version_29", "checkpoints")
#model_checkpoint_file = get_only_checkpoint_path(model_checkpoint_directory)
model_checkpoint_file = model_checkpoint_callback.best_model_path
print(f"Checkpoint file: {model_checkpoint_file}")

#best_token_model = TokenClassifier.load_from_checkpoint(str(model_checkpoint_file.resolve()))
best_token_model = TokenClassifier.load_from_checkpoint(model_checkpoint_file)
test_trainer = L.Trainer()
test_trainer.test(best_token_model, dataloaders=test_dataset_loader)


# Access the best model through
# model_checkpoint_callback.best_model_path

#import time
#start_time = time.perf_counter()
#output = None
#for sample in train_dataset_loader:
#    test_classifier(sample['input_ids'], sample['attention_mask'], sample['word_ids'])
#    break
    #print(sample['input_ids'].shape)
    #input = [train_dataset[index] for index in sample]
    #print(input)
    #print(_collate_function(input))
    #print(len(sample['input_ids']))
    #break
#print(time.perf_counter() - start_time)

