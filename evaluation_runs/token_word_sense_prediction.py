from dotenv import load_dotenv
from collections import Counter
load_dotenv()

from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives
from experimental_wsd.config import RaganatoEnglish
from experimental_wsd.wordnet_utils import get_definition, check_lexicon_exists, get_negative_wordnet_sense_ids, get_normalised_mwe_lemma_for_wordnet
from experimental_wsd.wsd import wsd_sentence_generator
from experimental_wsd.pos_constants import SEMCOR_TO_WORDNET
from pathlib import Path
import torch
import wn
from wn.compat import sensekey
from transformers import AutoTokenizer


#from evaluate_macro_f1 import load_keys, evaluate
from evaluate_micro_f1 import load_keys, evaluate



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

EN_LEXICON = 'omw-en:1.4'
check_lexicon_exists(EN_LEXICON)
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand='')
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)
GET_KEY = sensekey.sense_key_getter(EN_LEXICON)

raganato_validation_data = RaganatoEnglish.semeval_2007

base_model_name = "jhu-clsp/ettin-encoder-17m"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, add_prefix_space=True)

model_checkpoint_directory = Path("/", "workspaces", "experimental-wsd", "lightning_logs", "version_122", "checkpoints")
model_checkpoint_file = get_only_checkpoint_path(model_checkpoint_directory)
best_token_model = TokenSimilarityVariableNegatives.load_from_checkpoint(model_checkpoint_file)
best_token_model.eval()


pred_keys = {}

number_of_labels_to_predict = Counter()
predicted_index_counter = Counter()

with torch.inference_mode(mode=True):
    for wsd_sample in wsd_sentence_generator(raganato_validation_data, GET_SENSE):
        text = wsd_sample.text
        text = [token.raw for token in wsd_sample.tokens]
        tokenized_text = tokenizer(text, return_tensors="pt", padding=True, is_split_into_words=True)
        text_input_ids = tokenized_text.input_ids
        text_attention_mask = tokenized_text.attention_mask
        text_input_ids = text_input_ids.to(device=best_token_model.device)
        text_attention_mask = text_attention_mask.to(device=best_token_model.device)

        for annotation in wsd_sample.annotations:
            token_offsets = set(annotation.token_off)

            text_word_ids_mask = []
            for word_id in tokenized_text.word_ids():
                if word_id is None:
                    text_word_ids_mask.append(0)
                elif word_id in token_offsets:
                    text_word_ids_mask.append(1)
                else:
                    text_word_ids_mask.append(0)
            text_word_ids_mask = torch.tensor(text_word_ids_mask, dtype=torch.long)
            text_word_ids_mask = text_word_ids_mask.unsqueeze(0).unsqueeze(0).to(device=best_token_model.device)
            if text_word_ids_mask.sum() == 0:
                raise ValueError("Cannot find the token offsets in the given sample. "
                                f"Annotation: {annotation}")

            

            lemma = get_normalised_mwe_lemma_for_wordnet(annotation.lemma)
            pos_tag = SEMCOR_TO_WORDNET[annotation.pos]
        

            true_labels = annotation.labels
            labels = set()
            for true_label in true_labels:
                true_label_wordnet_id = GET_SENSE(true_label).id
                labels.add(true_label_wordnet_id)
                negative_labels = get_negative_wordnet_sense_ids(lemma, pos_tag, true_label_wordnet_id, ENGLISH_WN, get_random_sense=False)
                for negative_label in negative_labels:
                    labels.add(negative_label)
            all_labels = list(labels)
            all_label_definitions = []
            true_label_index = 0
            for label in all_labels:
                all_label_definitions.append(get_definition(label, ENGLISH_WN))
            number_of_labels_to_predict.update([len(all_labels) - len(true_labels)])

            positive_label_definition = all_label_definitions.pop(0)

            positive_tokenized_label_definitions = tokenizer(positive_label_definition, return_tensors="pt", padding=True)
            positive_tokenized_inputs = positive_tokenized_label_definitions.input_ids
            positive_tokenized_attention_mask = positive_tokenized_label_definitions.attention_mask

            # Handle the case when there is only one label option
            if len(all_label_definitions) == 0:
                predicted_sense_key = GET_KEY(ENGLISH_WN.sense(all_labels[0]))
                annotation_id = annotation.id
                pred_keys[annotation_id] = [predicted_sense_key]
                continue

            negative_tokenized_label_definitions = tokenizer(all_label_definitions, return_tensors="pt", padding=True)
            negative_tokenized_inputs = negative_tokenized_label_definitions.input_ids
            negative_tokenized_attention_mask = negative_tokenized_label_definitions.attention_mask

            negative_tokenized_inputs = negative_tokenized_inputs.unsqueeze(0).unsqueeze(0).to(device=best_token_model.device)
            negative_tokenized_attention_mask = negative_tokenized_attention_mask.unsqueeze(0).unsqueeze(0).to(device=best_token_model.device)

            positive_tokenized_inputs = positive_tokenized_inputs.unsqueeze(0).to(device=best_token_model.device)
            positive_tokenized_attention_mask = positive_tokenized_attention_mask.unsqueeze(0).to(device=best_token_model.device)


            out = best_token_model.forward(positive_tokenized_inputs, positive_tokenized_attention_mask, negative_tokenized_inputs, negative_tokenized_attention_mask, text_input_ids, text_attention_mask, text_word_ids_mask)
            
            predicted_index = torch.argmax(out).item()
            predicted_index_counter.update([predicted_index])
            predicted_label = all_labels[predicted_index]
            predicted_sense_key = GET_KEY(ENGLISH_WN.sense(predicted_label))
            annotation_id = annotation.id
            pred_keys[annotation_id] = [predicted_sense_key]

            

validation_gold_file_path = str(raganato_validation_data.gold.resolve())
validation_gold_keys = load_keys(validation_gold_file_path)
print(f"Number of gold annotations: {len(validation_gold_keys)}")
print(f"Number of prediction annotations: {len(pred_keys)}")
print(number_of_labels_to_predict)
number_non_determinstic_labels = number_of_labels_to_predict[0]
print(f"Number of non determinstic labels: {number_non_determinstic_labels}")
percentage_non_determinstic_labels = (number_non_determinstic_labels / len(validation_gold_keys)) * 100
print(f"Percentage of non-determinstic labels: {percentage_non_determinstic_labels:.2f}%")


evaluate(validation_gold_keys, pred_keys)

print(predicted_index_counter)
import pdb
pdb.set_trace()





#with torch.inference_mode(mode=True):

#    best_token_model()

#test_trainer = L.Trainer()
#test_trainer.test(best_token_model, dataloaders=test_dataset_loader)