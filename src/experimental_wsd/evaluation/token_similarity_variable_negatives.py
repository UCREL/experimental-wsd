from typing import Callable

import torch
import wn
from transformers import PreTrainedTokenizerFast
from wn.compat import sensekey

from experimental_wsd.config import WSDDataDirectory
from experimental_wsd.evaluation.similarity_utils import (
    evaluate_macro_f1,
    evaluate_micro_f1,
    load_keys,
)
from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives
from experimental_wsd.pos_constants import SEMCOR_TO_WORDNET
from experimental_wsd.wordnet_utils import (
    get_definition,
    get_negative_wordnet_sense_ids,
    get_normalised_mwe_lemma_for_wordnet,
)
from experimental_wsd.wsd import wsd_sentence_generator


def evaluate_on_wordnet_dataset(
    model: TokenSimilarityVariableNegatives,
    dataset_directory: WSDDataDirectory,
    wordnet_sense_getter: Callable[[str], wn.Sense | None],
    tokenizer: PreTrainedTokenizerFast,
    wordnet: wn.Wordnet,
    wordnet_lexicon_name: str,
):
    wordnet_get_key = sensekey.sense_key_getter(wordnet_lexicon_name)
    prediction_keys = {}
    with torch.inference_mode(mode=True):
        for wsd_sample in wsd_sentence_generator(
            dataset_directory, wordnet_sense_getter
        ):
            text = wsd_sample.text
            text = [token.raw for token in wsd_sample.tokens]
            tokenized_text = tokenizer(
                text, return_tensors="pt", padding=True, is_split_into_words=True
            )
            text_input_ids = tokenized_text.input_ids
            text_attention_mask = tokenized_text.attention_mask
            text_input_ids = text_input_ids.to(device=model.device)
            text_attention_mask = text_attention_mask.to(device=model.device)

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
                text_word_ids_mask = text_word_ids_mask.unsqueeze(0).to(
                    device=model.device
                )
                if text_word_ids_mask.sum() == 0:
                    raise ValueError(
                        "Cannot find the token offsets in the given sample. "
                        f"Annotation: {annotation}"
                    )

                lemma = get_normalised_mwe_lemma_for_wordnet(annotation.lemma)
                pos_tag = SEMCOR_TO_WORDNET[annotation.pos]

                true_labels = [
                    wordnet_sense_getter(label).id for label in annotation.labels
                ]
                negative_labels = get_negative_wordnet_sense_ids(
                    lemma, pos_tag, true_labels, wordnet, get_random_sense=False
                )
                all_labels = true_labels + negative_labels
                all_label_definitions = []
                for label in all_labels:
                    all_label_definitions.append(get_definition(label, wordnet))

                tokenized_label_definitions = tokenizer(
                    all_label_definitions, return_tensors="pt", padding=True
                )
                label_definitions_input_ids = (
                    tokenized_label_definitions.input_ids.unsqueeze(0).to(
                        device=model.device
                    )
                )
                label_definitions_attention_mask = (
                    tokenized_label_definitions.attention_mask.unsqueeze(0).to(
                        device=model.device
                    )
                )

                out = model.forward(
                    text_input_ids,
                    text_attention_mask,
                    text_word_ids_mask,
                    label_definitions_input_ids,
                    label_definitions_attention_mask,
                )
                predicted_index = torch.argmax(out).item()
                predicted_label = all_labels[predicted_index]

                predicted_sense_key = wordnet_get_key(wordnet.sense(predicted_label))
                annotation_id = annotation.id
                prediction_keys[annotation_id] = [predicted_sense_key]
    gold_keys = load_keys(dataset_directory.gold.resolve())
    if len(gold_keys) != len(prediction_keys):
        raise ValueError(
            f"The number of predictions ({len(prediction_keys)}) "
            "is not the same as the number of gold labels "
            f"({len(gold_keys)})."
        )
    micro_f1 = evaluate_micro_f1(gold_keys, prediction_keys)
    macro_f1 = evaluate_macro_f1(gold_keys, prediction_keys)
    return micro_f1, macro_f1


def evaluate_on_usas_dataset(model: TokenSimilarityVariableNegatives) -> None:
    return None
