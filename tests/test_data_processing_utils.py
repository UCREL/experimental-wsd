from typing import Callable

import pytest
import wn
from transformers import AutoTokenizer
from wn.compat import sensekey

from experimental_wsd.data_processing_utils import (
    get_align_labels_with_tokens,
    map_token_sense_labels,
    map_token_text_and_is_content_labels,
    tokenize_pre_processing,
)

EN_LEXICON = "omw-en:1.4"
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand="")
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)


@pytest.mark.parametrize("label_pad_id", [-100, 50])
def test_get_align_labels_with_tokens(label_pad_id: int):
    expected_sub_word_labels = [label_pad_id, 0, 1, 1, 1, 1, 1, 1, 1, 0, label_pad_id]
    labels = [0, 1, 0]
    word_ids = [None, 0, 1, 1, 1, 1, 1, 1, 1, 2, None]
    sub_word_labels = get_align_labels_with_tokens(labels, word_ids, label_pad_id)
    assert expected_sub_word_labels == sub_word_labels


def test_map_token_text_and_is_content_labels():
    test_data = {
        "tokens": [
            {"raw": "This", "is_content_word": False},
            {"raw": "is", "is_content_word": False},
            {"raw": "a", "is_content_word": False},
            {"raw": "test", "is_content_word": True},
        ]
    }

    expected_output = {
        "text": ["This", "is", "a", "test"],
        "is_content_word": [0, 0, 0, 1],
    }

    assert expected_output == map_token_text_and_is_content_labels(test_data)


@pytest.mark.parametrize("align_labels_with_tokens", [False, True])
@pytest.mark.parametrize("label_pad_id", [-100, 50])
def test_tokenize_pre_processing(align_labels_with_tokens: bool, label_pad_id: int):
    test_data = {
        "text": [["This", "is", "a", "test"]],
        "is_content_word": [[0, 0, 0, 1]],
    }
    expected_output = {
        "input_ids": [[0, 152, 16, 10, 1296, 2]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]],
        "word_ids": [[label_pad_id, 0, 1, 2, 3, label_pad_id]],
        "labels": [[label_pad_id, 0, 0, 0, 1, label_pad_id]],
    }
    if not align_labels_with_tokens:
        expected_output["labels"] = [[0, 0, 0, 1]]
    tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/roberta-base", add_prefix_space=True
    )
    assert expected_output == tokenize_pre_processing(
        test_data, tokenizer, label_pad_id, align_labels_with_tokens
    )


@pytest.mark.parametrize("word_net_sense_getter", [None, GET_SENSE])
def test_map_token_sense_labels(
    word_net_sense_getter: Callable[[str], wn.Sense | None] | None,
):
    test_data = {
        "tokens": [
            {"raw": "This", "is_content_word": False},
            {"raw": "is", "is_content_word": False},
            {"raw": "a", "is_content_word": False},
            {"raw": "test", "is_content_word": True},
        ],
        "annotations": [
            {
                "lemma": "be",
                "pos": "n",
                "token_off": [1],
                "labels": ["become%2:42:01::", "improved%3:00:00::"],
            },
            {
                "lemma": "a_test",
                "pos": "v",
                "token_off": [2, 3],
                "labels": ["review%2:31:00::"],
            },
            {
                "lemma": "this_is_a",
                "pos": None,
                "token_off": [0, 1, 2],
                "labels": ["review%2:31:00::"],
            },
        ],
    }
    expected_output = {
        "text": ["This", "is", "a", "test"],
        "lemmas": ["be", "be", "a_test", "this_is_a"],
        "pos_tags": ["n", "n", "v", None],
        "token_offsets": [(1, 2), (1, 2), (2, 4), (0, 3)],
        "labels": [
            "become%2:42:01::",
            "improved%3:00:00::",
            "review%2:31:00::",
            "review%2:31:00::",
        ],
        "sense_labels": [
            "omw-en-become-02626604-v",
            "omw-en-improved-01288396-a",
            "omw-en-review-00696189-v",
            "omw-en-review-00696189-v",
        ],
    }
    if word_net_sense_getter is None:
        del expected_output["sense_labels"]

    assert expected_output == map_token_sense_labels(
        test_data, word_net_sense_getter=word_net_sense_getter
    )
