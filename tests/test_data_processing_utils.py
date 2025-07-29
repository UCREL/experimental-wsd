import pytest
from transformers import AutoTokenizer

from experimental_wsd.data_processing_utils import (
    get_align_labels_with_tokens,
    map_token_text_and_is_content_labels,
    tokenize_pre_processing,
    map_token_sense_indexes
)


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

def test_map_token_sense_indexes():
    test_data = {
        "tokens": [
            {"raw": "This", "is_content_word": False},
            {"raw": "is", "is_content_word": False},
            {"raw": "a", "is_content_word": False},
            {"raw": "test", "is_content_word": True},
        ],
        "annotations": [
            {"lemma": "be", "pos": "VERB", "token_off": [1], "labels": ["omw-en-02604760-v", "omw-en-02616386-v"]},
            {"lemma": "a_test", "pos": "n", "token_off": [2, 3], "labels": ["review%2:31:00::"]},
            {"lemma": "this_is_a", "pos": None, "token_off": [0,1,2], "labels": ["review%2:31:00::"]}
        ]
    }
    expected_output = {
        "text": ["This", "is", "a", "test"],
        "lemmas": ["be", "be", "a_test", "this_is_a"],
        "pos_tags": ["VERB", "VERB", "n", None],
        "token_offsets": [(1,2), (1,2), (2, 4), (0, 3)],
        "labels": ["omw-en-02604760-v", "omw-en-02616386-v", "review%2:31:00::", "review%2:31:00::"]
    }
    assert expected_output == map_token_sense_indexes(test_data)