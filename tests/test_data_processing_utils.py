import pytest
from transformers import AutoTokenizer

from experimental_wsd.data_processing_utils import (
    align_labels_with_tokens,
    map_token_text_and_is_content_labels,
    tokenize_and_align_labels,
)


@pytest.mark.parametrize("label_pad_id", [-100, 50])
def test_align_labels_with_tokens(label_pad_id: int):
    expected_sub_word_labels = [label_pad_id, 0, 1, 1, 1, 1, 1, 1, 1, 0, label_pad_id]
    labels = [0, 1, 0]
    word_ids = [None, 0, 1, 1, 1, 1, 1, 1, 1, 2, None]
    sub_word_labels = align_labels_with_tokens(labels, word_ids, label_pad_id)
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


@pytest.mark.parametrize("label_pad_id", [-100, 50])
def test_tokenize_and_align_labels(label_pad_id: int):
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
    tokenizer = AutoTokenizer.from_pretrained(
        "FacebookAI/roberta-base", add_prefix_space=True
    )
    assert expected_output == tokenize_and_align_labels(
        test_data, tokenizer, label_pad_id
    )
