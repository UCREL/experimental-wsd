import pytest
import wn
from transformers import AutoTokenizer
from wn.compat import sensekey

from experimental_wsd.data_processing_utils import (
    get_align_labels_with_tokens,
    map_empty_removal_token_values,
    map_token_text_and_is_content_labels,
    token_word_id_mask,
    tokenize_pre_processing,
)

EN_LEXICON = "omw-en:1.4"
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand="")
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)
TOKENIZER = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-base", add_prefix_space=True
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

    assert expected_output == tokenize_pre_processing(
        test_data, TOKENIZER, label_pad_id, align_labels_with_tokens
    )


def test_map_empty_removal_token_values():
    # Case that nothing should be removed.
    test_data = {
        "lemmas": ["hi", "how", "are", "you"],
        "pos": [None, "v", "n", "a"],
        "senses": [["25", "36"], ["45"], ["2"], ["1"]],
    }
    assert test_data == map_empty_removal_token_values(
        test_data, "senses", ["lemmas", "pos"]
    )
    # Case whereby nothing is removed but as we did not pass any aligned keys only
    # the filtered key and it's value should be returned
    assert {
        "senses": [["25", "36"], ["45"], ["2"], ["1"]]
    } == map_empty_removal_token_values(test_data, "senses", [])

    # Case whereby one of the sense values is empty therefore it and it's aligned
    # values should be filtered out
    test_data["senses"] = [["25", "36"], [], ["2"], ["1"]]
    expected_output = {
        "lemmas": ["hi", "are", "you"],
        "pos": [None, "n", "a"],
        "senses": [["25", "36"], ["2"], ["1"]],
    }
    assert expected_output == map_empty_removal_token_values(
        test_data, "senses", ["lemmas", "pos"]
    )

    # Test the case whereby an aligned value is not the same length as the
    # filter key value
    test_data["pos"] = [None, None, None]
    with pytest.raises(ValueError):
        map_empty_removal_token_values(test_data, "senses", ["lemmas", "pos"])


def test_token_word_id_mask():
    word_ids_key = "word_ids"
    token_offsets_key = "token_offsets"
    word_id_mask_key = "word_ids_mask"
    test_data = {
        word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None],
        token_offsets_key: [[0, 1], [0, 3], [1, 4]],
    }

    expected_output = {
        word_id_mask_key: [
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0],
        ]
    }

    assert expected_output == token_word_id_mask(
        test_data, word_ids_key, token_offsets_key, word_id_mask_key
    )

    test_data = {word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None], token_offsets_key: []}

    expected_output = {word_id_mask_key: []}

    assert expected_output == token_word_id_mask(
        test_data, word_ids_key, token_offsets_key, word_id_mask_key
    )
