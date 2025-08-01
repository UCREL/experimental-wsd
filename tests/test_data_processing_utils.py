from typing import Callable

import pytest
import wn
from transformers import AutoTokenizer
from wn.compat import sensekey

from experimental_wsd.data_processing_utils import (
    filter_empty_values,
    get_align_labels_with_tokens,
    map_empty_removal_token_values,
    map_negative_sense_ids,
    map_to_definitions,
    map_token_sense_labels,
    map_token_text_and_is_content_labels,
    tokenize_key,
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


@pytest.mark.parametrize("get_random_sense", [True, False])
@pytest.mark.parametrize("pos_tag_mapper", [None])
@pytest.mark.parametrize("normalise_mwe_lemma", [True, False])
def test_map_negative_sense_ids(
    get_random_sense: bool,
    pos_tag_mapper: dict[str, str] | None,
    normalise_mwe_lemma: bool,
):
    sense_id_key = "sense_labels"
    lemma_key = "lemmas"
    pos_tag_key = "pos_tags"
    negative_sense_id_key = "negative_labels"

    test_data = {
        "text": ["This", "is", "a", "test", "New", "York"],
        lemma_key: ["be", "be", "be", "new_york"],
        pos_tag_key: ["n", "v", None, None],
        sense_id_key: [
            "omw-en-Be-14631295-n",
            "omw-en-be-02655135-v",
            "omw-en-be-02614181-v",
            "omw-en-New_York-09117351-n",
        ],
    }

    output = map_negative_sense_ids(
        test_data,
        ENGLISH_WN,
        sense_id_key,
        lemma_key,
        pos_tag_key,
        negative_sense_id_key,
        get_random_sense,
        pos_tag_mapper,
        normalise_mwe_lemma,
    )

    expected_negative_sense_ids = [
        [],
        [
            "omw-en-be-02604760-v",
            "omw-en-be-02616386-v",
            "omw-en-be-02603699-v",
            "omw-en-be-02749904-v",
            "omw-en-be-02664769-v",
            "omw-en-be-02620587-v",
            "omw-en-be-02445925-v",
            "omw-en-be-02697725-v",
            "omw-en-be-02268246-v",
            "omw-en-be-02614181-v",
            "omw-en-be-02744820-v",
            "omw-en-be-02702508-v",
        ],
        [
            "omw-en-Be-14631295-n",
            "omw-en-be-02604760-v",
            "omw-en-be-02616386-v",
            "omw-en-be-02655135-v",
            "omw-en-be-02603699-v",
            "omw-en-be-02749904-v",
            "omw-en-be-02664769-v",
            "omw-en-be-02620587-v",
            "omw-en-be-02445925-v",
            "omw-en-be-02697725-v",
            "omw-en-be-02268246-v",
            "omw-en-be-02744820-v",
            "omw-en-be-02702508-v",
        ],
        ["omw-en-New_York-09119277-n", "omw-en-New_York-09118181-n"],
    ]

    expected_output = {negative_sense_id_key: expected_negative_sense_ids}
    if not normalise_mwe_lemma:
        expected_negative_sense_ids[3] = []
        expected_output[negative_sense_id_key] = expected_negative_sense_ids

    if not get_random_sense:
        assert expected_output == output
    else:
        assert 1 == len(output)
        output_negative_sense_ids = output[negative_sense_id_key]
        assert 4 == len(output_negative_sense_ids)
        for index in range(1, 3):
            assert (
                expected_negative_sense_ids[index] == output_negative_sense_ids[index]
            )

        # As the first entry is non-ambagious and we have requested a random sense
        # need to check this has happened
        output_random_negative_sense_ids = output_negative_sense_ids[0]
        assert 1 == len(output_random_negative_sense_ids)
        assert test_data[sense_id_key][0] not in output_random_negative_sense_ids
        # If we have not normalised the MWE lemma then this is non-ambagious as
        # it does not exist in the sense inventory therefore need to check that
        # it creates a random sense as a negative sense.
        if not normalise_mwe_lemma:
            output_random_negative_sense_ids = output_negative_sense_ids[3]
            assert 1 == len(output_random_negative_sense_ids)
            assert test_data[sense_id_key][3] not in output_random_negative_sense_ids
        else:
            assert expected_negative_sense_ids[3] == output_negative_sense_ids[3]


def test_filter_empty_values():
    assert not filter_empty_values({"key": []}, "key")
    assert filter_empty_values({"key": ["yes", "no"]}, "key")


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


@pytest.mark.parametrize("add_word_ids", [True, False])
@pytest.mark.parametrize("output_key_prefix", ["", "test"])
def test_tokenize_key(output_key_prefix: str, add_word_ids: bool):
    text_key = "gloss"
    test_data = {text_key: ["Hello how are you", "I am ok", ""]}

    expected_output = {
        "input_ids": [[0, 20920, 141, 32, 47, 2], [0, 38, 524, 15983, 2], [0, 2]],
        "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1]],
    }
    if add_word_ids:
        expected_output["word_ids"] = [
            [None, 0, 1, 2, 3, None],
            [None, 0, 1, 2, None],
            [None, None],
        ]
    if output_key_prefix:
        tmp_expected_output = {}
        for key, value in expected_output.items():
            tmp_expected_output[f"{output_key_prefix}_{key}"] = value
        expected_output = tmp_expected_output
    assert expected_output == tokenize_key(
        test_data, TOKENIZER, text_key, output_key_prefix, add_word_ids
    )

    test_multiple_list_data = {text_key: [["Hello how are you"], ["I am ok", ""]]}
    expected_output = {
        "input_ids": [[[0, 20920, 141, 32, 47, 2]], [[0, 38, 524, 15983, 2], [0, 2]]],
        "attention_mask": [[[1, 1, 1, 1, 1, 1]], [[1, 1, 1, 1, 1], [1, 1]]],
    }
    if add_word_ids:
        expected_output["word_ids"] = [
            [[None, 0, 1, 2, 3, None]],
            [[None, 0, 1, 2, None], [None, None]],
        ]
    if output_key_prefix:
        tmp_expected_output = {}
        for key, value in expected_output.items():
            tmp_expected_output[f"{output_key_prefix}_{key}"] = value
        expected_output = tmp_expected_output
    assert expected_output == tokenize_key(
        test_multiple_list_data, TOKENIZER, text_key, output_key_prefix, add_word_ids
    )


def test_map_to_definitions():
    sense_key = "labels"
    definition_key = "label_definitions"
    test_data = {
        sense_key: [
            "omw-en-be-02702508-v",
            "omw-en-New_York-09119277-n",
            "omw-en-New_York-09118181-n",
        ]
    }
    expected_output = {
        definition_key: [
            "be priced at",
            "the largest city in New York State and in the United States; located in southeastern New York at the mouth of the Hudson river; a major financial and cultural center",
            "one of the British colonies that formed the United States",
        ]
    }
    assert expected_output == map_to_definitions(
        test_data, sense_key, ENGLISH_WN, definition_key
    )
    test_data = {
        sense_key: [
            ["omw-en-be-02702508-v"],
            ["omw-en-New_York-09119277-n", "omw-en-New_York-09118181-n"],
        ]
    }
    expected_output = {
        definition_key: [
            ["be priced at"],
            [
                "the largest city in New York State and in the United States; located in southeastern New York at the mouth of the Hudson river; a major financial and cultural center",
                "one of the British colonies that formed the United States",
            ],
        ]
    }
    assert expected_output == map_to_definitions(
        test_data, sense_key, ENGLISH_WN, definition_key
    )
