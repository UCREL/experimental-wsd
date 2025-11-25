from typing import Callable

import pytest
import wn
from transformers import AutoTokenizer
from wn.compat import sensekey

from experimental_wsd.data_processing.shared_token_utils import (
    filter_empty_values,
    join_positive_negative_labels,
    map_and_flatten_token_sense_labels,
    map_negative_sense_ids,
    map_to_definitions,
    map_token_sense_labels,
    sample_to_a_sense,
    token_word_id_mask,
    tokenize_key,
)

EN_LEXICON = "omw-en:1.4"
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand="")
GET_SENSE = sensekey.sense_getter(EN_LEXICON, ENGLISH_WN)
TOKENIZER = AutoTokenizer.from_pretrained(
    "FacebookAI/roberta-base", add_prefix_space=True
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
        "lemmas": ["be", "a_test", "this_is_a"],
        "pos_tags": ["n", "v", None],
        "token_offsets": [(1, 2), (2, 4), (0, 3)],
        "labels": [
            ["become%2:42:01::", "improved%3:00:00::"],
            ["review%2:31:00::"],
            ["review%2:31:00::"],
        ],
        "sense_labels": [
            ["omw-en-become-02626604-v", "omw-en-improved-01288396-a"],
            ["omw-en-review-00696189-v"],
            ["omw-en-review-00696189-v"],
        ],
    }
    if word_net_sense_getter is None:
        del expected_output["sense_labels"]

    assert expected_output == map_token_sense_labels(
        test_data, word_net_sense_getter=word_net_sense_getter
    )


@pytest.mark.parametrize("word_net_sense_getter", [None, GET_SENSE])
def test_map_and_flatten_token_sense_labels(
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

    assert expected_output == map_and_flatten_token_sense_labels(
        test_data, word_net_sense_getter=word_net_sense_getter
    )


@pytest.mark.parametrize("get_random_sense", [True, False])
@pytest.mark.parametrize("pos_tag_mapper", [None])
@pytest.mark.parametrize("normalise_mwe_lemma", [True, False])
@pytest.mark.parametrize("flattened_sense_ids", [True, False])
def test_map_negative_sense_ids(
    get_random_sense: bool,
    pos_tag_mapper: dict[str, str] | None,
    normalise_mwe_lemma: bool,
    flattened_sense_ids: bool,
):
    """
    Args:
        flattened_sense_ids (bool): If True then the sense ids value should be
            a list of strings else it should be a list of a list of strings.
    """
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
    if not flattened_sense_ids:
        test_data = {
            "text": ["This", "is", "a", "test", "New", "York"],
            lemma_key: ["be", "be", "new_york"],
            pos_tag_key: ["n", None, None],
            sense_id_key: [
                ["omw-en-Be-14631295-n"],
                ["omw-en-be-02655135-v", "omw-en-Be-14631295-n"],
                ["omw-en-New_York-09117351-n"],
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
    if not flattened_sense_ids:
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
            ["omw-en-New_York-09119277-n", "omw-en-New_York-09118181-n"],
        ]

    expected_output = {negative_sense_id_key: expected_negative_sense_ids}
    mwe_index = 3
    if not flattened_sense_ids:
        mwe_index = 2
    if not normalise_mwe_lemma:
        expected_negative_sense_ids[mwe_index] = []
        expected_output[negative_sense_id_key] = expected_negative_sense_ids

    if not get_random_sense:
        assert expected_output == output
    else:
        assert 1 == len(output)
        output_negative_sense_ids = output[negative_sense_id_key]

        number_outputs = 4
        start_index, end_index = 1, 3
        if not flattened_sense_ids:
            number_outputs = 3
            start_index, end_index = 1, 2
        assert number_outputs == len(output_negative_sense_ids)
        for index in range(start_index, end_index):
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
            output_random_negative_sense_ids = output_negative_sense_ids[mwe_index]
            assert 1 == len(output_random_negative_sense_ids)
            assert (
                test_data[sense_id_key][mwe_index]
                not in output_random_negative_sense_ids
            )
        else:
            assert (
                expected_negative_sense_ids[mwe_index]
                == output_negative_sense_ids[mwe_index]
            )


def test_sample_to_a_sense():
    test_data = {
        "text": [["This", "is", "a", "test"], ["A", "good", "day"]],
        "lemmas": [["be", "a_test", "this_is_a"], ["good"]],
        "pos_tags": [["n", "v", None], ["a"]],
        "token_offsets": [[(1, 2), (2, 4), (0, 3)], [(1, 2)]],
        "labels": [
            [
                ["become%2:42:01::", "improved%3:00:00::"],
                ["review%2:31:00::"],
                ["review%2:31:00::"],
            ],
            [["good%3:00:01::"]],
        ],
        "sense_labels": [
            [
                ["omw-en-become-02626604-v", "omw-en-improved-01288396-a"],
                ["omw-en-review-00696189-v"],
                ["omw-en-review-00696189-v"],
            ],
            [["omw-en-good-01123148-a"]],
        ],
        "negative_labels": [
            [
                ["example negative", "example negative"],
                ["example negative"],
                ["example negative"],
            ],
            [["example negative", "example negative", "example negative"]],
        ],
    }
    expected_output = {
        "text": [
            ["This", "is", "a", "test"],
            ["This", "is", "a", "test"],
            ["This", "is", "a", "test"],
            ["This", "is", "a", "test"],
            ["A", "good", "day"],
        ],
        "lemmas": ["be", "be", "a_test", "this_is_a", "good"],
        "pos_tags": ["n", "n", "v", None, "a"],
        "token_offsets": [(1, 2), (1, 2), (2, 4), (0, 3), (1, 2)],
        "labels": [
            "become%2:42:01::",
            "improved%3:00:00::",
            "review%2:31:00::",
            "review%2:31:00::",
            "good%3:00:01::",
        ],
        "sense_labels": [
            "omw-en-become-02626604-v",
            "omw-en-improved-01288396-a",
            "omw-en-review-00696189-v",
            "omw-en-review-00696189-v",
            "omw-en-good-01123148-a",
        ],
        "negative_labels": [
            ["example negative", "example negative"],
            ["example negative", "example negative"],
            ["example negative"],
            ["example negative"],
            ["example negative", "example negative", "example negative"],
        ],
    }
    output = sample_to_a_sense(test_data)
    assert len(expected_output) == len(output)

    for key in expected_output:
        assert key in output, key

    assert expected_output == output


@pytest.mark.repeat(30)
@pytest.mark.parametrize("randomize", [True, False])
def test_join_positive_negative_labels(randomize: bool):
    test_data = {
        "text": ["This", "is", "a", "test"],
        "lemmas": "be",
        "pos_tags": "n",
        "token_offsets": (1, 2),
        "labels": ["become%2:42:01::"],
        "sense_labels": "omw-en-become-02626604-v",
        "negative_labels": [
            "omw-en-good-01123148-a",
            "omw-en-review-00696189-v",
            "omw-en-improved-01288396-a",
        ],
    }
    expected_output = {
        "label_sense_ids": [
            "omw-en-become-02626604-v",
            "omw-en-good-01123148-a",
            "omw-en-review-00696189-v",
            "omw-en-improved-01288396-a",
        ],
        "label_ids": 0,
    }

    output = join_positive_negative_labels(test_data, randomize=randomize)
    assert len(expected_output) == len(output)
    if randomize:
        assert len(expected_output["label_sense_ids"]) == len(output["label_sense_ids"])
        for label_sense_id in expected_output["label_sense_ids"]:
            assert label_sense_id in output["label_sense_ids"]
        assert (
            "omw-en-become-02626604-v" == output["label_sense_ids"][output["label_ids"]]
        )
    else:
        assert expected_output == output


def test_filter_empty_values():
    assert not filter_empty_values({"key": []}, "key")
    assert filter_empty_values({"key": ["yes", "no"]}, "key")


def test_token_word_id_mask():
    word_ids_key = "word_ids"
    token_offsets_key = "token_offsets"
    word_id_mask_key = "word_ids_mask"
    test_data = {
        word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None],
        token_offsets_key: [0, 3],
    }

    expected_output = {
        word_id_mask_key: [0, 1, 1, 1, 1, 1, 1, 0, 0],
    }

    assert expected_output == token_word_id_mask(
        test_data, word_ids_key, token_offsets_key, word_id_mask_key
    )

    test_data = {
        word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None],
        token_offsets_key: [0, 1],
    }

    expected_output = {word_id_mask_key: [0, 1, 0, 0, 0, 0, 0, 0, 0]}

    assert expected_output == token_word_id_mask(
        test_data, word_ids_key, token_offsets_key, word_id_mask_key
    )

    test_data = {word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None], token_offsets_key: []}

    expected_output = {word_id_mask_key: []}

    assert expected_output == token_word_id_mask(
        test_data, word_ids_key, token_offsets_key, word_id_mask_key
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

    test_token_data = {
        text_key: [["Hello", "how", "are", "you"], ["I", "am", "ok"], [""]]
    }
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
        test_token_data,
        TOKENIZER,
        text_key,
        output_key_prefix=output_key_prefix,
        is_split_into_words=True,
        add_word_ids=add_word_ids,
    )

    test_token_data = {text_key: ["Hello", "how", "are", "you"]}
    expected_output = {
        "input_ids": [0, 20920, 141, 32, 47, 2],
        "attention_mask": [1, 1, 1, 1, 1, 1],
    }

    if add_word_ids:
        expected_output["word_ids"] = [None, 0, 1, 2, 3, None]

    if output_key_prefix:
        tmp_expected_output = {}
        for key, value in expected_output.items():
            tmp_expected_output[f"{output_key_prefix}_{key}"] = value
        expected_output = tmp_expected_output

    assert expected_output == tokenize_key(
        test_token_data,
        TOKENIZER,
        text_key,
        output_key_prefix=output_key_prefix,
        is_split_into_words=True,
        add_word_ids=add_word_ids,
    )


# def test_explode_key_value():
#    assert 1 == 1
