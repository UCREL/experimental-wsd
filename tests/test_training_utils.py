from pathlib import Path
from typing import Iterable

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from experimental_wsd.training_utils import (
    AscendingSequenceLengthBatchSampler,
    AscendingTokenNegativeExamplesBatchSampler,
    DescendingTokenSimilarityBatchSampler,
    collate_token_classification_dataset,
    collate_token_negative_examples_classification_dataset,
    collate_variable_token_similarity_dataset,
    get_prefix_suffix_special_token_indexes,
    read_from_jsonl_file,
    write_to_jsonl,
)


def test_get_prefix_suffix_special_token_indexes():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    roberta_prefix_suffix_special_tokens_ids = ([0], [2])
    assert (
        roberta_prefix_suffix_special_tokens_ids
        == get_prefix_suffix_special_token_indexes(tokenizer)
    )


@pytest.mark.parametrize("overwrite", [True, False])
def test_read_write_to_jsonl(tmp_path: Path, overwrite: bool):
    class TestAnimalModel(BaseModel):
        species: str

    def test_model_generator(data: list[dict[str, str]]) -> Iterable[BaseModel]:
        for instance in data:
            yield TestAnimalModel.model_validate(instance)

    data_to_write = [{"species": "dog"}, {"species": "cat"}]

    file_name = "animal_data.jsonl"
    written_file = write_to_jsonl(
        test_model_generator(data_to_write), tmp_path, file_name, overwrite=False
    )
    assert written_file == Path(tmp_path, "animal_data.jsonl")
    assert data_to_write == list(read_from_jsonl_file(written_file))

    written_file = write_to_jsonl(
        test_model_generator(data_to_write), tmp_path, file_name, overwrite=overwrite
    )
    assert written_file == Path(tmp_path, "animal_data.jsonl")
    assert data_to_write == list(read_from_jsonl_file(written_file))


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("random", [True, False])
@pytest.mark.parametrize("with_replacement", [True, False])
def test_ascending_sequence_length_batch_sampler(random: bool, with_replacement: bool):
    sampler_batch_size = 2
    data_length_key = "input_ids"
    test_data = [
        {"input_ids": list(range(0, 10))},
        {"input_ids": list(range(0, 8))},
        {"input_ids": list(range(0, 12))},
        {"input_ids": list(range(0, 7))},
        {"input_ids": list(range(0, 2))},
    ]
    sampler = AscendingSequenceLengthBatchSampler(
        test_data,
        sampler_batch_size,
        data_length_key,
        random=random,
        with_replacement=with_replacement,
    )

    expected_batch_lengths = set([1, 2])
    expected_total_size = 5
    total_size = 0
    batch_indexes = []
    for batch_index, batch in enumerate(sampler):
        assert isinstance(batch, list)
        assert isinstance(batch[0], int)
        batch_size = len(batch)
        if not random:
            if batch_index == 2:
                assert 1 == batch_size
            else:
                assert 2 == batch_size
        else:
            assert batch_size in expected_batch_lengths
        total_size += batch_size
        batch_indexes += batch
    # Can be the case that with_replacement selects the smallest batch for all
    # batch indexes
    if random and with_replacement:
        minimum_total_size = 3
        assert minimum_total_size <= total_size
        assert expected_total_size >= total_size
    else:
        assert expected_total_size == total_size, batch_indexes

    # Expect the indexes that are returned to be unique
    if random and with_replacement:
        assert len(batch_indexes) != len(set(batch_indexes))
    else:
        assert len(batch_indexes) == len(set(batch_indexes))

    expected_batch_indexes = [4, 3, 1, 0, 2]
    if not random:
        assert expected_batch_indexes == batch_indexes
    else:
        assert expected_batch_indexes != batch_indexes


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("with_replacement", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_descending_token_similarity_batch_sampler(
    random: bool, with_replacement: bool, batch_size: int
):
    similarity_sentence_key = "sentence_labels"
    test_data = [
        {
            similarity_sentence_key: [[0, 1], [0], [0, 1, 2], [0]],
        },
        {
            similarity_sentence_key: [[0, 1]],
        },
        {
            similarity_sentence_key: [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2]],
        },
    ]
    sampler = DescendingTokenSimilarityBatchSampler(
        test_data,
        batch_size=batch_size,
        similarity_sentence_key=similarity_sentence_key,
        random=random,
        with_replacement=with_replacement,
    )
    expected_batch_lengths = set([1, 2])
    if batch_size == 1:
        expected_batch_lengths = set([1])
    expected_total_size = 3
    total_size = 0
    batch_indexes = []
    for batch_index, batch in enumerate(sampler):
        assert isinstance(batch, list)
        assert isinstance(batch[0], int)
        sample_batch_size = len(batch)
        if not random:
            if batch_size == 2 and batch_index == 1:
                assert 1 == sample_batch_size
            elif batch_size == 2:
                assert 2 == sample_batch_size
            else:
                assert 1 == sample_batch_size
        else:
            assert batch_size in expected_batch_lengths
        total_size += sample_batch_size
        batch_indexes += batch
    # Can be the case that with_replacement selects the smallest batch for all
    # batch indexes
    if random and with_replacement:
        minimum_total_size = 2
        assert minimum_total_size <= total_size
        assert expected_total_size >= total_size
    else:
        assert expected_total_size == total_size, batch_indexes

    # Expect the indexes that are returned to be unique
    if random and with_replacement:
        assert len(batch_indexes) != len(set(batch_indexes))
    else:
        assert len(batch_indexes) == len(set(batch_indexes))

    expected_batch_indexes = [0, 2, 1]
    if not random:
        assert expected_batch_indexes == batch_indexes
    else:
        assert expected_batch_indexes != batch_indexes


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("with_replacement", [False, True])
def test_ascending_token_negative_examples_batch_sampler(
    random: bool, with_replacement: bool
):
    positive_sample_key = "positive_labels"
    negative_sample_key = "negative_labels"
    test_data = [
        {
            positive_sample_key: [0, 1, 2, 3],
            negative_sample_key: [[0, 1], [0], [0, 1, 2], [0]],
        },
        {
            positive_sample_key: [0],
            negative_sample_key: [[0, 1]],
        },
        {
            positive_sample_key: [0, 1],
            negative_sample_key: [[0], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        },
    ]
    sampler = AscendingTokenNegativeExamplesBatchSampler(
        test_data,
        batch_size=2,
        positive_sample_key=positive_sample_key,
        negative_sample_key=negative_sample_key,
        random=random,
        with_replacement=with_replacement,
    )
    expected_batch_lengths = set([1, 2])
    expected_total_size = 3
    total_size = 0
    batch_indexes = []
    for batch_index, batch in enumerate(sampler):
        assert isinstance(batch, list)
        assert isinstance(batch[0], int)
        batch_size = len(batch)
        if not random:
            if batch_index == 1:
                assert 1 == batch_size
            else:
                assert 2 == batch_size
        else:
            assert batch_size in expected_batch_lengths
        total_size += batch_size
        batch_indexes += batch
    # Can be the case that with_replacement selects the smallest batch for all
    # batch indexes
    if random and with_replacement:
        minimum_total_size = 2
        assert minimum_total_size <= total_size
        assert expected_total_size >= total_size
    else:
        assert expected_total_size == total_size, batch_indexes

    # Expect the indexes that are returned to be unique
    if random and with_replacement:
        assert len(batch_indexes) != len(set(batch_indexes))
    else:
        assert len(batch_indexes) == len(set(batch_indexes))

    expected_batch_indexes = [1, 0, 2]
    if not random:
        assert expected_batch_indexes == batch_indexes
    else:
        assert expected_batch_indexes != batch_indexes


@pytest.mark.parametrize(
    "tokenizer_name", ["FacebookAI/roberta-base", "jhu-clsp/ettin-encoder-17m"]
)
@pytest.mark.parametrize("expected_label_pad_id", [-100, 50])
@pytest.mark.parametrize("expected_attention_pad_id", [0, -51])
@pytest.mark.parametrize("attention_mask_key", ["attention_mask", "sequence_mask"])
@pytest.mark.parametrize("label_key_name", ["labels", "label_sequence"])
@pytest.mark.parametrize("word_ids_key_name", ["word_ids", "token_ids"])
def test_collate_token_classification_dataset(
    tokenizer_name: str,
    expected_label_pad_id: int,
    expected_attention_pad_id: int,
    attention_mask_key: str,
    label_key_name: str,
    word_ids_key_name: str,
):
    test_data = [
        {
            "input_ids": list(range(0, 10)),
            attention_mask_key: [1] * 10,
            label_key_name: [1] * 10,
            word_ids_key_name: [1] * 10,
        },
        {
            "input_ids": list(range(0, 8)),
            attention_mask_key: [1] * 8,
            label_key_name: [1] * 8,
            word_ids_key_name: [1] * 8,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    attention_mask_keys = set([attention_mask_key])
    label_keys = set([label_key_name, word_ids_key_name])
    collate_function = collate_token_classification_dataset(
        tokenizer,
        label_pad_id=expected_label_pad_id,
        attention_pad_id=expected_attention_pad_id,
        attention_mask_keys=attention_mask_keys,
        label_keys=label_keys,
    )

    collated_test_data = collate_function(test_data)
    expected_collated_data = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    tokenizer.pad_token_id,
                    tokenizer.pad_token_id,
                ],
            ],
            dtype=torch.long,
        ),
        attention_mask_key: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                ],
            ],
            dtype=torch.long,
        ),
        label_key_name: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, expected_label_pad_id, expected_label_pad_id],
            ],
            dtype=torch.long,
        ),
        word_ids_key_name: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, expected_label_pad_id, expected_label_pad_id],
            ],
            dtype=torch.long,
        ),
    }

    assert len(expected_collated_data) == len(collated_test_data)
    for data_key_name, expected_batched_data in expected_collated_data.items():
        assert (
            expected_batched_data.tolist() == collated_test_data[data_key_name].tolist()
        ), data_key_name

    # Test when the samples are the same size, no padding is required, only
    # converting to torch Tensors.
    test_data = [
        {
            "input_ids": list(range(0, 10)),
            attention_mask_key: [1] * 10,
            label_key_name: [1] * 10,
            word_ids_key_name: [1] * 10,
        },
        {
            "input_ids": list(range(0, 10)),
            attention_mask_key: [1] * 10,
            label_key_name: [1] * 10,
            word_ids_key_name: [1] * 10,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    attention_mask_keys = set([attention_mask_key])
    label_keys = set([label_key_name, word_ids_key_name])
    collate_function = collate_token_classification_dataset(
        tokenizer,
        label_pad_id=expected_label_pad_id,
        attention_pad_id=expected_attention_pad_id,
        attention_mask_keys=attention_mask_keys,
        label_keys=label_keys,
    )

    collated_test_data = collate_function(test_data)
    expected_collated_data = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        attention_mask_key: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        label_key_name: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        word_ids_key_name: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    }

    assert len(expected_collated_data) == len(collated_test_data)
    for data_key_name, expected_batched_data in expected_collated_data.items():
        assert (
            expected_batched_data.tolist() == collated_test_data[data_key_name].tolist()
        ), data_key_name

    # Test the case of labels being of different length to other inputs
    # the expected outcome is that we only pad to the maximum length of the labels

    test_data = [
        {
            "input_ids": list(range(0, 10)),
            attention_mask_key: [1] * 10,
            label_key_name: [1] * 4,
            word_ids_key_name: [1] * 10,
        },
        {
            "input_ids": list(range(0, 10)),
            attention_mask_key: [1] * 10,
            label_key_name: [1] * 6,
            word_ids_key_name: [1] * 10,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    attention_mask_keys = set([attention_mask_key])
    label_keys = set([label_key_name, word_ids_key_name])
    collate_function = collate_token_classification_dataset(
        tokenizer,
        label_pad_id=expected_label_pad_id,
        attention_pad_id=expected_attention_pad_id,
        attention_mask_keys=attention_mask_keys,
        label_keys=label_keys,
    )

    collated_test_data = collate_function(test_data)
    expected_collated_data = {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        attention_mask_key: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        label_key_name: torch.tensor(
            [
                [1, 1, 1, 1, expected_label_pad_id, expected_label_pad_id],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        word_ids_key_name: torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
    }

    assert len(expected_collated_data) == len(collated_test_data)
    for data_key_name, expected_batched_data in expected_collated_data.items():
        assert (
            expected_batched_data.tolist() == collated_test_data[data_key_name].tolist()
        ), data_key_name


@pytest.mark.parametrize(
    "tokenizer_name", ["FacebookAI/roberta-base", "jhu-clsp/ettin-encoder-17m"]
)
@pytest.mark.parametrize("expected_label_pad_id", [-100, 50])
@pytest.mark.parametrize("expected_attention_pad_id", [0, -51])
@pytest.mark.parametrize("attention_mask_key", ["attention_mask", "sequence_mask"])
@pytest.mark.parametrize("label_key_name", ["labels", "label_sequence"])
@pytest.mark.parametrize("word_ids_key", ["word_ids", "token_ids"])
def test_collate_token_negative_examples_classification_dataset(
    tokenizer_name: str,
    expected_label_pad_id: int,
    expected_attention_pad_id: int,
    attention_mask_key: str,
    label_key_name: str,
    word_ids_key: str,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    positive_input_ids = "positive_input_ids"
    negative_input_ids = "negative_input_ids"
    text_input_ids = "text_input_ids"
    positive_attention_mask = f"positive_{attention_mask_key}"
    negative_attention_mask = f"negative_{attention_mask_key}"
    text_attention_mask = f"text_{attention_mask_key}"
    text_word_ids = f"text_{word_ids_key}"

    # The first test data sample contain a text sequence with:
    # 2 samples, first sample contain 2 negative examples and the second 1 negative example
    # Second test data sample contains a text sequence with:
    # 1 sample, first sample contains 4 negative examples.
    test_data = [
        {
            positive_input_ids: [list(range(5)), list(range(2))],
            positive_attention_mask: [[1] * 5, [1] * 2],
            negative_input_ids: [[list(range(6)), list(range(3))], [list(range(4))]],
            negative_attention_mask: [[[1] * 6, [1] * 3], [[1] * 4]],
            text_input_ids: list(range(7)),
            text_attention_mask: [1] * 7,
            text_word_ids: [[0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]],
        },
        {
            positive_input_ids: [list(range(6))],
            positive_attention_mask: [[1] * 6],
            negative_input_ids: [
                [list(range(2)), list(range(8)), list(range(1)), list(range(3))]
            ],
            negative_attention_mask: [[[1] * 2, [1] * 8, [1], [1] * 3]],
            text_input_ids: list(range(10)),
            text_attention_mask: [1] * 10,
            text_word_ids: [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    label_key = label_key_name
    attention_mask_keys = set(
        [
            positive_attention_mask,
            negative_attention_mask,
            text_attention_mask,
            text_word_ids,
        ]
    )
    collate_function = collate_token_negative_examples_classification_dataset(
        tokenizer,
        label_key=label_key,
        text_token_word_ids_mask_key=text_word_ids,
        text_keys=[text_input_ids, text_attention_mask],
        positive_samples_keys=[positive_input_ids, positive_attention_mask],
        negative_samples_keys=[negative_input_ids, negative_attention_mask],
        label_pad_id=expected_label_pad_id,
        attention_pad_id=expected_attention_pad_id,
        attention_mask_keys=attention_mask_keys,
    )

    collated_test_data = collate_function(test_data)
    expected_collated_data = {
        positive_input_ids: torch.tensor(
            [
                [
                    [0, 1, 2, 3, 4, tokenizer.pad_token_id],
                    [
                        0,
                        1,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                    ],
                ],
                [[0, 1, 2, 3, 4, 5], [tokenizer.pad_token_id] * 6],
            ],
            dtype=torch.long,
        ),
        positive_attention_mask: torch.tensor(
            [
                [
                    [1, 1, 1, 1, 1, expected_attention_pad_id],
                    [
                        1,
                        1,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                    ],
                ],
                [[1, 1, 1, 1, 1, 1], [expected_attention_pad_id] * 6],
            ],
            dtype=torch.long,
        ),
        negative_input_ids: torch.tensor(
            [
                [
                    [
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                        [
                            0,
                            1,
                            2,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                    ],
                    [
                        [
                            0,
                            1,
                            2,
                            3,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                    ],
                ],
                [
                    [
                        [
                            0,
                            1,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [
                            0,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                        [
                            0,
                            1,
                            2,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                            tokenizer.pad_token_id,
                        ],
                    ],
                    [
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                        [tokenizer.pad_token_id] * 8,
                    ],
                ],
            ],
            dtype=torch.long,
        ),
        negative_attention_mask: torch.tensor(
            [
                [
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                        [
                            1,
                            1,
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                    ],
                    [
                        [
                            1,
                            1,
                            1,
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                    ],
                ],
                [
                    [
                        [
                            1,
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                        [
                            1,
                            1,
                            1,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                            expected_attention_pad_id,
                        ],
                    ],
                    [
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                        [expected_attention_pad_id] * 8,
                    ],
                ],
            ],
            dtype=torch.long,
        ),
        text_input_ids: torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    tokenizer.pad_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.pad_token_id,
                ],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        text_attention_mask: torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                ],
                [1] * 10,
            ],
            dtype=torch.long,
        ),
        text_word_ids: torch.tensor(
            [
                [
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                    ],
                    [
                        0,
                        0,
                        0,
                        0,
                        1,
                        1,
                        1,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                    ],
                ],
                [
                    [
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    [expected_attention_pad_id] * 10,
                ],
            ],
            dtype=torch.long,
        ),
        label_key_name: torch.tensor(
            [
                [0, 0],
                [0, expected_label_pad_id],
            ],
            dtype=torch.long,
        ),
    }

    assert len(expected_collated_data) == len(collated_test_data)
    for data_key_name, expected_batched_data in expected_collated_data.items():
        assert (
            expected_batched_data.tolist() == collated_test_data[data_key_name].tolist()
        ), data_key_name


@pytest.mark.parametrize(
    "tokenizer_name", ["FacebookAI/roberta-base", "jhu-clsp/ettin-encoder-17m"]
)
@pytest.mark.parametrize("expected_attention_pad_id", [0, -51])
@pytest.mark.parametrize("attention_mask_key", ["attention_mask", "sequence_mask"])
@pytest.mark.parametrize("label_key_name", ["labels", "label_sequence"])
@pytest.mark.parametrize("word_ids_key", ["word_ids", "token_ids"])
def test_collate_variable_token_similarity_dataset(
    tokenizer_name: str,
    expected_attention_pad_id: int,
    attention_mask_key: str,
    label_key_name: str,
    word_ids_key: str,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    label_definitions_input_ids = "label_definitions_input_ids"
    text_input_ids = "text_input_ids"
    label_definitions_attention_mask = f"label_definitions_{attention_mask_key}"
    text_attention_mask = f"text_{attention_mask_key}"
    text_word_ids = f"text_{word_ids_key}"

    # first test contains 2 similar sentences, and the second contains 1 similar sentence.
    test_data = [
        {
            label_definitions_input_ids: [list(range(5)), list(range(2))],
            label_definitions_attention_mask: [[1] * 5, [1] * 2],
            text_input_ids: list(range(7)),
            text_attention_mask: [1] * 7,
            text_word_ids: [0, 0, 0, 0, 1, 1, 1],
            label_key_name: 1,
        },
        {
            label_definitions_input_ids: [list(range(6))],
            label_definitions_attention_mask: [[1] * 6],
            text_input_ids: list(range(10)),
            text_attention_mask: [1] * 10,
            text_word_ids: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            label_key_name: 0,
        },
    ]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    assert "right" == tokenizer.padding_side

    label_key = label_key_name
    collate_function = collate_variable_token_similarity_dataset(
        tokenizer,
        label_key=label_key,
        text_input_ids=text_input_ids,
        text_attention_mask=text_attention_mask,
        text_word_ids_mask=text_word_ids,
        similarity_sentence_input_ids=label_definitions_input_ids,
        similarity_sentence_attention_mask=label_definitions_attention_mask,
        attention_pad_id=expected_attention_pad_id,
    )

    collated_test_data = collate_function(test_data)
    expected_collated_data = {
        label_definitions_input_ids: torch.tensor(
            [
                [
                    [0, 1, 2, 3, 4, tokenizer.pad_token_id],
                    [
                        0,
                        1,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                        tokenizer.pad_token_id,
                    ],
                ],
                [[0, 1, 2, 3, 4, 5], [tokenizer.pad_token_id] * 6],
            ],
            dtype=torch.long,
        ),
        label_definitions_attention_mask: torch.tensor(
            [
                [
                    [1, 1, 1, 1, 1, expected_attention_pad_id],
                    [
                        1,
                        1,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                        expected_attention_pad_id,
                    ],
                ],
                [[1, 1, 1, 1, 1, 1], [expected_attention_pad_id] * 6],
            ],
            dtype=torch.long,
        ),
        text_input_ids: torch.tensor(
            [
                [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    tokenizer.pad_token_id,
                    tokenizer.pad_token_id,
                    tokenizer.pad_token_id,
                ],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        ),
        text_attention_mask: torch.tensor(
            [
                [
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                ],
                [1] * 10,
            ],
            dtype=torch.long,
        ),
        text_word_ids: torch.tensor(
            [
                [
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                    expected_attention_pad_id,
                ],
                [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ],
            dtype=torch.long,
        ),
        label_key_name: torch.tensor(
            [
                1,
                0,
            ],
            dtype=torch.long,
        ),
    }

    assert len(expected_collated_data) == len(collated_test_data)
    for data_key_name, expected_batched_data in expected_collated_data.items():
        assert (
            expected_batched_data.tolist() == collated_test_data[data_key_name].tolist()
        ), data_key_name
