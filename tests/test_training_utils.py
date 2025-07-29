from pathlib import Path
from typing import Iterable

import pytest
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from experimental_wsd.training_utils import (
    AscendingSequenceLengthBatchSampler,
    collate_token_classification_dataset,
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
