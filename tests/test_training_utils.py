from pathlib import Path
from typing import Iterable

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from experimental_wsd.training_utils import (
    AscendingSequenceLengthBatchSampler,
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
