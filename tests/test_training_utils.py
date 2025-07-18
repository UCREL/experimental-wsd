from pathlib import Path
from typing import Iterable

import pytest
from pydantic import BaseModel
from transformers import AutoTokenizer

from experimental_wsd.training_utils import (
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
