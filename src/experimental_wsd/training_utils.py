import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from pydantic import BaseModel
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def get_prefix_suffix_special_token_indexes(
    a_tokenizer: PreTrainedTokenizerFast,
) -> tuple[list[int], list[int]]:
    """
    Finds the prefix and suffix special token ids for a given tokenizer when the
    tokenizer tokenizes text (this does not include padding or attention mask), e.g.
    for Roberta this would be `([0], [2])` whereby these ids represent the
    token strings `([<s>], [</s>])`

    These token ids are a list as some tokenizers may have 0 to many prefix and
    suffix special tokens.

    Args:
        a_tokenizer (PreTrainedTokenizerFast): The
            tokenizer to get the special token ids for.
    Returns:
        tuple[list[int], list[int]]: The prefix and suffix special token ids.
    """
    batch_encoding = a_tokenizer("a")
    sentence_encoding = batch_encoding[0]
    special_token_mask = sentence_encoding.special_tokens_mask
    prefix_ids: list[int] = []
    suffix_ids: list[int] = []
    token_ids = sentence_encoding.ids
    is_prefix = True
    for token_index, token_index_value in enumerate(special_token_mask):
        if token_index_value == 0:
            is_prefix = False
            continue
        if is_prefix:
            prefix_ids.append(token_ids[token_index])
        else:
            suffix_ids.append(token_ids[token_index])
    return prefix_ids, suffix_ids


def write_to_jsonl(
    pydantic_generator: Iterable[BaseModel],
    data_dir: Path,
    file_name: str,
    overwrite: bool = False,
) -> Path:
    """
    Given a generator of pydantic model instances it converts the instances into
    JSON string format and save each instance to a new line in it's JSON format
    to the file at `Path(data_dir, file_name)`, therefore creating a JSONL formatted
    file.

    Args:
        pydantic_generator (Iterable[BaseModel]): The data to write to the file
            in JSONL format.
        data_dir (Path): The directory that the file should be created/stored in.
        file_name (str): The file name of the file that will be created in
            `data_dir` and will store the JSONL formatted data.
        overwrite (bool): If True and the file to write too already exists, it
            will be overwritten. Default False.

    Returns:
        Path: The file path to where the data was written too.
    """
    file_path = Path(data_dir, file_name)
    if file_path.exists() and not overwrite:
        logger.info(
            f"The file {file_path} already exists and therefore "
            "not writing the data to it."
        )
        return file_path
    with file_path.open("w", encoding="utf-8") as write_fp:
        for pydantic_data in pydantic_generator:
            write_fp.write(f"{pydantic_data.model_dump_json()}\n")
    return file_path


def read_from_jsonl_file(file_path: Path) -> Iterable[dict[Any, Any]]:
    """
    Reads and returns the JSONL data one line at a time.

    Args:
        file_path (Path): The file path to read the JSONL data from.
    Returns:
        Iterable[dict[Any, Any]]: The JSON data from each new line within the
            given file path.
    """
    with file_path.open("r", encoding="utf-8") as read_fp:
        for line in read_fp:
            line = line.strip()
            json_data = json.loads(line)
            yield json_data


class AscendingSequenceLengthBatchSampler(Sampler[list[int]]):
    """
    A sampler that generates batch indexes in ascending order of length according
    to the `length_key`. In addition these ascending order or length batches can
    be generated in random order with or without replacement. For instances
    onces all of the batch indexes have been created we then perform a random order
    index so that the batches still contain samples that have a similar length
    according to the `length_key`.
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        batch_size: int,
        length_key: str,
        random: bool = True,
        with_replacement: bool = False,
    ) -> None:
        """
        Args:
            data (list[dict[str, Any]]): The data to batch. The outer list represents
                the number of samples in the dataset. The inner dictionary represents
                the data, e.g. `input_ids`, `labels`, etc.
            batch_size (int): The batch size.
            length_key (str): The key that determines how long a sample is, this
                key should be a key within the sample's dictionary, e.g. `input_ids`,
                `labels`, etc.
            random (bool): If True then the batches with similar lengths will be
                yielded in random order rather than the ascending order, this is
                useful when you want to have similar sized batches but in random
                order. When False batches will be in strict ascending order, this
                is useful in want to have something like criculuim learning,
                potentially the easiest and shortest samples first. Default
                True.
            with_replacement (bool): When `random` is `True` then this sets the
                `replacement` argument. Default False.
        """
        self.data = data
        self.batch_size = batch_size
        self.length_key = length_key
        self.random = random
        self.with_replacement = with_replacement

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        sizes = torch.tensor([len(x[self.length_key]) for x in self.data])
        number_batches = len(self)
        batch_index_chunks = list(torch.chunk(torch.argsort(sizes), number_batches))
        indexes_of_batch_index_chunks = torch.arange(
            0, number_batches, dtype=torch.float
        )
        if self.random:
            random_indexes_of_batch_index_chunks = torch.multinomial(
                indexes_of_batch_index_chunks,
                number_batches,
                replacement=self.with_replacement,
            )
            indexes_of_batch_index_chunks = random_indexes_of_batch_index_chunks
        else:
            # Indexes have to be of dtype long and not float, hence the conversion.
            indexes_of_batch_index_chunks = indexes_of_batch_index_chunks.to(
                dtype=torch.long
            )

        for index_of_batch_index_chunks in indexes_of_batch_index_chunks:
            batch_index_chunk = batch_index_chunks[index_of_batch_index_chunks]
            yield batch_index_chunk.tolist()
