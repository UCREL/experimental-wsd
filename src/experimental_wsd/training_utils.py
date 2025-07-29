import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

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


def collate_token_classification_dataset(
    tokenizer: PreTrainedTokenizerFast,
    label_keys: set = set(["labels", "word_ids"]),
    attention_mask_keys: set = set(["attention_mask"]),
    label_pad_id: int = -100,
    attention_pad_id: int = 0,
) -> Callable[[list[dict[str, list[int]]]], dict[str, torch.Tensor]]:
    """
    This generates a function that converts a batch, 1 or more samples, of
    token classification data into format suitable as input into a machine
    learning model for either training or inference.

    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer used to tokenize
            the text/pre-tokenized text. This determines the padding side
            and padding id for any key name that is not in the `label_keys` or
            `attention_mask_keys`.
        label_keys (set[str]): The key names in each sample that should be padded
            with the `label_pad_id`. Default `set([labels, word_ids])`
        attention_mask_keys (set[str]): The key names in each sample that should be padded
            with the `attention_pad_id`. Default `set([attention_mask])`
        label_pad_id (int): The label pad id value. Default is -100.
        attention_pad_id (int): The attention pad id value. Default is 0.
    """

    def _collate(data: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """
        Expects a batch, 1 or more samples, of token classification data and
        returns the data so that the dictionary keys are the same but each value
        is a torch Tensor containing a batch of data whereby the dimension of each
        torch tensor is the same, B x M whereby B is the batch size and M is the
        number of tokens including padding tokens.

        NOTE: That the returned torch.Tensors can be of different M lengths as
        we use the maximum length of each dictionary key as it's maximum padding
        length. This is most likely to occur for the key `labels` if the function
        `data_processing_utils.tokenize_pre_processing` has the argument
        `align_labels_with_tokens=False`, this is perfectly ok just something
        to be aware of. It is very useful for word level (not sub-word token)
        classification models.

        Args:
            data (list[dict[str, list[int]]]): The outer list is the batch size
                and each dictionary should contain the same key names.
        Returns:
            dict[str, torch.Tensor]: A dictionary that has the key names as in the
                given argument dictionaries, but each value is a tensor of size
                B x M whereby B is the batch size and M is the number of tokens
                including any padding tokens.
        """
        padding_side = tokenizer.padding_side
        padding_token = tokenizer.pad_token_id
        key_tensor_lengths: dict[str, list[int]] = defaultdict(list)
        for instance in data:
            for key, value in instance.items():
                key_tensor_lengths[key].append(len(value))

        batched_dict = defaultdict(list)

        # max_length = max(tensor_lengths)
        key_max_length = {
            key: max(tensor_legnths)
            for key, tensor_legnths in key_tensor_lengths.items()
        }
        for instance in data:
            for key, value in instance.items():
                max_length = key_max_length[key]

                tensor_value = torch.tensor(instance[key], dtype=torch.long)
                padding_id_for_key = padding_token
                if key in label_keys:
                    padding_id_for_key = label_pad_id
                if key in attention_mask_keys:
                    padding_id_for_key = attention_pad_id
                pad_length = max_length - tensor_value.size(-1)
                if pad_length == 0:
                    batched_dict[key].append(tensor_value)
                    continue
                pad_tensor = torch.tensor(
                    [padding_id_for_key] * pad_length, dtype=torch.long
                )
                if padding_side == "right":
                    tensor_value = torch.hstack((tensor_value, pad_tensor))
                else:
                    tensor_value = torch.hstack((pad_tensor, tensor_value))

                batched_dict[key].append(tensor_value)

        tensor_batched_dict = {}
        for key, value in batched_dict.items():
            tensor_batched_dict[key] = torch.vstack(value)
        return tensor_batched_dict

    return _collate
