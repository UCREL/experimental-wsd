import copy
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
    to the `length_key`. In addition these ascending order of length batches can
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


class AscendingTokenNegativeExamplesBatchSampler(Sampler[list[int]]):
    """
    A sampler that generates batch indexes in ascending order of number of
    positive samples (M) within a sequence multiplied by the
    maximum number of negative samples (N) for a positive sample within that
    sequence.

    The length of the data is not the number of positive samples but rather
    the number of sequences whereby the number of samples in a sequence in
    variable.
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        batch_size: int,
        positive_sample_key: str,
        negative_sample_key: str,
        random: bool = True,
        with_replacement: bool = False,
    ) -> None:
        """
        Args:
            data (list[dict[str, Any]]): The data to batch. The outer list represents
                the number of samples in the dataset. The inner dictionary represents
                the data, e.g. `positive_label_input_ids`, `negative_label_input_ids`, etc.
            batch_size (int): The batch size.
            positive_sample_key (str): The key that represents the positive
                samples within a sequence. The value of the key per sequence
                should be a list of strings/list of integer IDS representing
                tokens.
            negative_sample_key (str): The key that represents the negative
                samples within a sequence. The value of the key per sequence
                should be a list of a list of strings/list of integer IDS representing
                tokens. The outer list per sequence should be the same size as the
                positive sample key value list and the inner list is variable
                in length as each positive sample can have a different number
                of negative samples.
            random (bool): If True then the batches with similar sizes will be
                yielded in random order rather than the ascending order.
                This is useful when you want to have similar sized batches but in random
                order. Default True.
            with_replacement (bool): When `random` is `True` then this sets the
                `replacement` argument. Default False.
        """
        self.data = data
        self.batch_size = batch_size
        self.positive_sample_key = positive_sample_key
        self.negative_sample_key = negative_sample_key
        self.random = random
        self.with_replacement = with_replacement

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        sizes = []
        for sequence in self.data:
            number_positive_samples = len(sequence[self.positive_sample_key])
            largest_number_negative_samples = 0
            for negative_sample in sequence[self.negative_sample_key]:
                number_negative_samples = len(negative_sample)
                if number_negative_samples > largest_number_negative_samples:
                    largest_number_negative_samples = number_negative_samples
            sequence_size = number_positive_samples * largest_number_negative_samples
            sizes.append(sequence_size)
        sizes = torch.tensor(sizes)
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

class DescendingTokenSimilarityBatchSampler(Sampler[list[int]]):
    """
    A sampler that generates batch indexes in descending order of number of
    similarity sentences that a token has to match with, of which 1 sentence 
    is expected to be the positive matched sentence.
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        batch_size: int,
        similarity_sentence_key: str,
        random: bool = True,
        with_replacement: bool = False,
    ) -> None:
        """
        Args:
            data (list[dict[str, Any]]): The data to batch. The outer list represents
                the number of samples in the dataset. The inner dictionary represents
                the data, e.g. `positive_label_input_ids`, `negative_label_input_ids`, etc.
            batch_size (int): The batch size.
            similarity_sentence_key (str): The key that represents the 
                sentences that the token has to be matched within. This key 
                should contain a list of a list of token IDs.
            random (bool): If True then the batches with similar sizes will be
                yielded in random order rather than the ascending order.
                This is useful when you want to have similar sized batches but in random
                order. Default True.
            with_replacement (bool): When `random` is `True` then this sets the
                `replacement` argument. Default False.
        """
        self.data = data
        self.batch_size = batch_size
        self.similarity_sentence_key = similarity_sentence_key
        self.random = random
        self.with_replacement = with_replacement

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        sizes = []
        for sequence in self.data:
            number_similarity_sentences = len(sequence[self.similarity_sentence_key])
            sizes.append(number_similarity_sentences)
        sizes = torch.tensor(sizes)
        number_batches = len(self)
        batch_index_chunks = list(torch.chunk(torch.argsort(sizes, descending=True), number_batches))
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


def collate_variable_token_similarity_dataset(
    tokenizer: PreTrainedTokenizerFast,
    text_input_ids: str = "text_input_ids",
    text_attention_mask = "text_attention_mask",
    text_word_ids_mask = "text_word_ids_mask",
    similarity_sentence_input_ids = "label_definitions_input_ids",
    similarity_sentence_attention_mask = "label_definitions_attention_mask",
    label_key: str = "label_ids",
    attention_pad_id: int = 0,
        
) -> Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]:
    """
    This generates a function that converts a batch, 1 or more samples, of
    token level semantic similarity data whereby a token/MWE has a variable number 
    of semantically similar sentences of which one sentence is the correct 
    sentence, this correct sentence ID should be expressed via the label_ids 
    key value into a format suitable as input into a machine learning model 
    for either training or inference.

    The `attention_pad_id` is used for the following key-values:
    * `text_attention_mask`
    * `text_word_ids_mask`
    * `similarity_sentence_attention_mask`
    """

    def _collate(data: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Expects a batch, 1 or more samples, of token level semantic similarity data 
        and returns the data so that the dictionary keys are the same with each 
        value as a torch tensor containing a batch of data whereby the 
        dimensions are the same for the relevant keys;

        B represents the batch size. This is the number of text sequences.
        S represents the largest number of similarity sentences within one text
        sequence.
        T represents the largest token length for the text sample.
        ST represents the largest token length for the similarity sentences.

        `text` input attention IDs, and word ids mask keys will have a value
        of a torch tensor with the following shape; (B, T)
        `similarity_sentence` input and attention IDs keys will have a value
        of a torch tensor with the following shape; (B, S, ST)
        `label_key` tensor will have a value of a torch tensor with the following
        shape; (B). The value of the tensor is an integer representing the
        True sentence similarity label index.

        The `label_key` will contain no padding.

        The `similarity_sentence` input ids and attention masks could contain 
        entire vectors of padding. This is due to the fact that not all text 
        sequences will have the same number of similar sentences.

        Args:
            data (list[dict[str, Any]]): Data to transform.
        Returns:
            dict[str, torch.Tensor]: A dictionary that has the key names as in the
                given argument dictionaries , but each value will be a torch tensor.
        """

        def _pad_sequence(
            padding_token: int,
            padding_side: str,
            sequence: list[int],
            expected_sequence_length: int,
        ) -> torch.Tensor:
            tensor_value = torch.tensor(sequence, dtype=torch.long)
            pad_length = expected_sequence_length - tensor_value.size(-1)
            if pad_length == 0:
                return tensor_value
            pad_tensor = torch.tensor([padding_token] * pad_length, dtype=torch.long)
            if padding_side == "right":
                tensor_value = torch.hstack((tensor_value, pad_tensor))
            else:
                tensor_value = torch.hstack((pad_tensor, tensor_value))
            return tensor_value
        
        def _add_padding_lists(
            sequence: list[Any], expected_sequence_length: int
        ) -> list[Any]:
            pad_length = expected_sequence_length - len(sequence)
            copied_sequence = copy.deepcopy(sequence)
            if pad_length == 0:
                return copied_sequence
            else:
                for _ in range(pad_length):
                    copied_sequence.append([])
                return copied_sequence


        padding_side = tokenizer.padding_side
        text_padding_token = tokenizer.pad_token_id

        max_text_length = 0 # T
        max_number_similarity_sentences = 0 # S
        max_similarity_sentence_length = 0 # ST
        label_data = []

        for sample in data:
            text_length = len(sample[text_input_ids])
            if max_text_length < text_length:
                max_text_length = text_length
            
            similarity_sentences = sample[similarity_sentence_input_ids]
            number_similarity_sentences = len(similarity_sentences)

            if max_number_similarity_sentences < number_similarity_sentences:
                max_number_similarity_sentences = number_similarity_sentences
            for similarity_sentence in similarity_sentences:
                similarity_sentence_length = len(similarity_sentence)
                if max_similarity_sentence_length < similarity_sentence_length:
                    max_similarity_sentence_length = similarity_sentence_length
            
            label_data.append(sample[label_key])

        batched_data = defaultdict(list)
        for sample in data:
            for key, value in sample.items():
                if key == text_input_ids:
                    padded_text = _pad_sequence(text_padding_token, padding_side, value, max_text_length)
                    batched_data[key].append(padded_text)
                elif key == text_attention_mask or key == text_word_ids_mask:
                    padded_text_masks = _pad_sequence(attention_pad_id, padding_side, value, max_text_length)
                    batched_data[key].append(padded_text_masks)
                elif key == similarity_sentence_input_ids or key == similarity_sentence_attention_mask:
                    padded_similarity_sentence = _add_padding_lists(value, max_number_similarity_sentences)
                    padding_value = text_padding_token
                    if key == similarity_sentence_attention_mask:
                        padding_value = attention_pad_id
                    sample_padded_similarity_sentences = []
                    for similarity_sentence_value in padded_similarity_sentence:
                        padded_sentence_value = _pad_sequence(padding_value, padding_side, similarity_sentence_value, max_similarity_sentence_length)
                        sample_padded_similarity_sentences.append(padded_sentence_value)
                    stacked_sample_padded_similarity_sentences = torch.vstack(sample_padded_similarity_sentences).unsqueeze(0)
                    batched_data[key].append(stacked_sample_padded_similarity_sentences)

        batched_tensor_data: dict[str, torch.Tensor] = {}
        for key, value in batched_data.items():
            batched_tensor_data[key] = torch.vstack(value)
        batched_tensor_data[label_key] = torch.tensor(label_data, dtype=torch.long)
        return batched_tensor_data
    
    return _collate


def collate_token_negative_examples_classification_dataset(
    tokenizer: PreTrainedTokenizerFast,
    label_key: str = "labels",
    text_token_word_ids_mask_key: str = "text_word_ids_mask",
    text_keys: set[str] = set(["text_input_ids", "text_attention_mask"]),
    positive_samples_keys: set[str] = set(
        ["positive_input_ids", "positive_attention_mask"]
    ),
    negative_samples_keys: set[str] = set(
        ["negative_input_ids", "negative_attention_mask"]
    ),
    attention_mask_keys: set[str] = set(
        [
            "positive_attention_mask",
            "negative_attention_mask",
            "text_attention_mask",
            "text_word_ids_mask",
        ]
    ),
    label_pad_id: int = -100,
    attention_pad_id: int = 0,
) -> Callable[[list[dict[str, list[int]]]], dict[str, torch.Tensor]]:
    """
    This generates a function that converts a batch, 1 or more samples, of
    token level semantic similarity data (whereby a token/MWE has 1 positive
    similar sentence and N negative/dis-similar sentences) into a a format
    suitable as input into a machine learning model for either training or
    inference.

    Args:

        text_keys (set[str]): All of the text sample keys, should not include
            the `text_token_word_ids_mask_key`.
            Default set([text_input_ids, text_attention_mask])
    """

    def _collate(data: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """
        Expects a batch, 1 or more samples, of token level semantic similarity data
        and returns the data so that the dictionary keys are the same with one
        additional key `label_key` with each value as a torch tensor containing
        a batch of data whereby the dimensions are the same for the relevant keys;

        B represents the batch size. This is the number of text sequences.
        M represents the largest number of positive samples within one text
        sequence.
        N represents the largest number of negative samples for a positive sample
        within the batch of sequences (B).
        T represents the largest token length for the text sample.
        PT represents the largest token length for the positive text samples.
        NT represents the largest token length for the negative text samples.

        `positive` input and attention IDs keys will have a value
        of a torch tensor with the following shape; (B, M, PT)
        `negative` input and attention IDs keys will have a value
        of a torch tensor with the following shape; (B, M, N, NT)
        `text` input, and attention mask will have a value
        of a torch tensor with the following shape; (B, T)
        `text` token word ids mask will have a value of a torch tensor with the
        following shape; (B, M, T)
        `label_key` tensor will have a value of a torch tensor with the following
        shape; (B, M). The value of the tensor is an integer representing the
        True label index. In all cases this will be 0 unless it is a sample to
        ignore then it will be the `label_pad_id`.

        The additional 1 on the N for the `label_key` is the positive sample whereby
        the positive example is expected to be the first value.

        All the keys apart from the `text` input, and attention mask can contain
        entire vectors of tokenizer, attention, or label pad/masking IDs. This is
        due to the fact that not all text sequences will have the same number of
        positive or negative samples.

        Args:
            data (list[dict[str, list[int]]]): The outer list is the batch size
                and each dictionary should contain the same key names.
        Returns:
            dict[str, torch.Tensor]: A dictionary that has the key names as in the
                given argument dictionaries with the addition of `label_key`, but
                each value will be a torch tensor.
        """

        def _pad_sequence(
            padding_token: int,
            padding_side: str,
            sequence: list[int],
            expected_sequence_length: int,
        ) -> torch.Tensor:
            tensor_value = torch.tensor(sequence, dtype=torch.long)
            pad_length = expected_sequence_length - tensor_value.size(-1)
            if pad_length == 0:
                return tensor_value
            pad_tensor = torch.tensor([padding_token] * pad_length, dtype=torch.long)
            if padding_side == "right":
                tensor_value = torch.hstack((tensor_value, pad_tensor))
            else:
                tensor_value = torch.hstack((pad_tensor, tensor_value))
            return tensor_value

        def _add_padding_lists(
            sequence: list[Any], expected_sequence_length: int
        ) -> list[Any]:
            pad_length = expected_sequence_length - len(sequence)
            copied_sequence = copy.deepcopy(sequence)
            if pad_length == 0:
                return copied_sequence
            else:
                for _ in range(pad_length):
                    copied_sequence.append([])
                return copied_sequence

        padding_side = tokenizer.padding_side
        text_padding_token = tokenizer.pad_token_id

        largest_number_positive_samples = 0  # The M value
        largest_number_of_negative_samples = 0  # The N value
        largest_positive_token_sequence = 0  # The PT value
        largest_negative_token_sequence = 0  # The NT value
        largest_text_token_sequence = 0  # The T value
        batch_size = len(data)

        number_of_negatives_per_positive_sample = defaultdict(defaultdict)

        for batch_index, instance in enumerate(data):
            positive_sample_length_tested = False
            negative_sample_length_tested = False
            text_sample_length_tested = False
            for key, value in instance.items():
                # Positive sample lengths
                if key in positive_samples_keys and not positive_sample_length_tested:
                    number_positive_samples = len(value)
                    if number_positive_samples > largest_number_positive_samples:
                        largest_number_positive_samples = number_positive_samples
                    for positive_sample in value:
                        positive_sample_length = len(positive_sample)
                        if positive_sample_length > largest_positive_token_sequence:
                            largest_positive_token_sequence = positive_sample_length
                    positive_sample_length_tested = True
                # Negative sample lengths
                if key in negative_samples_keys and not negative_sample_length_tested:
                    number_positive_samples = len(value)
                    if not number_positive_samples:
                        continue
                    for positive_sample_index, positive_sample in enumerate(value):
                        number_negative_samples = len(positive_sample)

                        number_of_negatives_per_positive_sample[batch_index][
                            positive_sample_index
                        ] = number_negative_samples

                        if number_negative_samples > largest_number_of_negative_samples:
                            largest_number_of_negative_samples = number_negative_samples
                        for negative_sample in positive_sample:
                            negative_sample_length = len(negative_sample)
                            if negative_sample_length > largest_negative_token_sequence:
                                largest_negative_token_sequence = negative_sample_length
                    negative_sample_length_tested = True
                # Text sample lengths
                if key in text_keys and not text_sample_length_tested:
                    text_sample_length = len(value)
                    if text_sample_length > largest_text_token_sequence:
                        largest_text_token_sequence = text_sample_length
                    text_sample_length_tested = True

        labels_tensor = torch.zeros(
            (batch_size, largest_number_positive_samples),
            dtype=torch.long,
        )
        labels_tensor += label_pad_id

        for batch_index in number_of_negatives_per_positive_sample:
            for (
                instance_index,
                number_negative_samples,
            ) in number_of_negatives_per_positive_sample[batch_index].items():
                labels_tensor[batch_index][instance_index] = 0

        batched_dict = defaultdict(list)

        for instance in data:
            for key, value in instance.items():
                if key in text_keys:
                    padding_token = text_padding_token
                    if key in attention_mask_keys:
                        padding_token = attention_pad_id
                    batched_dict[key].append(
                        _pad_sequence(
                            padding_token,
                            padding_side,
                            value,
                            largest_text_token_sequence,
                        )
                    )
                elif (
                    key == text_token_word_ids_mask_key or key in positive_samples_keys
                ):
                    padded_value = _add_padding_lists(
                        value, largest_number_positive_samples
                    )

                    positive_padding_id = attention_pad_id
                    if key not in attention_mask_keys:
                        positive_padding_id = text_padding_token

                    largest_token_sequence = largest_positive_token_sequence
                    if key == text_token_word_ids_mask_key:
                        largest_token_sequence = largest_text_token_sequence

                    for positive_key_value_index in range(len(padded_value)):
                        positive_key_value = padded_value[positive_key_value_index]
                        padded_positive_key_value = _pad_sequence(
                            positive_padding_id,
                            padding_side,
                            positive_key_value,
                            largest_token_sequence,
                        )
                        padded_value[positive_key_value_index] = (
                            padded_positive_key_value
                        )

                    stacked_padded_value = torch.vstack(padded_value).unsqueeze(0)
                    batched_dict[key].append(stacked_padded_value)
                elif key in negative_samples_keys:
                    padded_value = _add_padding_lists(
                        value, largest_number_positive_samples
                    )
                    negative_padding_id = attention_pad_id
                    if key not in attention_mask_keys:
                        negative_padding_id = text_padding_token

                    for positive_key_value_index in range(len(padded_value)):
                        positive_key_value = padded_value[positive_key_value_index]
                        padded_positive_key_value = _add_padding_lists(
                            positive_key_value, largest_number_of_negative_samples
                        )
                        for negative_key_value_index in range(
                            len(padded_positive_key_value)
                        ):
                            negative_key_value = padded_positive_key_value[
                                negative_key_value_index
                            ]
                            padded_negative_key_value = _pad_sequence(
                                negative_padding_id,
                                padding_side,
                                negative_key_value,
                                largest_negative_token_sequence,
                            )
                            padded_positive_key_value[negative_key_value_index] = (
                                padded_negative_key_value
                            )
                        padded_value[positive_key_value_index] = torch.vstack(
                            padded_positive_key_value
                        ).unsqueeze(0)
                    stacked_padded_value = torch.vstack(padded_value).unsqueeze(0)
                    batched_dict[key].append(stacked_padded_value)

        tensor_batched_dict = {}
        for key, value in batched_dict.items():
            stacked_value = torch.vstack(value)
            tensor_batched_dict[key] = stacked_value
        tensor_batched_dict[label_key] = labels_tensor

        return tensor_batched_dict

    return _collate


def collate_token_classification_dataset(
    tokenizer: PreTrainedTokenizerFast,
    label_keys: set[str] = set(["labels", "word_ids"]),
    attention_mask_keys: set[str] = set(["attention_mask"]),
    label_pad_id: int = -100,
    attention_pad_id: int = 0,
) -> Callable[[list[dict[str, list[int]]]], dict[str, torch.Tensor]]:
    """
    This generates a function that converts a batch, 1 or more samples, of
    token classification data into format a suitable as input into a machine
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
            key: max(tensor_lengths)
            for key, tensor_lengths in key_tensor_lengths.items()
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
