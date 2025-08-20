from collections import Counter
from typing import Any, Literal

import transformers


def get_align_labels_with_tokens(
    labels: list[int], word_ids: list[int | None], label_pad_id: int = -100
) -> list[int]:
    """
    Given a list of label ids matching at the word level and the corresponding
    word ids matching at the sub-word level, it returns a list of label ids that
    match at this expanded sub-word level.

    E.g.:
    `labels` = [0,1,0]
    `word_ids` = [None, 0, 1, 1, 1, 1, 1, 1, 1, 2, None]
    `label_pad_id` = -100
    Returns: [-100, 0, 1, 1, 1, 1, 1, 1, 1, 0, -100]

    Reference:
    https://huggingface.co/learn/llm-course/chapter7/2?fw=pt#fine-tuning-the-model-with-the-trainer-api

    Whereby -100 indicates an index to ignore by the loss function. If not -100
    then the Pytorch loss functions will not ignore that label id.

    Args:
        labels (list[int]): The label ids matching at the word level.
        word_ids (list[int | None]): A list of word indexes at the sub word level.
            When None then this indicates a special token that should be padded
            with the `label_pad_id`.
        label_pad_id (int): The value to give to list indexes that reflect a
            special token. Default is -100 which corresponds to the label id
            that Pytorch ignores in the loss function.
    Returns:
        list[int]: The label ids matching at the sub-word level whereby special
            tokens are assigned the `label_pad_id` value.
    """
    new_labels = []
    current_word_id: None | int = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(label_pad_id)
        elif word_id != current_word_id:
            # Start of a new word!
            current_word_id = word_id
            new_labels.append(labels[word_id])
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)
    assert len(new_labels) == len(word_ids)
    return new_labels


def tokenize_pre_processing(
    batched_text_and_labels: dict[str, list[list[str]] | list[list[int]]],
    tokenizer: transformers.PreTrainedTokenizerFast,
    label_pad_id: int = -100,
    align_labels_with_tokens: bool = False,
) -> dict[str, list[list[int]]]:
    """
    Given a dictionary that contains `text` and `is_content_word` keys
    containing a list of of a list of tokens and a list of a list of boolean
    values respectively whereby the outer list represents a batch of samples.
    This returns the data pre-processed using the given tokenizer
    with the following key-values:
    * input_ids -> Truncated tokenized tokens. Even though the tokens are already
        tokenized the tokenizer sub-word tokenizes those tokens. A list of integers.
    * attention_mask -> The attention mask. Should be a list of integers all 1's.
    * word_ids -> A list of integers representing a mapping from sub-word token
        to work level token. Any special tokens have a value of `label_pad_id`.
    * labels -> The integer value of `is_content_word`, if a special token exists
        this will be given the value of `label_pad_id` if `align_labels_with_tokens`
        is True.
    All samples for each key should be of the same length, except for the labels.

    A HuggingFace Datasets mapper function which should be ran in batch mode.

    This is useful for token classification pre-processing.

    Args:
        batched_text_and_labels (dict[str, list[list[str]] | list[list[int]]]): Data
            to be pre-processed. Has to contain the following keys; `text` and
            `is_content_word`
        tokenizer (transformers.PreTrainedTokenizerFast): The tokenizer to
            pre-process the tokens.
        label_pad_id (int): The integer value to give to special tokens that should
            be ignored as a label. Only applicable when
            `align_labels_with_tokens` is True as this is when special tokens
            needs to be considered. Default -100.
        align_labels_with_tokens (bool): Whether the labels should be aligned
            with the sub-word tokens they represent, this uses the function
            `get_align_labels_with_tokens`. If this is True then the length of
            the labels will be the same as all other returned key values.
            Default False.
    Returns:
        dict[str, list[list[int]]]: The data pre-processed.
    """
    tokenized_inputs = tokenizer(
        batched_text_and_labels["text"], truncation=True, is_split_into_words=True
    )
    batched_is_content_word_labels = batched_text_and_labels["is_content_word"]
    batched_labels = []
    batched_word_ids = []
    for batch_index, is_content_word_labels in enumerate(
        batched_is_content_word_labels
    ):
        word_ids = tokenized_inputs.word_ids(batch_index)
        temp_word_ids = []
        for word_id in word_ids:
            if word_id is None:
                temp_word_ids.append(label_pad_id)
            else:
                temp_word_ids.append(word_id)
        batched_word_ids.append(temp_word_ids)
        if align_labels_with_tokens:
            batched_labels.append(
                get_align_labels_with_tokens(
                    is_content_word_labels, word_ids, label_pad_id
                )
            )
        else:
            batched_labels.append(is_content_word_labels)
    tokenized_inputs["labels"] = batched_labels
    tokenized_inputs["word_ids"] = batched_word_ids
    return tokenized_inputs


def get_pre_processed_label_statistics(
    batched_text_and_labels: dict[str, list[list[str]] | list[list[int]]],
    label_key: str,
    label_value_to_ignore: Any,
) -> dict[str, list[Any]]:
    """
    Returns a dictionary with one key `label_counts` that contains all of the
    labels as one list. This list can then be given to a Counter object to
    create a dictionary of unique labels to counts.

    This is a useful function for getting label weights for loss functions.

    A HuggingFace Datasets mapper function which be ran in batch mode.

    NOTE: this should be refactored to be more efficient and only return label
    counts rather than all labels.

    Args:
        batched_text_and_labels (dict[str, list[list[str]] | list[list[int]]]): The
            data that contains the `label_key`
        label_key (str): The key that represents the labelled data.
        label_value_to_ignore (Any): A label value to ignore.
    Returns:
        dict[str, list[Any]]: A dictionary with one key, `label_counts`, that contains
            all of the label values in the given batched text and labels data
            as one list.
    """
    label_data = batched_text_and_labels[label_key]
    label_counter = Counter()
    for sample in label_data:
        label_counter.update(sample)
    if label_value_to_ignore in label_counter:
        del label_counter[label_value_to_ignore]
    return {"label_counts": list(label_counter.elements())}


def map_token_text_and_is_content_labels(
    wsl_instance: dict[str, Any],
) -> dict[str, list[str] | list[int]]:
    """
    Given a sample that comes from a `wsl.WSLSentence` it will return a dictionary
    of:
    `text`: A list of token texts
    `is_content_word`: A list of 1 or 0 whereby 1 indicates that the token text at that
        index is a content word and 0 indicates it is a non-content word.

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.

    Args:
        wsl_instance (dict[str, Any]): A wsl.WSLSentence in python dictionary
            format. This can be achieved using wsl.WSLSentence.model_dump().
            As a minimum this should be a dictionary with the key `tokens`
            which contains a list of dictionaries containing `raw` and `is_content_word`
            keys whereby `raw` is the token text and `is_content_word` is a boolean
            value determining if the token text is a content word or not.
    Returns:
        dict[str, list[str] | list[int]]: A dictionary of two keys; `text` and
            `is_content_word` which contain a list of token strings and a list
            of 1 or 0 values determining if the token strings are content tokens
            respectively.
    """
    token_text: list[str] = []
    is_content: list[int] = []
    for token in wsl_instance["tokens"]:
        token_text.append(token["raw"])
        is_content.append(int(token["is_content_word"]))
    return {"text": token_text, "is_content_word": is_content}


def map_empty_removal_token_values(
    data: dict[str, list[str | None] | list[list[str]]],
    filter_key: str,
    aligned_keys: list[str],
) -> dict[str, list[str | None] | list[list[str]]]:
    """
    Given a filter key, it is assumed that value in this key is a list as well
    all values associated to the aligned_keys, it will remove all values in that
    filter key if they are empty/evaluate to False, in addition all values that are
    empty their associated values in the aligned keys will also be removed. It
    will return a dictionary of filter and aligned keys with their associated
    filtered values.

    Args:
        data (dict[str, list[str | None] | list[list[str]]]): The data to be
            filtered.
        filter_key (str): The key to filter by
        aligned_keys (list[str]): The keys of data that have to be filtered if
            any data is filtered within the filter key.
    Returns:
        dict[str, list[str | None] | list[list[str]]]: The filtered data which
            will contain the filtered and aligned keys and the their associated
            filtered values.
    Raises:
        ValueError: If the lengths of the values of the filter and aligned keys
            are not the same.
    """
    expected_value_lengths = len(data[filter_key])
    for key in aligned_keys:
        if expected_value_lengths != len(data[key]):
            raise ValueError(
                f"The lengths of the lists of the aligned keys {aligned_keys} "
                "must be the same, but they are not for this "
                f"sample: {data}"
            )
    new_key_values = {key: [] for key in [*aligned_keys, filter_key]}
    for index, filter_value in enumerate(data[filter_key]):
        if not filter_value:
            continue
        new_key_values[filter_key].append(filter_value)
        for key in aligned_keys:
            new_key_values[key].append(data[key][index])
    return new_key_values


def token_word_id_mask(
    data: dict[str, list[list[int] | int | None]],
    word_ids_key: str,
    token_offsets_key: str,
    word_id_mask_key: str,
) -> dict[str, list[list[Literal[0, 1]]]]:
    """
    Creates a token id mask for each token offset based off the word ids whereby
    the token offsets represent word indexes and the word ids represent the
    word id for a given sub token.

    Example;
    test_data = {
        word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None],
        token_offsets_key: [[0,1], [0,3], [1,4]]
    }

    expected_output = {
        word_id_mask_key: [
            [0,1,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,0,0],
            [0,0,1,1,1,1,1,1,0]
        ]
    }
    Whereby we can see that the word 1 and 2 are represented by many sub word
    tokens.

    The word id mask created is useful within a neural network to easily
    identify the sub word tokens that make up a token offset
    (usually a whole word or a Multi Word Expression).

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.

    Args:
        data (dict[str, list[list[int | None]]]): A dictionary that contains
            the `word_ids_key` and `token_offsets_key`.
        word_ids_key (str): The key that contains word ids, a list of either
            None or integer values.
        token_offsets_key (str): The key that contains token offsets, a list
            of lists whereby the inner list should contain two integers representing
            the start and end word indexes of a word or multi word expression.
            For example `[[0, 1]]` would represent the first word in the text.
        word_id_mask_key (str): The key name of the returned token id mask.
    Returns:
        dict[str, list[list[Literal[0, 1]]]]: A dictionary with the key `word_id_mask_key`
            which contains a list of token id masks for each given token offset.
    """
    word_id_mask = []
    word_ids = data[word_ids_key]
    for token_offset in data[token_offsets_key]:
        start_offset, end_offset = token_offset
        relevant_word_ids = set(range(start_offset, end_offset))
        token_offset_word_id_mask = []
        for word_id in word_ids:
            if word_id in relevant_word_ids:
                token_offset_word_id_mask.append(1)
            else:
                token_offset_word_id_mask.append(0)
        word_id_mask.append(token_offset_word_id_mask)

    return {word_id_mask_key: word_id_mask}



