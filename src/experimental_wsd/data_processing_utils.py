from collections import Counter
from typing import Any

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


def map_token_sense_labels(
    wsl_instance: dict[str, Any],
) -> dict[str, list[str] | list[str | None] | list[tuple[int, int]]]:
    """
    Given a sample that comes from a `wsl.WSLSentence` it will return a dictionary
    of:
    `text`: A list of token texts that represents the contextualized text.
        list[str].
    `lemmas`: A list of lemmas which represent the lemmas of each annotation.
        list[str]
    `pos_tags`: A list of POS tags which represent the POS tags of each annotation.
        list[str | None]. When None we do not know the POS tag of the label data.
    `token_offsets`: A list of tuples which contain token start and end indexes
        for each annotation. One for each annotation. list[tuple[int, int]]
    `labels`: A list of WordNet sense keys that represent a gold label for the
        given annotation, e.g. `[`carrousel%1:06:01::`]`. One for each annotation.
        list[str]

    A HuggingFace Datasets mapper function which be ran in non-batch mode.

    Args:
        wsl_instance (dict[str, Any]): A wsl.WSLSentence in python dictionary
            format. This can be achieved using wsl.WSLSentence.model_dump().
            As a minimum this should be a dictionary with the key `tokens`
            which contains a list of dictionaries containing `raw` key whereby
            `raw` is the token text. The dictionary also has to contain another
            key `annotations` which contains a list of dictionaries containing
            `lemma` (str), `pos` (str | None),  `token_off`
            (list[int] each representing a token index that it relates too),
            and `labels` (list[str] the WordNet sense key [`carrousel%1:06:01::`])
            keys which represents the annotation data from the sample.
    Returns:
        dict[str, list[str] | list[str | None] | list[tuple[int, int]]]: A
            dictionary of five keys; `text`, `lemmas`, `pos_tags`, `token_offsets`,
            and `labels`. NOTE that the length of the `text` will be different to
            the length of the other list values however all other list values should
            be the same length as they represent the annotations from the sample.
    """
    token_text: list[str] = []
    lemmas: list[str] = []
    pos_tags: list[str] = []
    token_start_end_offsets: list[tuple[int, int]] = []
    labels: list[str] = []
    for token in wsl_instance["tokens"]:
        token_text.append(token["raw"])
    for annotation in wsl_instance["annotations"]:
        for label in annotation["labels"]:
            token_offsets = annotation["token_off"]
            number_of_token_offsets = len(token_offsets)
            assert number_of_token_offsets > 0
            start_token_offset = token_offsets[0]
            end_token_offset = start_token_offset
            if len(token_offsets) > 1:
                end_token_offset = token_offsets[-1]
            end_token_offset += 1
            token_start_end_offsets.append((start_token_offset, end_token_offset))
            labels.append(label)
            lemmas.append(annotation["lemma"])
            pos_tags.append(annotation["pos"])
    return {
        "text": token_text,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "token_offsets": token_start_end_offsets,
        "labels": labels,
    }
