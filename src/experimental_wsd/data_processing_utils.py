from typing import Any

import transformers


def align_labels_with_tokens(
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


def tokenize_and_align_labels(
    batched_text_and_labels: dict[str, list[list[str]] | list[list[int]]],
    tokenizer: transformers.PreTrainedTokenizerFast,
    label_pad_id: int = -100,
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
        this will be given the value of `label_pad_id`.
    All samples for each key should be of the same length.

    A HuggingFace Datasets mapper function which should be ran in batch mode.

    This is useful for token classification pre-processing.

    Args:
        batched_text_and_labels (dict[str, list[list[str]] | list[list[int]]]): Data
            to be pre-processed. Has to contain the following keys; `text` and
            `is_content_word`
        tokenizer (transformers.PreTrainedTokenizerFast): The tokenizer to
            pre-process the tokens.
        label_pad_id (int): The integer value to give to special tokens that should
            be ignored as a label. Default -100.
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
        batched_labels.append(
            align_labels_with_tokens(is_content_word_labels, word_ids, label_pad_id)
        )
    tokenized_inputs["labels"] = batched_labels
    tokenized_inputs["word_ids"] = batched_word_ids
    return tokenized_inputs


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
