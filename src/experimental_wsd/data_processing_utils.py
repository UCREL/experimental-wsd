from collections import Counter, defaultdict
from typing import Any, Callable

import transformers
import wn

from experimental_wsd.wordnet_utils import (
    get_definition,
    get_negative_wordnet_sense_ids,
    get_normalised_mwe_lemma_for_wordnet,
)


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
    word_net_sense_getter: Callable[[str], wn.Sense | None] | None = None,
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
    `sense_labels`: A list of WordNet sense IDs that represent the Sense ID of
        the labels, e.g. [`omw-en-carrousel-02966372-n`]. list[str]. This key
        will not exist if `word_net_sense_getter` is None.

    NOTE: That compared to the original sample whereby each annotation can contain
    more than one label, when this function is applied in essence these labels
    are flattened therefore if an annotation contains more than one label then
    all of it's attributes, lemma, POS tag, token offsets are duplicated for
    each label.

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
        word_net_sense_getter (Callable[str, [wn.Sense | None]]): A callable
            that takes as input a sense key, e.g. `carrousel%1:06:01::`
            and returns a WordNet Sense if it can be found else None. NOTE that
            sense keys are specific to English WordNets I believe and they are not
            the same as Sense IDs like `omw-en-carrousel-02966372-n`. The best way
            to get this callable is through the function:
            `wn.compat.sensekey.sense_getter` see for more details:
            `https://wn.readthedocs.io/en/latest/api/wn.compat.sensekey.html`.
            If None then the `sense_labels` will not be in the returned dictionary.
            Default None.
    Returns:
        dict[str, list[str] | list[str | None] | list[tuple[int, int]]]: A
            dictionary of five keys; `text`, `lemmas`, `pos_tags`, `token_offsets`,
            `sense_labels`, and `labels`. NOTE that the length of the `text`
            will be different to the length of the other list values however
            all other list values should be the same length as they represent
            the annotations from the sample.
    """
    token_text: list[str] = []
    lemmas: list[str] = []
    pos_tags: list[str] = []
    token_start_end_offsets: list[tuple[int, int]] = []
    labels: list[str] = []
    sense_labels: list[str] = []

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

            if word_net_sense_getter is not None:
                label_sense = word_net_sense_getter(label)
                if label_sense is None:
                    raise ValueError(
                        "The sense cannot be found for this label "
                        f"{label} which should not be the case."
                    )
                sense_labels.append(label_sense.id)
    data = {
        "text": token_text,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "token_offsets": token_start_end_offsets,
        "labels": labels,
    }
    if word_net_sense_getter is not None:
        data["sense_labels"] = sense_labels

    return data


def map_negative_sense_ids(
    text_with_annotations: dict[str, list[str] | list[str | None]],
    word_net_lexicon: wn.Wordnet,
    sense_id_key: str,
    lemma_key: str = "lemma",
    pos_tag_key: str = "pos_tag",
    negative_sense_id_key: str = "negative_labels",
    get_random_sense: bool = False,
    pos_tag_mapper: dict[str, str] | None = None,
    normalise_mwe_lemma: bool = True,
) -> dict[str, list[str]]:
    """
    Given a data sample containing at least the following key-values:
    * `sense_id_key` (list[str]) - The list of positive word net sense ID for the sample.
    * `lemma_key` (list[str]) - The list of lemmas.
    * `pos_tag_key` (list[str | None]) - The list of POS tags, the list can
        contain None values for unknown POS tag values.
    All of the lists above should be the same length as each index value should
    be associated with each other, e.g. lemma[0] should be the lemma of the
    POS tag and positive word net sense ID at index 0.


    It will return all of the negative Wordnet sense ids for this sample based on
    all of the senses that are associated to the (lemma, POS tag) which are not
    the positive word net sense ID.

    Args:
        text_with_annotations (dict[str, list[str] | list[str | None]): The
            sample to get negative word net sense ids for.
        word_net_lexicon (wn.Wordnet): Wordnet lexicon to find the senses
            of the given lemma and POS tag.
        sense_id_key (str): The key in the sample that is associated to the
            positive/correct sense ID. The sense ID should be a Wordnet sense
            ID, e.g. `omw-en-become-02626604-v`.
        lemma_key (str): The key in the sample that is associated to the
            lemma. Default is "lemma".
        pos_tag_key (str): The key in the sample that is associated to the
            POS tag value. Default is "pos_tag".
        negative_sense_id_key (str): The key name to give to the newly generated
            negative sense IDs. Default is "negative_labels".
        get_random_sense (bool): If True for non-ambiguous lemma and pos tags
            a random sense ID is created as negative sense ID, else when False
            no negative sense ID will be given for that sample, e.g. returns an
            empty list. In essence if all Senses that are returned for the
            given lemma and POS tag is the positive sense and this is True, then
            a random sense ID is created as a negative sense ID. Default False.
        pos_tag_mapper (dict[str, str] | None): POS tagger mapper to use, if None then
            the original POS tags will be used. This can be essential if there
            is a mismatch in POS tagging schemes between the training data and the
            sense inventory. Default is None.
        normalise_mwe_lemma (bool): Whether to normalise the Multi Word Expressions (MWE)
            of the lemma. In SemCor dataset the lemma for MWE are normally represented
            with an `_` instead of spaces, but Wordnet represents MWEs with a
            whitespace, e.g. `New York` in SemCor the lemma would be `new_york`
            whereas in Wordnet it would be `new york`. Default is True.
    Returns:
        dict[str, list[str]]: A dictionary containing 1 key named `negative_sense_id_key`
            with the following as it's value, negative Wordnet sense IDs in
            Wordnet order, meaning the first sense ID should be the most
            likely for the given (lemma, POS tag).
    """

    lemmas = text_with_annotations[lemma_key]
    pos_tags = text_with_annotations[pos_tag_key]
    positive_sense_ids = text_with_annotations[sense_id_key]

    all_field_values = [lemmas, pos_tags, positive_sense_ids]
    first_field_length = len(lemmas)
    field_lengths = [
        len(field_values) == first_field_length for field_values in all_field_values
    ]
    if not all(field_lengths):
        raise ValueError(
            "The lengths of the lists of the lemmas, POS tags, and "
            "sense ids must be the same, but they are not for this "
            f"sample: {text_with_annotations}"
        )

    all_negative_sense_ids = []
    for lemma, pos_tag, positive_sense_id in zip(*all_field_values):
        if normalise_mwe_lemma:
            lemma = get_normalised_mwe_lemma_for_wordnet(lemma)
        if pos_tag_mapper:
            pos_tag = pos_tag_mapper[pos_tag]
        negative_sense_ids = get_negative_wordnet_sense_ids(
            lemma,
            pos_tag,
            positive_sense_id,
            word_net_lexicon,
            get_random_sense=get_random_sense,
        )
        all_negative_sense_ids.append(negative_sense_ids)

    return {negative_sense_id_key: all_negative_sense_ids}


def filter_empty_values(data: dict[str, list[Any]], key: str) -> bool:
    """
    Returns the value in the data for the given key when evaluated, in essence
    determines if the value for the key in the data is empty or not.

    HuggingFace dataset filter.

    Args:
        data (dict[str, list[Any]]): The data to be filtered.
        key (str): The key for the value in the data that will be evaluated on
            whether it is empty or not.
    Returns:
        bool: True if the value for the key in the data is empty or not.
    """
    if data[key]:
        return True
    return False


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
    data: dict[str, list[list[int | None]]],
    word_ids_key: str,
    token_offsets_key: str,
    word_id_mask_key: str,
) -> dict[str, list[list[int]]]:
    """ """
    return {}


def tokenize_key(
    data: dict[str, list[str] | list[list[str]]],
    tokenizer: transformers.PreTrainedTokenizerFast,
    text_key: str,
    output_key_prefix: str = "",
    add_word_ids: bool = True,
    is_split_into_words: bool = False,
    truncation: bool = False,
) -> dict[str, list[list[int]] | list[list[list[int]]]]:
    """
    Given data with the following key, values;
    * `text_key` (list[str] | list[list[str]]): The of texts to be tokenized.
        It can be a list of a list of texts in this case this is represented
        in the output having an additional outer list.
    It will tokenize the text with the given tokenizer and return the following
    key, values;
    * `{output_key_prefix}_input_ids` (list[list[int]] | list[list[list[int]]]):
        For each text a list integers representing the token indexes.
    * `{output_key_prefix}_attention_mask` (list[list[int]] | list[list[list[int]]]):
        For each text a list of 1 or 0s representing the attention mask for each token.
        In practice it is a list of 1s as no padding is performed.
    * `{output_key_prefix}_word_ids` (list[list[int | None]] | list[list[list[int | None]]]):
        For each text a list of ids representing the mapping between sub word
        token and the word it is linked too. The Word IDs for special tokens
        like <CLS> is `None`. Optional.

    If `output_key_prefix` is empty then they keys will be `input_ids` and
    `attention_mask`.

    A HuggingFace Datasets mapper function which should be ran in non-batch mode
    when the data to be tokenized contains a list of a list of texts else
    best ran in batch mode to make the most of the tokenizer. Batch mode can be
    ran with a list of a list of strings if and only if the inner list is a
    list of tokens and the `is_split_into_words` argument is set to True.

    Args:
        data (dict[str, list[str]]): Data containing the text to be tokenized.
        tokenizer (transformers.PreTrainedTokenizerFast): The tokenizer to
            tokenize the text with.
        text_key (str): The key in the `data` that represents the
            text to be tokenized.
        output_key_prefix (str): If not an empty string, "", then it will prefix
            the output keys like so `{output_key_prefix}_`. Default an
            empty string ("").
        add_word_ids (bool): Whether to include Word IDs in the output.
            Default True.
        is_split_into_words (bool): Sets the `is_split_into_words` argument
            for the tokenizer. If the text_key is already tokenized this should
            be True else False. Default False.
        truncation (bool): Sets the `truncation` argument for the tokenizer.
            When False it means that the tokenizer could produce token sequences
            that are longer than the maximum intended sequence length of the
            model associated to the given tokenizer. Default False.
    Returns:
        dict[str, list[list[int | None]] | list[list[list[int | None]]]]: The
            tokenized text represented by `input_ids` and `attention_mask`.
    """

    def _tokenize_text_list(
        text_list: list[str] | list[list[str]],
    ) -> dict[str, list[int]]:
        tokenized_text_output = tokenizer(
            text_list, truncation=truncation, is_split_into_words=is_split_into_words
        )
        tokenized_text = tokenized_text_output.data
        if output_key_prefix:
            tmp_tokenized_text = {}
            for key, value in tokenized_text.items():
                tmp_tokenized_text[f"{output_key_prefix}_{key}"] = value
            tokenized_text = tmp_tokenized_text

        if add_word_ids:
            all_word_ids = []
            # Special case handling for when the input text is just tokens
            # and is not batched.
            if is_split_into_words and isinstance(text_list[0], str):
                word_ids = tokenized_text_output.word_ids(0)
                all_word_ids = word_ids
            else:
                for text_index in range(len(text_list)):
                    word_ids = tokenized_text_output.word_ids(text_index)
                    all_word_ids.append(word_ids)
            word_ids_key = "word_ids"
            if output_key_prefix:
                word_ids_key = f"{output_key_prefix}_{word_ids_key}"
            tokenized_text[word_ids_key] = all_word_ids
        return tokenized_text

    if isinstance(data[text_key][0], list) and not is_split_into_words:
        tokenized_outputs = defaultdict(list)
        for text_list in data[text_key]:
            for key, value in _tokenize_text_list(text_list).items():
                tokenized_outputs[key].append(value)
        return dict(tokenized_outputs)
    else:
        tokenized_outputs = _tokenize_text_list(data[text_key])
        return tokenized_outputs


def map_to_definitions(
    data: dict[str, list[str] | list[list[str]]],
    sense_key: str,
    word_net_lexicon: wn.Wordnet,
    definition_key: str,
) -> dict[str, list[str] | list[list[str]]]:
    """
    The data should contain a key, `sense_key`, that either contains a list of
    sense IDs or a list of a list of sense IDs. This function will then return
    the definitions of these sense IDs using the given word net lexicon. These
    definitions are then returned as either a list of definitions or a list of
    a list of definitions depending on how the sense IDs were formatted. These
    sense definitions will be stored in the `definition_key` of the returned
    dictionary.

    Args:
        data (dict[str, list[str] | list[list[str]]]): The sample that contains
            sense IDs that will be converted 1-to-1 with the sense's definition.
        sense_key (str): The key within data that contains the sense id values.
        word_net_lexicon (wn.Wordnet): Wordnet lexicon used to get the definitions
            of the given sense ids.
        definition_key (str): The key within the returned dictionary that will
            contain the sense definitions.
    Returns:
        dict[str, list[str] | list[list[str]]]: The sense definitions contained
            within the key `definition_key`.
    """
    sense_definitions = {definition_key: []}

    for sense_ids in data[sense_key]:
        if isinstance(sense_ids, list):
            definitions = []
            for sense_id in sense_ids:
                definitions.append(get_definition(sense_id, word_net_lexicon))
            sense_definitions[definition_key].append(definitions)
        else:
            sense_id = sense_ids
            definition = get_definition(sense_id, word_net_lexicon)
            sense_definitions[definition_key].append(definition)

    return sense_definitions
