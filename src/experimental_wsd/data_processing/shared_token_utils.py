"""
These methods are to be used for numerous token level similarity tasks, e.g. 
tasks at the token/Multi Word Expression level whereby given the token which is 
the most similar text from a list of given texts. Examples of this type of 
task is Word Sense Disambiguation with a setup like, 
[Blevins and Zettlemoyer 2020](https://aclanthology.org/2020.acl-main.95/) and 
[Harsh Kohli 2021](https://arxiv.org/pdf/2105.10146)
"""
from collections import defaultdict
from random import shuffle as random_shuffle
from typing import Any, Callable, Literal

import transformers
import wn

from experimental_wsd.wordnet_utils import (
    get_definition,
    get_negative_wordnet_sense_ids,
    get_normalised_mwe_lemma_for_wordnet,
)
from experimental_wsd.data_processing import processed_usas_utils

def usas_map_to_definitions(
    data: dict[str, list[str] | list[list[str]]],
    usas_mapper: dict[str, str]
) -> dict[str, list[str] | list[list[str]]]:
    """
    """
    usas_definitions = {"label_definitions": []}
    usas_labels = data["usas_labels"]
    for usas_label in usas_labels:
        usas_definition = usas_mapper[usas_label]
        usas_definitions["label_definitions"].append(usas_definition)
    return usas_definitions


def usas_join_positive_negative_labels(data: dict[str, Any], randomize: bool = True) -> dict[str, Any]:
    usas_label = data["usas"]
    negative_usas_labels = data["negative_usas"]
    combined_labels = [usas_label, *negative_usas_labels]
    label_dict = {
        "usas_labels": combined_labels,
        "label_ids": 0
    }
    if randomize:
        shuffle_index = list(range(len(combined_labels)))
        random_shuffle(shuffle_index)
        tmp_combined_labels = []
        for index, shuffled_index in enumerate(shuffle_index):
            if shuffled_index == 0:
                label_dict["label_ids"] = index
            tmp_combined_labels.append(combined_labels[shuffled_index])
        label_dict["usas_labels"] = tmp_combined_labels

    return label_dict

def usas_samples_to_single_sample(data: dict[str, Any]) -> dict[str, Any]:
    """
    """

    exploded_data = {
        key: [] for key in data
    }
    for batch_index, all_token_usas_tags in enumerate(data["usas"]):
        for token_index, token_usas_tags in enumerate(all_token_usas_tags):
            for usas_tag in token_usas_tags:
                exploded_data["id"].append(data["id"][batch_index])
                exploded_data["text"].append(data["text"][batch_index])
                #exploded_data["tokens"].append(data["tokens"][batch_index][token_index])
                exploded_data["usas_token_offsets"].append(data["usas_token_offsets"][batch_index][token_index])
                exploded_data["usas"].append(usas_tag)
                exploded_data["negative_usas"].append(data["negative_usas"][batch_index][token_index])
    return exploded_data


def map_negative_usas_labels(
        mosaic_usas_sentence_instance: dict[str, Any],
        positive_usas_key: str,
        negative_usas_key: str,
        usas_weighting: dict[str, float],
        use_weights: bool
) -> dict[str, str]:
    """
    A HuggingFace Dataset mapper function which should be ran in non-batch mode.
    """
    all_positive_usas_labels = mosaic_usas_sentence_instance[positive_usas_key]
    

    all_negative_usas_labels = [[] for _ in all_positive_usas_labels]
    if negative_usas_key in mosaic_usas_sentence_instance:
        all_negative_usas_labels = mosaic_usas_sentence_instance[negative_usas_key]

    new_all_negative_usas_labels = []
    for negative_token_usas_tags, positive_token_usas_tags in zip(all_negative_usas_labels, all_positive_usas_labels):
        new_negative_usas_tag = processed_usas_utils.random_negative_usas_label(positive_token_usas_tags, negative_token_usas_tags, usas_weighting, use_weights)
        new_all_negative_usas_labels.append([new_negative_usas_tag, *negative_token_usas_tags])

    return {negative_usas_key: new_all_negative_usas_labels}


def remove_duplicate_list_of_list_entries_while_maintaining_order(data: dict[str, Any], key: str) -> dict[str, Any]:
    """
    Given a dataset sample, it will de-duplicate the data that is associated with 
    the given key. The data to de-duplicate should be a list of a list of Any hashable 
    value, the de-duplicated version of this data is de-duplicated at the inner list 
    level, e.g.

    Input:
    `key`: [[0,1,1,2,3], [1,2,3,3]]

    Output:
    `key`: [[0,1,2,3], [1,2,3]]

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.
    """
    all_token_usas_tags = data[key]
    all_de_duplicated_usas_tags = []

    for token_usas_tags in all_token_usas_tags:
        unique_tags = set()
        de_duplicated_tags = []
        for usas_tag in token_usas_tags:
            if usas_tag in unique_tags:
                continue
            de_duplicated_tags.append(usas_tag)
            unique_tags.add(usas_tag)
        all_de_duplicated_usas_tags.append(de_duplicated_tags)
    return {key: all_de_duplicated_usas_tags}
    


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
    `labels`: A list of a list of WordNet sense keys that represent the gold labels 
        for the given annotation, e.g. `[[`carrousel%1:06:01::`]]`. 
        list[list[str]], it can be the case per annotation there is more than 
        one true/gold label.
    `sense_labels`: A list of a list of WordNet sense IDs that represent the 
        Sense ID of the labels, e.g. [[`omw-en-carrousel-02966372-n`]]. 
        list[list[str]]. This key will not exist if `word_net_sense_getter` is None.

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
    labels: list[list[str]] = []
    sense_labels: list[list[str]] = []

    for token in wsl_instance["tokens"]:
        token_text.append(token["raw"])
    for annotation in wsl_instance["annotations"]:
        label_annotations = annotation["labels"]
        if not label_annotations:
            continue
        sample_labels: list[str] = []
        sample_sense_labels: list[str] = []
        for label in label_annotations:
            token_offsets = annotation["token_off"]
            number_of_token_offsets = len(token_offsets)
            assert number_of_token_offsets > 0
            start_token_offset = token_offsets[0]
            end_token_offset = start_token_offset
            if len(token_offsets) > 1:
                end_token_offset = token_offsets[-1]
            end_token_offset += 1
            sample_labels.append(label)
            

            if word_net_sense_getter is not None:
                label_sense = word_net_sense_getter(label)
                if label_sense is None:
                    raise ValueError(
                        "The sense cannot be found for this label "
                        f"{label} which should not be the case."
                    )
                sample_sense_labels.append(label_sense.id)
        lemmas.append(annotation["lemma"])
        pos_tags.append(annotation["pos"])
        token_start_end_offsets.append((start_token_offset, end_token_offset))
        labels.append(sample_labels)
        sense_labels.append(sample_sense_labels)

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


def map_and_flatten_token_sense_labels(
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
    text_with_annotations: dict[str, list[str| None | list[str]]],
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
    Given a data sample, sample being here a text with multiple tokens that have 
    annotations associated to them which in this case are a sense ID, 
    containing at least the following key-values:
    * `sense_id_key` (list[str] | list[list[str]]) - The list of positive word net 
        sense ID for the sample. e.g. list[`omw-en-carrousel-02966372-n`]. It can 
        also be a list[list[str]], e.g. list[list[`omw-en-carrousel-02966372-n`]] 
        allowing for more than one positive label to be associated to a token.
    * `lemma_key` (list[str]) - The list of lemmas.
    * `pos_tag_key` (list[str | None]) - The list of POS tags, the list can
        contain None values for unknown POS tag values.
    All of the lists above should be the same length as each index value should
    be associated with each other, e.g. lemma[0] should be the lemma of the
    POS tag and positive word net sense ID at index 0.


    It will return all of the negative Wordnet sense ids, 
    e.g. [`omw-en-carrousel-02966372-n`], for this sample based on
    all of the senses that are associated to the (lemma, POS tag) which are not
    the positive word net sense ID.

    A HuggingFace Datasets mapper function which be ran in non-batch mode.

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
        dict[str, list[list[str]]]: A dictionary containing 1 key named 
            `negative_sense_id_key` with the following as it's value, 
            negative Wordnet sense IDs in Wordnet order, meaning the first 
            sense ID should be the most likely for the given (lemma, POS tag). 
            There are a list of negative Wordnet sense IDs per a positive 
            sense ID.
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
        if not isinstance(positive_sense_id, list):
            positive_sense_id = [positive_sense_id]
        negative_sense_ids = get_negative_wordnet_sense_ids(
            lemma,
            pos_tag,
            positive_sense_id,
            word_net_lexicon,
            get_random_sense=get_random_sense,
        )
        all_negative_sense_ids.append(negative_sense_ids)

    return {negative_sense_id_key: all_negative_sense_ids}

def sample_to_a_sense(data: dict[str, Any]) -> dict[str, Any]:
    """
    Given data with the following keys, whereby as we are running this as a 
    HuggingFace map function in batch mode each one of these key-values 
    will be wrapped around an outer list representing the batch sample:
    `text`: A list of token texts that represents the contextualized text.
        list[str].
    `lemmas`: A list of lemmas which represent the lemmas of each annotation.
        list[str]
    `pos_tags`: A list of POS tags which represent the POS tags of each annotation.
        list[str | None]. When None we do not know the POS tag of the label data.
    `token_offsets`: A list of tuples which contain token start and end indexes
        for each annotation. One for each annotation. list[tuple[int, int]]
    `labels`: A list of a list of WordNet sense keys that represent the gold labels 
        for the given annotation, e.g. `[[`carrousel%1:06:01::`]]`. 
        list[list[str]], it can be the case per annotation there is more than 
        one true/gold label.
    `sense_labels`: A list of a list of WordNet sense IDs that represent the 
        Sense ID of the labels, e.g. [[`omw-en-carrousel-02966372-n`]]. 
        list[list[str]].
    `negative_labels`: A list of a list of negative Wordnet sense IDs associated 
        to the positive sense IDs.
    
    It will return this data with the same keys but all values apart from the 
    `text` and `negative_labels` will be represented as a String as the labels 
    will have been exploded (term used within the Pandas ecosystem) so that we have 
    one sample per a positive label.

    `text`: A list of token texts that represents the contextualized text.
        list[str].
    `lemmas`: The lemma of the token associated to the positive label. str.
    `pos_tags`: A POS tag, which can be None if not known, associated to the 
        the positive label. str | None.
    `token_offsets`: token start and end indexes for the token associated to the 
        positive label. tuple[int, int]
    `labels`: A WordNet sense keys that represent the gold label e.g. 
        `carrousel%1:06:01::`. str.
    `sense_labels`: A WordNet sense IDs that represent the Sense ID of the 
        label, e.g. `omw-en-carrousel-02966372-n`. str.
    `negative_labels`: A list of negative Wordnet sense IDs associated 
        to the positive sense ID.

    A HuggingFace Datasets mapper function which be ran in batch mode.

    Args:
        data (dict[str, Any]): The data dictionary.
    Returns:
        dict[str, Any]: The data dictionary which has been exploded so that 
            each sample contains one sense of data.
    """

    exploded_data = {
        key: [] for key in data
    }
    for batch_index, sample_sense_labels in enumerate(data["sense_labels"]):
        for sample_index, sense_labels in enumerate(sample_sense_labels):
            for sense_label_index, sense_label in enumerate(sense_labels):
                exploded_data["text"].append(data["text"][batch_index])
                exploded_data["lemmas"].append(data["lemmas"][batch_index][sample_index])
                exploded_data["pos_tags"].append(data["pos_tags"][batch_index][sample_index])
                exploded_data["token_offsets"].append(data["token_offsets"][batch_index][sample_index])
                exploded_data["sense_labels"].append(sense_label)
                exploded_data["labels"].append(data["labels"][batch_index][sample_index][sense_label_index])
                exploded_data["negative_labels"].append(data["negative_labels"][batch_index][sample_index])
    return exploded_data


def join_positive_negative_labels(data: dict[str, Any], randomize: bool = True) -> dict[str, Any]:
    """
    Given data with the following key names and values:
    `text`: A list of token texts that represents the contextualized text.
        list[str].
    `lemmas`: The lemma of the token associated to the positive label. str.
    `pos_tags`: A POS tag, which can be None if not known, associated to the 
        the positive label. str | None.
    `token_offsets`: token start and end indexes for the token associated to the 
        positive label. tuple[int, int]
    `labels`: A WordNet sense keys that represent the gold label e.g. 
        `carrousel%1:06:01::`. str.
    `sense_labels`: A WordNet sense IDs that represent the Sense ID of the 
        label, e.g. `omw-en-carrousel-02966372-n`. str.
    `negative_labels`: A list of negative Wordnet sense IDs associated 
        to the positive sense ID.
    
    It will return two key names and values:

    `label_sense_ids`: The combination of `sense_labels` and `negative_labels`
        whereby when combined they are they randomized if `randomize` is `True`.
    `label_ids`: The index of the correct/positive sense ID within the list of 
        label sense IDs.

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.

    Args:
        data (dict[str, Any]): The data that contains the expected input.
        randomize (bool): If True then the combined `sense_labels` and `negative_labels` 
            will be randomized else they will not. Default is `False`.
    Returns:
        dict[str, Any]: The expected output.
    """
    sense_labels = data["sense_labels"]
    negative_sense_labels = data["negative_labels"]
    combined_labels = [sense_labels, *negative_sense_labels]
    label_dict = {
        "label_sense_ids": combined_labels,
        "label_ids": 0
    }
    if randomize:
        shuffle_index = list(range(len(combined_labels)))
        random_shuffle(shuffle_index)
        tmp_combined_labels = []
        for index, shuffled_index in enumerate(shuffle_index):
            if shuffled_index == 0:
                label_dict["label_ids"] = index
            tmp_combined_labels.append(combined_labels[shuffled_index])
        label_dict["label_sense_ids"] = tmp_combined_labels

    return label_dict

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


def token_word_id_mask(
    data: dict[str, Any],
    word_ids_key: str,
    token_offsets_key: str,
    word_id_mask_key: str,
) -> dict[str, list[list[Literal[0, 1]]]]:
    """
    Creates a token id mask for a token offset based off the word ids from the 
    tokenizer output of the given text. The mask represents 1's for tokens that 
    are within the token offset and 0's for tokens outside the offset.

    Example;
    test_data = {
        word_ids_key: [None, 0, 1, 1, 2, 2, 2, 3, None],
        token_offsets_key: [0,3]
    }

    expected_output = {
        word_id_mask_key: 
            [0,1,1,1,1,1,1,0,0]
    }
    Whereby we can see that the word 1 and 2 are represented by many sub word
    tokens.

    The word id mask is useful within a neural network to easily
    identify the sub word tokens that make up a token offset
    (usually a whole word or a Multi Word Expression).

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.

    Args:
        data (dict[str, Any): A dictionary that contains
            the `word_ids_key` and `token_offsets_key`.
        word_ids_key (str): The key that contains word ids, a list of either
            None or integer values.
        token_offsets_key (str): The key that contains the token offset, each 
            offset should contain two integers representing
            the start and end word indexes of a word or multi word expression.
            For example `[0, 1]` would represent the first word in the text.
        word_id_mask_key (str): The key name of the returned token id mask.
    Returns:
        dict[str, Any]: A dictionary with the key `word_id_mask_key`
            which contains a token id mask for the given token offset.
    """
    token_offset_word_id_mask = []
    word_ids = data[word_ids_key]
    token_offset = data[token_offsets_key]
    if not token_offset:
        return {word_id_mask_key: token_offset_word_id_mask}

    start_offset, end_offset = token_offset
    relevant_word_ids = set(range(start_offset, end_offset))
    
    for word_id in word_ids:
        if word_id in relevant_word_ids:
            token_offset_word_id_mask.append(1)
        else:
            token_offset_word_id_mask.append(0)

    return {word_id_mask_key: token_offset_word_id_mask}


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

    A HuggingFace Datasets mapper function which should be ran in non-batch mode.

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

def filter_sequences_too_long(
        data: dict[str, list[str]],
        key: str,
        length: int
) -> bool:
    """
    Returns False if the length of the value of the `key` from `data` is longer 
    than or equal to the `length`.

    HuggingFace dataset filter.

    Args:
        data (dict[str, list[str]]): The data that contains the `key` and it's 
            value.
        key (str): The key whole value cannot be as long or longer than `length`
        length (int): The maximum length.
    Returns:
        bool: True if the key-value is less than `length` else False.
    """

    if len(data[key]) >= length:
        return False
    return True

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
    * `text_key` (list[str] | list[list[str]]): The texts to be tokenized.
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

#def explode_key_value(data: dict[str, Any],
#                      explode_key: str,
#                      ) -> list[dict[str, Any]]:
#    """
#    Like the Pandas (and other data science tools) this explode function 
#    takes a explode_key name within the data and transforms each list like value 
#    in the explode_key value into it's own sample.
#
#    We assume the following for all key value's that are not the explode_key:
#    * If the value is a list of values that is not equal in length to the explode_key 
#        value then it is to be replicated as is.
#    * If the value is not a list it is to be replicated as is.
#    * If the value is a 
#    """
#
#    return {}