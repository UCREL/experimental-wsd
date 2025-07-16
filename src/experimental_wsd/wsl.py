import copy
import re
from collections import Counter, defaultdict
from itertools import tee
from typing import Any, Callable, ClassVar, Iterator, Literal, Union

import wn
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure
from bokeh.transform import dodge
from datasets import DatasetDict
from pydantic import BaseModel, Field, field_validator, model_validator

from experimental_wsd.pos_constants import UniversalDepPOSTags


class WSLID(BaseModel):
    """
    This is a structured format of the sentence ID from the WSL dataset.
    """

    dataset_id: str = Field(examples=["senseval2"])
    document_index: int = Field(examples=[0, 1])
    sentence_index: int = Field(examples=[0, 1])

    WSLID_RE: ClassVar[re.Pattern] = re.compile(r"^([\w\d]+).d(\d+).s(\d+)$")

    def __str__(self) -> str:
        """
        Returns:
            str: `{self.dataset_id}.{self.document_index}.{self.sentence_index}`
                The original representation of the sentence ID from the WSL
                dataset for this instance.
        """
        return f"{self.dataset_id}.{self.document_index}.{self.sentence_index}"


class Token(BaseModel):
    """
    Validates each token in the WSL dataset.
    """

    _raw_description = "The token text."
    _pos_description = (
        "The POS tag of this token. The POS tag should be a "
        "Universal Dependency POS tag."
    )
    _offset_description = (
        "The character offsets that when paired with the associated context text "
        "represents this token. This is always a list of 2 integers representing "
        "the start and end offsets."
    )
    _is_content_word_description = (
        "If True then the token is associated with a semantic label, this "
        "could be part of a Multi Word Expression or only part of the token "
        "is associated with a semantic label (sub-words). In WSD most content "
        "words are the only words that are tagged with semantic labels, whereby "
        "content words are typically nouns, verbs, adjectives, and adverbs."
    )

    raw: str = Field(description=_raw_description)
    lemma: str = Field(description="The lemma of this token.")
    pos: UniversalDepPOSTags = Field(description=_pos_description)
    offset: list[int] = Field(description=_offset_description)
    is_content_word: bool = Field(description=_is_content_word_description)

    @field_validator("offset", mode="after")
    @classmethod
    def offset_correct_length(cls, offset: list[int]) -> list[int]:
        """
        Ensures that the character offsets are always a list of two integers.
        """
        offset_length = len(offset)
        if offset_length != 2:
            raise ValueError(
                f"Expect the offset to be of length 2 and not {offset_length}"
            )
        return offset

    @field_validator("pos", mode="before")
    @classmethod
    def lower_case_pos_tags(cls, pos_tag: str) -> str:
        """
        Ensures that the POS tags when given are lower cased.

        This is required for matching with the enum values which are all lower
        case.
        """
        if isinstance(pos_tag, str):
            pos_tag = pos_tag.lower()
        return pos_tag


class Annotation(BaseModel):
    """
    Validates each annotation in the WSL dataset.

    NOTE: The token offsets for 5 tokens in the original WSL dataset contained the value
    None, however through the character offsets we have managed to recover the
    token offsets. These token offsets were found through the following function
    `get_token_off_for_annotation`.
    """

    _pos_description = (
        "The POS tag of the token, expected to be a Universal Dependency POS "
        "tag. We have various mappers between different POS tagsets in "
        "`experimental_wsd.pos_constants`. NOTE that for MWEs whereby the POS "
        "tags for each token in the MWE is different the POS tag of the "
        "annotation is likely to be None."
    )
    _offset_description = (
        "Character offsets of the token text in the original text, expected to "
        "be a list of two integers the start and end character offset indexes."
    )
    _token_off_description = (
        "Token offsets, this is a list of integers, in most cases it is just "
        "one integer but can be more than one for MWEs."
    )
    _source_description = (
        "A very specific field for the WSL dataset, represents whether the "
        "annotation is `NEW` or `OLD`. This is None for all datasets other than WSL."
    )
    _label_description = "A list of WordNet sense keys"
    _mwe_description = "Whether this annotation is a Multi Word Expression"
    _a_sub_token_description = (
        "True if the annotation text is part of a "
        "whole token, e.g. annotation text is `state` "
        "the linked token is `state/district`"
    )
    _number_labels_description = "The number of labels"
    _id_description = (
        "The unique ID for this annotation, this is the "
        "sentence id combined with the index of the annotation "
        "{sentence_id}:{annotation_index}"
    )
    _overlapping_annotations_description = (
        "A list of annotation ids that lexically overlap, "
        "e.g. Multi Word Expression and annotation of "
        "single tokens within the MWE."
    )

    raw: str = Field(description="Text of the token")
    lemma: str = Field(description="Lemma of the token")
    pos: UniversalDepPOSTags | None = Field(description=_pos_description)
    offset: list[int] = Field(description=_offset_description)
    token_off: list[int] = Field(description=_token_off_description)
    source: Literal["OLD", "NEW"] | None = Field(description=_source_description)
    labels: list[str] = Field(
        description=_label_description,
        examples=[["carrousel%1:06:01::", "not%4:02:00::"]],
    )
    is_mwe: bool = Field(description=_mwe_description)
    a_sub_token: bool = Field(description=_a_sub_token_description)
    number_labels: int = Field(description=_number_labels_description)
    id: str = Field(description=_id_description, examples=["semeval2015.d003.s023:4"])
    overlapping_annotations: list[str] = Field(
        description=_overlapping_annotations_description,
        examples=[["semeval2015.d003.s023:3", "semeval2015.d003.s023:4"]],
    )

    @field_validator("pos", mode="before")
    @classmethod
    def lower_case_pos_tags(cls, pos_tag: str | None) -> str | None:
        """
        Ensures that the POS tags when given are lower cased.

        This is required for matching with the enum values which are all lower
        case.
        """
        if isinstance(pos_tag, str):
            pos_tag = pos_tag.lower()
        return pos_tag

    @field_validator("source", mode="before")
    @classmethod
    def upper_case_source(cls, source: str | None) -> str | None:
        if isinstance(source, str):
            source = source.upper()
        return source

    @field_validator("offset", mode="after")
    @classmethod
    def offset_correct_length(cls, offset: list[int]) -> list[int]:
        offset_length = len(offset)
        if offset_length != 2:
            raise ValueError(
                f"Expect the offset to be of length 2 and not {offset_length}"
            )
        return offset

    @field_validator("labels", mode="after")
    @classmethod
    def at_least_one_value(cls, list_of_values: list[Any]) -> list[Any]:
        if len(list_of_values) < 1:
            raise ValueError(
                "Expected the list of values to contain at least one value."
            )
        return list_of_values

    @field_validator("token_off", mode="before")
    @classmethod
    def contains_value_or_not(cls, list_of_values: list[Any]) -> list[Any] | None:
        """
        Args:
            list_of_values (list[Any]): A list that can contain anything.
        Returns:
            list[Any] | None: The same list if it contains at least one value,
                else if the value contained nothing then it returns None.
        """

        if isinstance(list_of_values, list):
            if len(list_of_values) > 0:
                return list_of_values
        return list_of_values


class WSLSentence(BaseModel):
    """
    Validates each sentence in the WSL dataset.

    NOTE within the original WSL dataset that some sentences do not contain
    any annotations, this is expected I believe.
    """

    sentence_id: WSLID
    text: str
    tokens: list[Token]
    annotations: list[Annotation]

    @field_validator("tokens", mode="after")
    @classmethod
    def at_least_one_value(cls, list_of_values: list[Any]) -> list[Any]:
        if len(list_of_values) < 1:
            raise ValueError(
                "Expected the list of values to contain at least one value."
            )
        return list_of_values

    def validate_wordnet_sense_keys(
        self, word_net_sense_getter: Callable[[str], wn.Sense | None]
    ) -> None:
        """
        Validates that all labels from all annotations are valid WordNet sense
        keys, if not it raises a ValueError.

        Args:
            word_net_sense_getter (Callable[str, [wn.Sense | None]]): A callable
            that takes as input a sense key, e.g. `carrousel%1:06:01::`
            and returns a WordNet Sense if it can be found else None. NOTE that
            sense keys are specific to English WordNets I believe and they are not
            the same as Sense IDs like `omw-en-carrousel-02966372-n`. The best way
            to get this callable is through the function:
            `wn.compat.sensekey.sense_getter` see for more details:
            `https://wn.readthedocs.io/en/latest/api/wn.compat.sensekey.html`

        Raises:
            ValueError: If any of the labels from any of the annotations are not
                valid WordNet sense keys according to the `word_net_sense_getter`.
        """
        for annotation in self.annotations:
            for word_net_sense_key in annotation.labels:
                if not isinstance(word_net_sense_getter(word_net_sense_key), wn.Sense):
                    raise ValueError(
                        f"The label: {word_net_sense_key} in "
                        f"annotation {annotation} for sentence ID: "
                        f"{self.sentence_id} is not a valid "
                        "Word Net sense key."
                    )

    @model_validator(mode="after")
    def check_token_annotation_offsets(self):
        """
        Ensures that the text when using the character offsets from
        `self.tokens` and `self.annotations` match the text of the token and
        annotation. If they do not match it raises a ValueError.

        NOTE: When matching we match after both text values have been lower
        cased.

        Raises:
            ValueError: If the character offset text does not match the Token
            or Annotation text.
        """

        name_data = [("token", self.tokens), ("annotation", self.annotations)]
        for data_name, data in name_data:
            for data_instance in data:
                start_offset, end_offset = data_instance.offset
                text_from_offset = self.text[start_offset:end_offset]
                data_text = data_instance.raw
                if text_from_offset.lower() != data_text.lower():
                    raise ValueError(
                        f"The raw {data_name} text: {data_text} does not "
                        f"match the text taken from the {data_name} "
                        f"offsets: {text_from_offset} "
                        f"This is found for the sentence ID: {self.sentence_id}"
                    )
        return self

    def get_content_tokens(self) -> list[Token]:
        """
        Returns:
            list[Token]: A subset of the tokens that are content words/tokens.
        """
        content_tokens = []
        for token in self.tokens:
            if token.is_content_word:
                content_tokens.append(token)
        return content_tokens

    def get_non_content_tokens(self) -> list[Token]:
        """
        Returns:
            list[Token]: A subset of the tokens that are non-content words/tokens.
        """
        non_content_tokens = []
        for token in self.tokens:
            if not token.is_content_word:
                non_content_tokens.append(token)
        return non_content_tokens

    def get_mwes(self) -> list[Annotation]:
        """
        Returns:
            list[Token]: A subset of the annotations that are
                Multi Word Expressions.
        """
        mwe_annotations = []
        for annotation in self.annotations:
            if annotation.is_mwe:
                mwe_annotations.append(annotation)
        return mwe_annotations

    def get_sub_tokens(self) -> list[Annotation]:
        """
        Sub-tokens here means a token that is not a whole token but rather it is
        part of a whole token, e.g. given the token `state/district` then
        `state` would be a sub-token.
        Returns:
            list[Token]: A subset of the annotations that are
                sub-tokens.
        """
        sub_token_annotations = []
        for annotation in self.annotations:
            if annotation.a_sub_token:
                sub_token_annotations.append(annotation)
        return sub_token_annotations


def get_token_off_for_annotation(
    annotation_char_offset: list[int], tokens: list[Token]
) -> list[int]:
    """
    Returns the token offsets from the character offsets.

    Args:
        annotation_char_offset (list[int]): The start and end character offsets.
        tokens (list[Token]): The list of tokens, whereby each token contains it's
            own character start and end offsets for the token.
    Returns:
        list[int]: A list of token one or more token offsets that represent the
            tokens that appear within the `annotation_char_offset` character
            offsets.
    Raises:
        ValueError: If the `annotation_char_offset` are not a list of 2 integers.
        ValueError: If no token offsets can be found.
    """
    if len(annotation_char_offset) != 2:
        raise ValueError(
            f"The character offsets should be of length 2: {annotation_char_offset}"
        )
    char_start_offset, char_end_offset = annotation_char_offset
    token_offsets = []
    in_token_span = False
    for index, token in enumerate(tokens):
        token_char_start, token_char_end = token.offset
        if in_token_span:
            token_offsets.append(index)
        if (
            char_start_offset >= token_char_start
            and char_start_offset <= token_char_end
        ):
            token_offsets.append(index)

            if token_char_end < char_end_offset:
                in_token_span = True
        if char_end_offset == token_char_end:
            in_token_span = False
    if in_token_span or not token_offsets:
        raise ValueError(
            "Could not find the token offsets for these character "
            f"offsets: {annotation_char_offset}. Relevant tokens: \n"
            f"{tokens}"
        )
    return token_offsets


def is_a_sub_token(annotation_char_offset: list[int], tokens: list[Token]) -> bool:
    """
    Returns True if the annotation character offsets are part of a token but
    crucially do not make up an entire token.

    Args:
        annotation_char_offset (list[int]): The start and end character offsets.
        tokens (list[Token]): The list of tokens, whereby each token contains it's
            own character start and end offsets for the token.
    Returns:
        bool: True if the annotation character offsets are part of a token but
        crucially do not make up an entire token else returns False.
    Raises:
        ValueError: If the `annotation_char_offset` are not a list of 2 integers.
    """
    if len(annotation_char_offset) != 2:
        raise ValueError(
            f"The character offsets should be of length 2: {annotation_char_offset}"
        )
    char_start_offset, char_end_offset = annotation_char_offset
    for token in tokens:
        token_char_start, token_char_end = token.offset
        if char_start_offset >= token_char_start and char_end_offset < token_char_end:
            return True
        if char_start_offset > token_char_start and char_end_offset <= token_char_end:
            return True
    return False


def get_overlapping_annotations(
    annotation: Annotation, sentence_annotations: list[Annotation]
) -> list[str]:
    """
    Given an annotation, C, and it's sentence of annotations that it is part of, it
    will return the list of annotation IDs for each annotation that has a token
    offset ID that is within C's token offset IDs. Therefore returning a list of
    annotation IDs that lexical overlap.

    Note: we assume all annotations `overlapping_annotations` attribute are empty
    lists, as this function should be used to set that attribute.

    Note: in most cases this will return an empty list, as most annotations do not
    overlap lexically with each other.

    Will return a list of annotation IDs for the given single annotation whereby
    each returned annotation ID overlaps
    Args:
        annotation (Annotation): An annotation from the sentence of annotations.
        sentence_annotations (list[Annotation]): A list of annotations from the same
            sentence as the single annotation that is given as an argument.
    Raises:
        ValueError: If the given `annotation` has the same token offsets as
            another annotation in the same sentence and neither annotation is a
            sub token.
        ValueError: If duplicate annotation ids are in the list of overlapping
            annotations that would have been returned.
    """
    annotation_token_offsets = annotation.token_off
    annotation_token_offsets_set = set(annotation_token_offsets)
    annotation_id = annotation.id
    annotations_overlapping: list[str] = []
    for other_annotation in sentence_annotations:
        other_annotation_id = other_annotation.id
        if other_annotation_id == annotation_id:
            continue
        other_annotation_token_offsets = other_annotation.token_off
        if annotation_token_offsets == other_annotation_token_offsets:
            if annotation.a_sub_token:
                pass
            elif other_annotation.a_sub_token:
                pass
            else:
                raise ValueError(
                    "Two annotation samples have the same "
                    "token offsets, this should not be the case. "
                    "unless one of the annotations is a sub token "
                    "which is not the case here. \n"
                    f"Annotation 1: {annotation} and \n"
                    f"Annotation 2: {other_annotation}"
                )

        for other_token_offset in other_annotation_token_offsets:
            if other_token_offset in annotation_token_offsets_set:
                annotations_overlapping.append(other_annotation_id)
                break
    if len(annotations_overlapping) != len(set(annotations_overlapping)):
        raise ValueError(
            "The number of annotations this annotation is overlapping with "
            f"contains duplicate annotations: {annotations_overlapping} "
            f"This is for the following annotation: {annotation}"
        )
    return annotations_overlapping


def is_content_word(token_index: int, sentence_annotations: list[Annotation]) -> bool:
    """
    Given a token's index it will verify if that token has an associated
    annotation linked to it, e.g. does an annotation in the same sentence as
    the token contain a token offset the same as this token's index.

    A content word in WSD is typically a noun, verb, adjective, or adverb. However,
    this can be different from dataset to dataset hence why we rely on the
    annotations from the dataset.

    Args:
        token_index (int): The index within the sentence of the token.
        sentence_annotations (list[Annotation]): List of annotation from the
            same sentence as the token.
    Returns:
        bool: True if the token at this token index is a content word else False.
    """

    for annotation in sentence_annotations:
        annotation_token_offsets = annotation.token_off
        if token_index in annotation_token_offsets:
            return True
    return False


def wsl_annotations_to_amend(
    annotations: list[Annotation], sentence_id: str
) -> list[Annotation]:
    """
    This function amends annotations that exist in the WSL dataset. It
    will remove any duplicate annotations and if required update labels for
    annotations whereby the label of the duplicate is different to the
    non-duplicate.

    Args:
        annotations (list[Annotation]): The annotations to de-duplicate and/or amend.
        sentence_id (str): The sentence ID of the annotations.
    Returns:
        list[Annotation]: A list of de-duplicated and/or amended annotations.
            In most cases this would be the same annotations list as we have
            very few duplicates/amendments.
    """
    affected_sentence_ids = set(
        [
            "senseval3.d001.s041",
            "senseval3.d001.s045",
            "senseval3.d002.s034",
            "senseval3.d002.s109",
            "semeval2013.d008.s001",
            "semeval2013.d008.s026",
        ]
    )
    if sentence_id not in affected_sentence_ids:
        return annotations
    non_duplicate_annotations: list[Annotation] = []

    removal_sentence_annotation_ids = {
        "senseval3.d001.s041": set(["senseval3.d001.s041:42"]),
        "senseval3.d001.s045": set(["senseval3.d001.s045:27"]),
        "senseval3.d002.s034": set(["senseval3.d002.s034:7"]),
        "senseval3.d002.s109": set(["senseval3.d002.s109:14"]),
        "semeval2013.d008.s001": set(["semeval2013.d008.s001:10"]),
        "semeval2013.d008.s026": set(["semeval2013.d008.s026:16"]),
    }

    label_sentence_annotation_ids = {
        "senseval3.d002.s109": {
            "senseval3.d002.s109:13": [
                "crazy%5:00:00:excited:00",
                "crazy%5:00:00:insane:00",
            ]
        },
        "semeval2013.d008.s001": {
            "semeval2013.d008.s001:10": ["enterprise%1:14:00::", "enterprise%1:04:00::"]
        },
    }

    if sentence_id in removal_sentence_annotation_ids:
        removal_annotation_ids = removal_sentence_annotation_ids[sentence_id]
        for annotation in annotations:
            if annotation.id in removal_annotation_ids:
                continue
            non_duplicate_annotations.append(annotation)

    if not non_duplicate_annotations:
        non_duplicate_annotations = copy.deepcopy(annotations)

    if sentence_id in label_sentence_annotation_ids:
        label_annotation_ids = label_sentence_annotation_ids[sentence_id]
        for annotation in non_duplicate_annotations:
            if annotation.id in label_annotation_ids:
                amended_labels = label_annotation_ids[annotation.id]
                annotation.labels = amended_labels

    return non_duplicate_annotations


def wsl_sentence_generator(
    dataset: DatasetDict,
    split: str,
    word_net_sense_getter: Callable[[str], wn.Sense | None],
    filter_by_dataset_id: str | None = None,
) -> Iterator[WSLSentence]:
    """
    Given a dataset dictionary, of which we expect the dataset to be
    [a Babelscape WSL dataset](https://huggingface.co/datasets/Babelscape/wsl),
    this will yield sentence level data for each sentence in the dataset for
    the given split. The sentence level data is validated and formatted into a
    WSLSentence.

    The `word_net_sense_getter` argument is used by
    `WSLSentence.validate_wordnet_sense_keys` to ensure that all labels for
    all annotations are valid WordNet sense keys.


    We in the future might want to add an additional argument on whether to
    amend the annotations through the function `wsl_annotations_to_amend` or not
    at the moment by default we do.

    Args:
        dataset (DatasetDict): A dictionary whereby the keys are split names and
            the value is a Babelscape WSL dataset.
        split (str): The split of the dataset that the sentence level data should
            be retrieved from.
        word_net_sense_getter (Callable[str, [wn.Sense | None]]): A callable
            that takes as input a sense key, e.g. `carrousel%1:06:01::`
            and returns a WordNet Sense if it can be found else None. NOTE that
            sense keys are specific to English WordNets I believe and they are not
            the same as Sense IDs like `omw-en-carrousel-02966372-n`. The best way
            to get this callable is through the function:
            `wn.compat.sensekey.sense_getter` see for more details:
            `https://wn.readthedocs.io/en/latest/api/wn.compat.sensekey.html`
        filter_by_dataset_id (Union[str, None]): Will filter the sentences so that
            only sentence from the given dataset id will be yielded. If None no
            filtering is applied. Default is None.
    Returns:
        Iterator[WSLSentence]: An iterator of formatted and validated sentence
            level data from the given split of the given WSL dataset.

    Raises:
        ValueError: If the sentence ID cannot be validated.
        ValueError: Any Pydantic validation error generated when the Pydantic
            object cannot be created. This will come from either `WSLID`,
            `Token`, `Annotation`, or `WSLSentence`.
    """
    split_dataset = dataset[split]

    for sample in split_dataset:
        text = sample["text"]
        sentence_id = sample["sentence_id"]
        tokens = sample["tokens"]
        annotations = sample["annotations"]

        sentence_id_match = WSLID.WSLID_RE.match(sentence_id)
        if sentence_id_match is None:
            raise ValueError(
                f"Cannot validate the sentence ID: {sentence_id} "
                f"for the following sample: {sample}"
            )
        if len(sentence_id_match.groups()) != 3:
            raise ValueError(
                f"Cannot validate the sentence ID: {sentence_id} "
                f"for the following sample: {sample}"
            )

        dataset_id, document_index_str, sentence_index_str = sentence_id_match.groups()

        if filter_by_dataset_id:
            if dataset_id != filter_by_dataset_id:
                continue

        document_index = int(document_index_str)
        sentence_index = int(sentence_index_str)
        validated_sentence_id = WSLID(
            dataset_id=dataset_id,
            document_index=document_index,
            sentence_index=sentence_index,
        )

        validated_tokens: list[Token] = []
        validated_annotations: list[Annotation] = []

        for token in tokens:
            # Placeholder until we have the sentence annotations.
            token["is_content_word"] = False
            validated_tokens.append(Token.model_validate(token))

        for index, annotation in enumerate(annotations):
            if not annotation["token_off"]:
                token_offset = get_token_off_for_annotation(
                    annotation["offset"], validated_tokens
                )
                annotation["token_off"] = token_offset

            token_pos = annotation["pos"]
            token_offsets = annotation["token_off"]
            is_mwe = True if len(token_offsets) > 1 else False
            annotation["is_mwe"] = is_mwe
            annotation["number_labels"] = len(annotation["labels"])
            if token_pos is None:
                pos_tag = None
                if not is_mwe:
                    pos_tag = tokens[token_offsets[0]]["pos"]
                annotation["pos"] = pos_tag
            annotation["a_sub_token"] = False
            if not is_mwe:
                annotation["a_sub_token"] = is_a_sub_token(
                    annotation["offset"], validated_tokens
                )
            annotation["id"] = f"{sentence_id}:{index}"
            annotation["overlapping_annotations"] = []
            validated_annotations.append(Annotation.model_validate(annotation))

        validated_annotations = wsl_annotations_to_amend(
            validated_annotations, sentence_id
        )

        temp_validated_annotations = []
        for annotation in validated_annotations:
            overlapping_annotations = get_overlapping_annotations(
                annotation, validated_annotations
            )
            annotation.overlapping_annotations = overlapping_annotations
            temp_validated_annotations.append(annotation)

        temp_validated_tokens = []
        for token_index, token in enumerate(validated_tokens):
            token.is_content_word = is_content_word(
                token_index, temp_validated_annotations
            )
            temp_validated_tokens.append(token)

        wsl_sentence = WSLSentence(
            text=text,
            sentence_id=validated_sentence_id,
            tokens=temp_validated_tokens,
            annotations=temp_validated_annotations,
        )
        wsl_sentence.validate_wordnet_sense_keys(word_net_sense_getter)
        yield wsl_sentence


def get_all_dataset_ids(wsl_dataset: Iterator[WSLSentence]) -> set[str]:
    """
    Given the arguments it returns a unique set of dataset ids from the data.

    Args:
        wsl_sentences (Iterator[WSLSentence]): The data to to get the dataset
            ids from.
    Returns:
        set[str]: A set of dataset ids.
    """
    all_dataset_ids = set()
    for sentence in wsl_dataset:
        all_dataset_ids.add(sentence.sentence_id.dataset_id)
    return all_dataset_ids


def wsl_data_statistics(
    wsl_sentences: Iterator[WSLSentence],
) -> dict[str, Union[int, str]]:
    """
    Generates a dictionary containing the following datasets statistics for the
    WSLSentence data given:

    `No. Docs`: int
    `No. Sent`: int
    `No. Tokens`: int,
    `No. Content Tokens (%)`: str
    `No. Annotations`: int
    `No. MWEs (%)`: str
    `No. Sub tokens`: int

    For the values that are strings, they are strings as they are a combination of
    the integer value and their normalized percentage value.

    Args:
        wsl_sentences (Iterator[WSLSentence]): The data to generate statistics
            from.
    Returns:
        dict[str, Union[int, str]]: The formatted data statistics.
    """
    document_ids = set()
    number_sentences = 0
    number_tokens = 0
    number_content_tokens = 0
    number_annotations = 0
    number_mwes = 0
    number_sub_tokens = 0

    for wsl_sentence in wsl_sentences:
        document_ids.add(
            f"{wsl_sentence.sentence_id.dataset_id} {wsl_sentence.sentence_id.document_index}"
        )
        number_sentences += 1
        number_tokens += len(wsl_sentence.tokens)
        number_content_tokens += len(wsl_sentence.get_content_tokens())
        number_annotations += len(wsl_sentence.annotations)
        number_mwes += len(wsl_sentence.get_mwes())
        number_sub_tokens += len(wsl_sentence.get_sub_tokens())

    number_documents = len(document_ids)

    content_tokens_percentage = (float(number_content_tokens) / number_tokens) * 100
    formatted_content_tokens = (
        f"{number_content_tokens:,} ({content_tokens_percentage:.2f}%)"
    )

    mwe_percentage = (float(number_mwes) / number_annotations) * 100
    formatted_mwes = f"{number_mwes} ({mwe_percentage:.2f}%)"

    return {
        "No. Docs": number_documents,
        "No. Sent": number_sentences,
        "No. Tokens": number_tokens,
        "No. Content Tokens (%)": formatted_content_tokens,
        "No. Annotations": number_annotations,
        "No. MWEs (%)": formatted_mwes,
        "No. Sub tokens": number_sub_tokens,
    }


def get_overlapping_occurrences_statistics(
    wsl_sentences: Iterator[WSLSentence],
) -> dict[str, int]:
    """
    Generates a dictionary containing the following datasets statistics on
    overlapping annotation occurrences for the WSLSentence data given:

    `Number of entities in overlapping groups`: int
    `Number of overlapping groups`: int
    `One common label groups`: int
    `Overlapping groups of N`: int

    All of these statistics relate to overlapping occurrences, overlapping
    occurrence is whereby a group of tokens in a sentence have more than one
    annotation related to them, this typically happens through MWE, e.g.
    "lung cancer", where "lung", "cancer", and "lung cancer" all have annotations
    associated to them.

    * `Number of entities in overlapping groups` - Number of overlapping groups,
      whereby a group can contain two or more entities (annotations).
    * `Number of overlapping groups` - Number of entities/annotations
      that are in these overlapping groups.
    * `One common label groups` - Number of groups whereby all entities in that
      group have one common WSD label.
    * `Overlapping groups of N` - A breakdown of the number of overlapping groups
      based on number of entities in the groups (`N`). Here `N` means that
      there can be multiple fields whereby `N` is greater than 1.

    Args:
        wsl_sentences (Iterator[WSLSentence]): The data to generate statistics
            from.
    Returns:
        dict[str, Union[int, str]]: The formatted data statistics.
    """
    number_overlapping_groups = 0
    number_of_annotations_in_overlapping_groups = 0
    number_groups_with_at_least_one_common_label = 0
    overlapping_group_sizes = defaultdict(lambda: 0)
    for sentence in wsl_sentences:
        overlapping_groups: list[set[str]] = []

        annotation_id_to_annotation = {}
        for annotation in sentence.annotations:
            annotation_id_to_annotation[str(annotation.id)] = annotation
            if not annotation.overlapping_annotations:
                continue

            current_overlapping_annotation_ids = copy.deepcopy(
                annotation.overlapping_annotations
            )
            current_overlapping_annotation_ids_set = set(
                current_overlapping_annotation_ids
            )
            current_overlapping_annotation_ids_set.add(annotation.id)
            for overlapping_group in overlapping_groups:
                if current_overlapping_annotation_ids_set.intersection(
                    overlapping_group
                ):
                    overlapping_group.update(current_overlapping_annotation_ids_set)
                    break
            else:
                overlapping_groups.append(current_overlapping_annotation_ids_set)
        if not overlapping_groups:
            continue
        number_overlapping_groups += len(overlapping_groups)
        number_of_annotations_in_overlapping_groups += sum(
            len(group) for group in overlapping_groups
        )

        for overlapping_group in overlapping_groups:
            label_sets: list[set[str]] = []
            for annotation_id in overlapping_group:
                annotation = annotation_id_to_annotation[str(annotation_id)]
                assert isinstance(annotation, Annotation)
                label_sets.append(set(annotation.labels))

            if set.intersection(*label_sets):
                number_groups_with_at_least_one_common_label += 1

            overlapping_group_sizes[len(overlapping_group)] += 1

    overlapping_group_sizes_str = {
        f"Overlapping groups of {group_size}": count
        for group_size, count in overlapping_group_sizes.items()
    }
    wsd_overlapping_group_statistics = {
        "Number of entities in overlapping groups": number_of_annotations_in_overlapping_groups,
        "Number of overlapping groups": number_overlapping_groups,
        "One common label groups": number_groups_with_at_least_one_common_label,
        **overlapping_group_sizes_str,
    }
    return wsd_overlapping_group_statistics


def get_content_words_pos_tag_frequencies(
    wsl_sentences: Iterator[WSLSentence],
) -> dict[str, int]:
    """
    Given the data generates a dictionary of POS tags and their frequency. The
    POS tags have come from the tokens of content tokens/words.

    Args:
        wsl_sentences (Iterator[WSLSentence]): The data to generate the
            frequencies from.
    Returns:
        dict[str, int]: A dictionary of POS tags and their frequency for all
            content tokens/words in the given dataset.
    """
    pos_tag_counter = Counter()
    for sentence in wsl_sentences:
        sentence_pos_tags = [
            content_token.pos for content_token in sentence.get_content_tokens()
        ]
        pos_tag_counter.update(sentence_pos_tags)
    return dict(pos_tag_counter)


def get_non_content_words_pos_tag_frequencies(
    wsl_sentences: Iterator[WSLSentence],
) -> dict[str, int]:
    """
    Given the data generates a dictionary of POS tags and their frequency. The
    POS tags have come from the tokens of non-content tokens/words.

    Args:
        wsl_sentences (Iterator[WSLSentence]): The data to generate the
            frequencies from.
    Returns:
        dict[str, int]: A dictionary of POS tags and their frequency for all
            non-content tokens/words in the given dataset.
    """
    pos_tag_counter = Counter()
    for sentence in wsl_sentences:
        sentence_pos_tags = [
            content_token.pos for content_token in sentence.get_non_content_tokens()
        ]
        pos_tag_counter.update(sentence_pos_tags)
    return dict(pos_tag_counter)


def add_zero_frequency_pos_tags(
    pos_tag_frequencies: dict[str, int], relevant_pos_tags: list[UniversalDepPOSTags]
) -> dict[str, int]:
    """
    Given a dictionary of POS tags and their frequencies for any POS tag that is
    within `relevant_pos_tags` and is not in the dictionary the POS tag will be
    added to the dictionary with a frequency value of 0.

    Args:
        pos_tags_frequencies (dict[str, int]):  A dictionary of POS tags and
            their frequency.
        relevant_pos_tags (list[UniversalDepPOSTags]): A list of POS tags whereby
            any tag that is not in the dictionary should be added with a frequency
            of 0.
    Returns:
        dict[str, int]: An expanded POS tag to frequency dictionary whereby all
            new POS tag entires have a frequency of 0.
    """
    for universal_pos_tag in relevant_pos_tags:
        if universal_pos_tag not in pos_tag_frequencies:
            pos_tag_frequencies[universal_pos_tag] = 0
    return pos_tag_frequencies


def pos_tag_frequencies_in_given_order(
    pos_tag_frequencies: dict[str, int], pos_tag_order: list[str]
) -> list[int]:
    """
    Given the `pos_tag_order` order it will return the frequency of those POS
    tags in that given order from `pos_tag_frequencies`.

    Args:
        pos_tag_frequencies (dict[str, int]): Dictionary of POS tags and their
            frequency, of which these frequencies will be returned in the
            `pos_tag_order` order.
        pos_tag_order (list[str]): List of POS tags that determines the order
            of the returned frequencies.
    Returns:
        list[int]: An ordered list of POS tag frequencies from `pos_tag_frequencies`
            in the order determined by `pos_tag_order`.
    """
    frequencies = []
    for pos_tag in pos_tag_order:
        frequencies.append(pos_tag_frequencies[pos_tag])
    return frequencies


def create_wsd_pos_content_words_plot(
    wsl_sentences: Iterator[WSLSentence],
    title: str = "POS tag frequencies for Content and Non-Content tokens",
    height: int = 500,
    width: int = 700,
    y_range: tuple[int, int] = (0, 6500),
) -> figure:
    """
    Given the arguments it generates a bar plot of the frequency of content
    and non-content words given the UD POS tag of those words. The UD POS
    tag label is shown on the X-Axis.

    Args:
            wsl_sentences (Iterator[WSLSentence]): The data to generate statistics
                    from.
            content_word_frequencies_y_axis (list[int]): The frequency of the
                    content words for each POS tag within the `ud_pos_tags_x_axis`.
            non_content_word_requencies_y_axis (list[int]): The frequency of the
                    non-content words for each POS tag within the `ud_pos_tags_x_axis`.
            title (str): The title of the plot.
            height (int): The height of the plot.
            width (int): The width of the plot.
            y_range (tuple[int, int]): The range of the y-axis values.
    Returns:
            bokeh.plotting.figure: A plot whereby the UD POS tags are on the
                    x-axis and per POS tags two bars in different colors, one
                    representing the frequency of the content words and the other
                    non-content words for the given POS tag.
    """
    content_word_wsl_iter, non_content_word_wsl_iter = tee(wsl_sentences)
    content_word_pos_tag_frequencies = get_content_words_pos_tag_frequencies(
        content_word_wsl_iter
    )
    non_content_word_pos_tag_frequencies = get_non_content_words_pos_tag_frequencies(
        iter(non_content_word_wsl_iter)
    )

    UD_pos_tags = list(UniversalDepPOSTags)
    relevant_ud_pos_tags = []
    for ud_pos_tag in UD_pos_tags:
        if (
            ud_pos_tag in content_word_pos_tag_frequencies
            or ud_pos_tag in non_content_word_pos_tag_frequencies
        ):
            relevant_ud_pos_tags.append(ud_pos_tag)
    content_word_pos_tag_frequencies = add_zero_frequency_pos_tags(
        content_word_pos_tag_frequencies, relevant_ud_pos_tags
    )
    non_content_word_pos_tag_frequencies = add_zero_frequency_pos_tags(
        non_content_word_pos_tag_frequencies, relevant_ud_pos_tags
    )

    ordered_UD_pos_tags = [
        ud_pos_tag
        for ud_pos_tag, _ in sorted(
            content_word_pos_tag_frequencies.items(), key=lambda x: x[1], reverse=True
        )
    ]
    UD_pos_tags_str = [ud_pos_tag.value for ud_pos_tag in ordered_UD_pos_tags]

    content_word_pos_frequencies = pos_tag_frequencies_in_given_order(
        content_word_pos_tag_frequencies, ordered_UD_pos_tags
    )
    non_content_word_pos_frequencies = pos_tag_frequencies_in_given_order(
        non_content_word_pos_tag_frequencies, ordered_UD_pos_tags
    )

    x_axis_data_label = "UD POS tags"
    y_axis_content_label = "Content"
    y_axis_non_content_label = "Non-Content"
    data = {
        x_axis_data_label: UD_pos_tags_str,
        y_axis_content_label: content_word_pos_frequencies,
        y_axis_non_content_label: non_content_word_pos_frequencies,
    }
    source = ColumnDataSource(data=data)

    bar_figure = figure(
        x_range=UD_pos_tags_str,
        title=title,
        height=height,
        width=width,
        toolbar_location=None,
        tools="hover",
        tooltips="@$name",
        y_range=y_range,
    )
    bar_figure.vbar(
        x=dodge(x_axis_data_label, 0.0, range=bar_figure.x_range),
        top=y_axis_content_label,
        source=source,
        width=0.2,
        color="#e84d60",
        legend_label=y_axis_content_label,
        name=y_axis_content_label,
    )

    bar_figure.vbar(
        x=dodge(x_axis_data_label, 0.25, range=bar_figure.x_range),
        top=y_axis_non_content_label,
        source=source,
        width=0.2,
        color="#718dbf",
        legend_label=y_axis_non_content_label,
        name=y_axis_non_content_label,
    )

    content_labels = LabelSet(
        x=x_axis_data_label,
        y=y_axis_content_label,
        text=y_axis_content_label,
        x_offset=5,
        y_offset=5,
        source=source,
        angle=90,
        angle_units="deg",
    )
    non_content_labels = LabelSet(
        x=x_axis_data_label,
        y=y_axis_non_content_label,
        text=y_axis_non_content_label,
        x_offset=20,
        y_offset=5,
        source=source,
        angle=90,
        angle_units="deg",
    )

    bar_figure.x_range.range_padding = 0.1
    bar_figure.xgrid.grid_line_color = None
    bar_figure.legend.location = "top_right"
    bar_figure.legend.orientation = "horizontal"
    bar_figure.add_layout(content_labels)
    bar_figure.add_layout(non_content_labels)
    return bar_figure
