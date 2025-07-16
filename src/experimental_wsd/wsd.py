import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Iterator

import wn

from experimental_wsd.config import WSDDataDirectory
from experimental_wsd.pos_constants import SEMCOR_TO_UNI_DEP_POS_TAGS
from experimental_wsd.wsl import WSLID, Annotation, Token, WSLSentence


def wsd_sentence_generator(
    data_directory: WSDDataDirectory,
    word_net_sense_getter: Callable[[str], wn.Sense | None],
    filter_by_dataset_id: str | None = None,
) -> Iterator[WSLSentence]:
    """
    Given a WSD data directory, this will yield sentence level data for each
    sentence in the dataset. The sentence level data is validated and
    formatted into a WSLSentence.

    The `word_net_sense_getter` argument is used by
    `WSLSentence.validate_wordnet_sense_keys` to ensure that all labels for
    all annotations are valid WordNet sense keys.

    NOTE: At the moment we are not identifying MWE or sub word tokens. In
    addition the overlapping annotations for each annotation will always be
    an empty list.

    Args:
        dataset (WSDDataDirectory): A folder that contains both the raw XML
            data and gold annotations in text format.
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
            level data from the given WSD dataset.

    Raises:
        ValueError: If the sentence ID cannot be validated.
        ValueError: If an unknown XML tag is found.
        ValueError: Any Pydantic validation error generated when the Pydantic
            object cannot be created. This will come from either `WSLID`,
            `Token`, `Annotation`, or `WSLSentence`.
    """

    dataset_sentence_id_re = WSLID.WSLID_RE
    non_dataset_sentence_id_re = re.compile(r"^d(\d+).s(\d+)$")

    semcor_pos_tag_dataset_names = set(
        [
            "semcor",
            "mun",
            "senseval2",
            "senseval3",
            "semeval2007",
            "semeval2010",
            "semeval2013",
            "semeval2015",
            "hardEN",
            "softEN",
            "42D",
        ]
    )
    annotation_id_to_labels = load_annotations_from_file(data_directory.gold)

    with data_directory.data.open("r", encoding="utf-8") as data_fp:
        corpus_root = ET.parse(data_fp)

        dataset_name: str | None = None
        if "source" in corpus_root.getroot().attrib:
            dataset_name = corpus_root.getroot().attrib["source"]

        for text in corpus_root.getroot():
            for sentence in text:
                sentence_id = sentence.attrib["id"]
                document_index: None | int = None
                sentence_index: None | int = None

                dataset_sentence_id_match = dataset_sentence_id_re.match(sentence_id)
                non_dataset_sentence_id_match = non_dataset_sentence_id_re.match(
                    sentence_id
                )
                if dataset_sentence_id_match:
                    dataset_name = dataset_sentence_id_match.groups()[0]
                    document_index = int(dataset_sentence_id_match.groups()[1])
                    sentence_index = int(dataset_sentence_id_match.groups()[2])
                elif non_dataset_sentence_id_match:
                    document_index = int(non_dataset_sentence_id_match.groups()[0])
                    sentence_index = int(non_dataset_sentence_id_match.groups()[1])
                else:
                    raise ValueError(
                        "Could not parse the sentence ID to "
                        "identify the document and sentence ID: "
                        f"{sentence_id} for the dataset: "
                        f"{data_directory}"
                    )

                if filter_by_dataset_id:
                    if filter_by_dataset_id != dataset_name:
                        continue

                formatted_sentence_id = WSLID(
                    dataset_id=dataset_name,
                    document_index=document_index,
                    sentence_index=sentence_index,
                )
                text = ""
                token_offset_start = 0
                wsd_tokens: list[Token] = []
                wsd_annotations: list[Annotation] = []

                for token_index, token_instance in enumerate(sentence):
                    xml_token_tag = token_instance.tag
                    content_word = False
                    if xml_token_tag == "wf":
                        pass
                    elif xml_token_tag == "instance":
                        content_word = True
                    else:
                        raise ValueError(
                            "For the sentence ID: "
                            f"{formatted_sentence_id} in "
                            f"{data_directory}, we cannot process "
                            "the tokens in this sentence as we "
                            "have identified an unknown XML tag: "
                            f"{xml_token_tag}"
                        )
                    pos_tag = token_instance.attrib["pos"]
                    if dataset_name in semcor_pos_tag_dataset_names:
                        pos_tag = SEMCOR_TO_UNI_DEP_POS_TAGS.get(pos_tag)
                    lemma = token_instance.attrib["lemma"]
                    token_text = token_instance.text
                    token_offset_end = token_offset_start + len(token_text)
                    token_offsets = [token_offset_start, token_offset_end]
                    wsd_token = Token(
                        raw=token_text,
                        lemma=lemma,
                        pos=pos_tag,
                        offset=token_offsets,
                        is_content_word=content_word,
                    )

                    if content_word:
                        annotation_id = token_instance.attrib["id"]
                        labels = annotation_id_to_labels[annotation_id]
                        wsd_annotation = Annotation(
                            raw=token_text,
                            lemma=lemma,
                            pos=pos_tag,
                            offset=token_offsets,
                            token_off=[token_index],
                            source=None,
                            labels=labels,
                            is_mwe=False,
                            a_sub_token=False,
                            number_labels=len(labels),
                            id=annotation_id,
                            overlapping_annotations=[],
                        )
                        wsd_annotations.append(wsd_annotation)

                    text += f"{token_text} "
                    # Add 1 for the single whitespace
                    token_offset_start = token_offset_end + 1

                    wsd_tokens.append(wsd_token)
                # Remove the last single whitespace that was added by the program
                text = text.rstrip(" ")
                wsl_sentence = WSLSentence(
                    text=text,
                    sentence_id=formatted_sentence_id,
                    tokens=wsd_tokens,
                    annotations=wsd_annotations,
                )
                wsl_sentence.validate_wordnet_sense_keys(word_net_sense_getter)
                yield wsl_sentence


def load_annotations_from_file(annotation_file: Path) -> dict[str, list[str]]:
    """
    Given an annotation file in the text format of:
    `{annotation_id} {label_1} {label_2} {label_n}`
    on each new line, whereby a line can be empty and therefore ignored. It will
    return these annotation as a dictionary of `annotation_id`: list[label_0,...,label_n].

    The order of the annotations can potentially have the meaning of the first label
    is the most relevant label or best match.

    Args:
        annotation_file (Path): A file containing the annotations in the
            relevant format.
    Returns:
        dict[str, list[str]]: A dictionary of annotation id to labels.

    Raises:
        KeyError: If an annotation id exists more than once.
        ValueError: If an annotation id does not contain at least one label.
        ValueError: If an annotation id contains non-unique list of labels.
    """
    annotation_id_to_labels: dict[str, list[str]] = {}

    with annotation_file.open("r", encoding="utf-8") as annotation_fp:
        for line_index, line in enumerate(annotation_fp):
            line = line.strip()
            if not line:
                continue
            annotation_parts = line.split()
            if len(annotation_parts) < 2:
                raise ValueError(
                    "No label and/or annotation ID can be found "
                    f"for the following annotation line: {line} "
                    f"for line index: {line_index} and for "
                    f"the following annotation file: {annotation_file}"
                )
            annotation_id = annotation_parts[0]
            annotation_labels = annotation_parts[1:]

            if annotation_id in annotation_id_to_labels:
                raise KeyError(
                    "The following annotation ID already exists "
                    "(no duplicates allowed): "
                    f"{annotation_id} found on line: {line_index} "
                    "within the following annotation file: "
                    f"{annotation_file}"
                )
            if len(annotation_labels) != len(set(annotation_labels)):
                raise ValueError(
                    "The annotation labels are not unique: "
                    f"{annotation_labels} for the annotation "
                    f"ID: {annotation_id} on line index: "
                    f"{line_index} for the following annotation "
                    f"file: {annotation_file}"
                )

            annotation_id_to_labels[annotation_id] = annotation_labels
    return annotation_id_to_labels
