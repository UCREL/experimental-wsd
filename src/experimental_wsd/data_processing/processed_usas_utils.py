from typing import Any, Literal
from pathlib import Path
import logging
import math
from collections import Counter
import random

import yaml
from pydantic import BaseModel, Field, create_model
import datasets

logger = logging.getLogger(__name__)

USAS_TAGS_TO_IGNORE = set([
    "PUNC",
    "S",
    "M3.3",
    "V",
    "Z",
    "Q4.4",
    "X7.2",
    "A1.2.4",
    "A9.1",
    "G1.1.1",
    "S9.2",
    "N"
])

class USASTag(BaseModel):
    """
    Represents all of the properties associated with a USAS tag.
    """

    tag: str = Field(title="USAS Tag", description="USAS Tag", examples=["A1.1.1"])
    number_positive_markers: int = Field(
        0,
        title="Positive Markers",
        description="Number of positive markers.",
        examples=[0, 1, 2, 3],
    )
    number_negative_markers: int = Field(
        0,
        title="Negative Markers",
        description="Number of negative markers.",
        examples=[0, 1, 2, 3],
    )
    rarity_marker_1: bool = Field(
        False, title="Rare Marker 1", description="Rarity marker 1 indicated by %"
    )
    rarity_marker_2: bool = Field(
        False, title="Rare Marker 2", description="Rarity marker 2 indicated by @"
    )
    female: bool = Field(False, title="Female", description="Female")
    male: bool = Field(False, title="Male", description="Male")
    antecedents: bool = Field(
        False,
        title="Antecedents",
        description="Potential antecedents of conceptual anaphors (neutral for number)",
    )
    neuter: bool = Field(False, title="Neuter", description="Neuter")
    idiom: bool = Field(False, title="Idiom", description="Is it an idiom")


class USASTagGroup(BaseModel):
    """
    Represents a grouping of one or more USAS tags that are associated to a
    token.
    """

    _tags_description = (
        "A grouping of one or more USAS tags whereby if more "
        "than one exists then the word is an equal member of "
        "all semantic tags/categories"
    )
    _tags_examples = [
        [USASTag(tag="A1.1.1")],
        [
            USASTag(tag="E2", number_negative_markers=1),
            USASTag(tag="S7.1", number_positive_markers=1),
        ],
    ]
    tags: list[USASTag] = Field(
        title="USAS Tags", description=_tags_description, examples=_tags_examples
    )

class WikiDataExport(BaseModel):
    document_id: str = Field(
        title="Document ID",
        description="ID that uniquely represents that document within the Mosaico Mongo DB",
        examples=["3021080"],
    )
    wikidata_id: str = Field(
        title="Wikidata ID",
        description="Wikidata ID, every Wikipedia page should one as it is an unique ID that allows you to access it's global unique URL https://www.wikidata.org/entity/ID",
        examples=["Q921355"],
    )
    title: str = Field(
        title="Title",
        description="Wikipedia page title",
        examples=["Erik Adolf von Willebrand"],
    )
    text: str = Field(
        title="Text", description="The UTF-8 encoded Wikipedia article text"
    )

    ascii_text: str = Field(
        title="ASCII Text",
        description="ASCII encoded version of the Wikipedia article text",
    )
    language: Literal["en"] = Field(
        title="Language",
        description="Language of the Wikipedia article",
        examples=["en"],
    )
    quality: Literal["good", "featured"] = Field(
        title="Quality",
        description="Quality of the Wikipedia page as determined by the Wikipedia community",
        examples=["good", "featured"],
    )

    _ores_articletopics_description = (
        "High level article topics that are easily "
        "searchable and have been predicted by a machine learning model. "
        "This will be represented as a dictionary of topic "
        "and score whereby the score is between 0-1 where 1 "
        "indicates the model is more confident of it's prediction."
    )
    ores_articletopics: dict[str, float] = Field(
        title="ORES article topics",
        description=_ores_articletopics_description,
        examples=[{"Geography.Regions.Europe.Northern Europe": 0.584}],
    )

    _categories_description = (
        "A noisy list of article topics that are found on the Wikipedia page "
        "at the end. To note that the hierarchy of this category system can "
        "be found through the SQL database dumps according to this source. "
        "The reason these are noisy is that they sometimes contain meta data "
        "topics like `CS1 Swedish-language sources (sv)` or `Good articles`."
    )
    _categories_example = [
        "CS1 Swedish-language sources (sv)",
        "AC with 0 elements",
        "1870 births",
        "1949 deaths",
        "Academics of the University of Helsinki",
        "Finnish hematologists",
        "Finnish people of German descent",
        "People from Vaasa",
    ]
    categories: list[str] = Field(
        title="Categories",
        description=_categories_description,
        examples=[_categories_example],
    )

    _popularity_description = (
        "As defined in the cirrus schema, "
        "'A floating point number representing the "
        "percentage of page views to this wiki that "
        "requests this page. This is only available for "
        "content pages.' If the popularity score cannot "
        "be validated or found it will have a value of "
        "`None`."
    )
    popularity_score: float | None = Field(
        title="Popularity Score",
        description=_popularity_description,
        examples=[8.327128616319467e-08, None],
    )

    _timestamp_description = (
        "Timestamp of the most recently index/edited version "
        "of the page. If the timestamp cannot be found "
        "it will have a value of `None`."
    )
    timestamp: str | None = Field(
        title="timestamp",
        description=_timestamp_description,
        examples=["2021-12-26T18:49:13Z", None],
    )


def create_tagged_wiki_data_export_model(
    tokenizer_key: str,
    lemmas_key: str,
    pos_key: str,
    usas_key: str,
    usas_raw_key: str,
    sentence_boundaries_key: str,
) -> BaseModel:
    model_fields = {
        tokenizer_key: (list[str], Field(title="Tokens")),
        lemmas_key: (list[str], Field(title="Lemmas")),
        pos_key: (list[list[tuple[str, int]]], Field(title="POS")),
        usas_key: (list[list[USASTagGroup]], Field(title="USAS")),
        usas_raw_key: (list[str], Field(title="USAS Raw")),
        sentence_boundaries_key: (
            list[tuple[int, int]],
            Field(title="Sentence Boundaries"),
        ),
    }

    tagged_wiki_data_export_model_cls = create_model(
        "TaggedWikiDataExport", __base__=WikiDataExport, **model_fields
    )
    return tagged_wiki_data_export_model_cls

def get_usas_tag_descriptions(usas_tag_name: str,
                              usas_tag_dict: dict[str, Any],
                              collected_tag_descriptions: dict[str, str]
                              ) -> dict[str, str]:
    """
    A recursive function that returns a dictionary of a USAS tag and as a value 
    it's description.
    """
    if "title" in usas_tag_dict and "description" in usas_tag_dict:
        title_description = f"title: {usas_tag_dict['title']} description: {usas_tag_dict['description']}"
        if usas_tag_name in collected_tag_descriptions:
            raise KeyError(f"Duplicate usas tag name found: {usas_tag_name} "
                           "when reading the following data: "
                           f"{usas_tag_dict}, currently found usas tags: "
                           f"{collected_tag_descriptions}")
        collected_tag_descriptions[usas_tag_name] = title_description.strip()
    elif "title" in usas_tag_dict:
        raise KeyError("No description key found when it is expected for: "
                       f"{usas_tag_name} {usas_tag_dict}")
    elif "description" in usas_tag_dict:
        raise KeyError("No title key found when it is expected for: "
                       f"{usas_tag_name} {usas_tag_dict}")

    keys_to_ignore = set(["title", "description"])
    for child_usas_tag_name, child_usas_tag_dict in usas_tag_dict.items():
        if child_usas_tag_name not in keys_to_ignore:
            collected_tag_descriptions = get_usas_tag_descriptions(child_usas_tag_name,
                                                                   child_usas_tag_dict,
                                                                   collected_tag_descriptions)
    return collected_tag_descriptions

def load_usas_mapper(usas_tag_descriptions_file: Path,
                     tags_to_filter_out: set[str] | None) -> dict[str, str]:

    usas_mapping = {}
    with usas_tag_descriptions_file.open("r") as usas_mapper_fp:
        usas_mapping_data = usas_mapper_fp.read()
        for high_level_usas_tag, high_level_usas_tag_dict in yaml.safe_load(usas_mapping_data).items():
            get_usas_tag_descriptions(high_level_usas_tag, high_level_usas_tag_dict, usas_mapping)
    if tags_to_filter_out:
        tmp_usas_mapping = {}
        for key, value in usas_mapping.items():
            if key in tags_to_filter_out:
                continue
            tmp_usas_mapping[key] = value
        usas_mapping = tmp_usas_mapping
    return usas_mapping


class MosaicoDocumentID(BaseModel):
    dataset_id: str = Field(examples=["senseval2"])
    wikidata_id: str = Field(examples=["Q194422"])
    document_id: str = Field(examples=["34256325"])
    sentence_id: str = Field(examples=["5736275"])
    

    def __str__(self) -> str:

        return f"{self.dataset_id}.{self.wikidata_id}.{self.document_id}.{self.sentence_id}"
    
class MosaicoUSASSentence(BaseModel):
    _text_description = ("This is represented as the tokens for the given "
                         "sentence boundaries joined together by a single whitespace. "
                         "This is due to not being able to recover the "
                         "character offsets for a sentence boundary from the CLAWS tokenizer.")
    id: MosaicoDocumentID
    text: str = Field(description=_text_description)
    tokens: list[str]
    lemmas: list[str]
    pos: list[str]
    is_content_token: list[bool]
    usas: list[list[str]]
    usas_token_offsets: list[list[int]]



def parse_usas_document(usas_document: BaseModel) -> list[MosaicoUSASSentence]:
    dataset_id = "mosaico_core_usas"
    structured_usas_document: list[MosaicoUSASSentence] = []
    for sentence_id, sentence_boundary in enumerate(usas_document.sentence_boundaries):
        mosaico_id = MosaicoDocumentID(dataset_id=dataset_id,
                                       wikidata_id=usas_document.wikidata_id,
                                       document_id=usas_document.document_id,
                                       sentence_id=str(sentence_id))

        start_sentence_offset, end_sentence_offset = sentence_boundary
        
        relevant_tokens = usas_document.tokens[start_sentence_offset: end_sentence_offset]
        relevant_lemmas = usas_document.lemmas[start_sentence_offset: end_sentence_offset]
        sentence_text = " ".join(relevant_tokens)

        relevant_usas = usas_document.usas[start_sentence_offset: end_sentence_offset]
        relevant_token_offsets: list[list[int]] = []
        relevant_filtered_usas = []
        is_content_tokens = []
        for usas_index, usas_tag_groups in enumerate(relevant_usas):
            filtered_usas = []
            content_token = True
            for usas_tag_group in usas_tag_groups:
                for usas_tag in usas_tag_group.tags:
                    usas_tag = usas_tag.tag.upper()
                    if usas_tag == 'Z5':
                        content_token = False
                    if usas_tag in USAS_TAGS_TO_IGNORE:
                        continue
                    filtered_usas.append(usas_tag)
            if not filtered_usas:
                continue
            relevant_filtered_usas.append(filtered_usas)
            relevant_token_offsets.append([usas_index, usas_index + 1])
            is_content_tokens.append(content_token)
        if not relevant_filtered_usas:
            continue
        
        relevant_pos = usas_document.pos[start_sentence_offset: end_sentence_offset]
        relevant_filtered_pos = []
        for pos_tag_group in relevant_pos:
            if len(pos_tag_group) == 0:
                relevant_filtered_pos.append("")
            else:
                relevant_filtered_pos.append(pos_tag_group[0][0])
        
        
        
        

        structured_usas_document.append(MosaicoUSASSentence(id=mosaico_id,
                                   text=sentence_text,
                                   tokens=relevant_tokens,
                                   lemmas=relevant_lemmas,
                                   pos=relevant_filtered_pos,
                                   is_content_token=is_content_tokens,
                                   usas=relevant_filtered_usas,
                                   usas_token_offsets=relevant_token_offsets))
    return structured_usas_document


def process_file(usas_file: Path,
                 data_dir: Path,
                 file_name: str,
                 overwrite: bool = False,
                 ) -> Path:
    file_path = Path(data_dir, file_name)
    if file_path.exists() and not overwrite:
        logger.info(
            f"The file {file_path} already exists and therefore "
            "not writing the data to it."
        )
        return file_path
    usas_data_model = create_tagged_wiki_data_export_model("tokens", "lemmas", "pos", "usas", "usas_raw", "sentence_boundaries")
    with file_path.open("w", encoding="utf-8") as write_fp:
        with usas_file.open("r", encoding="utf-8") as usas_fp:
            for line in usas_fp:
                usas_tagged_sentences = parse_usas_document(usas_data_model.model_validate_json(line))
                for usas_tagged_sentence in usas_tagged_sentences:
                    write_fp.write(f"{usas_tagged_sentence.model_dump_json()}\n")
    return file_path


def get_usas_label_statistics(
    hf_dataset: datasets.Dataset,
    usas_mapper: dict[str, str] | None = None
) -> dict[str, int]:
    """
    Given a Mosaico USAS labelled dataset it returns the count of all the 
    USAS labels within that dataset

    Args:

        usas_mapper (dict[str, str] | None): If not None it will add the USAS 
            labels that are not within the dataset to the counter as entries 
            that contain 0 counts.
    """
    label_counter = Counter()
    for all_token_usas_tags in hf_dataset["usas"]:
        for token_usas_tags in all_token_usas_tags:
            label_counter.update(token_usas_tags)
    if usas_mapper:
        for usas_label in usas_mapper.keys():
            if usas_label not in label_counter:
                label_counter[usas_label] = 0
    return label_counter

def usas_inverse_label_statistics(usas_label_frequency: dict[str, int],
                                  log_scaled: int | None = None
                                  ) -> dict[str, float]:
    """
    Given a dictionary of USAS label to frequency within a dataset, it returns 
    the USAS label to inverse term frequency in that dataset.

    NOTE: If the frequency of a USAS label is 0 then that USAS label will be 
    skipped/ignored and will not be in the returned dictionary.

    Args:
        log_scaled (int | None): If an integer then the inverse term frequency will 
            be log scaled by a log to the base of this log scaled integer.
    """
    number_labels = sum(usas_label_frequency.values())
    usas_inverse_frequency = {}
    for usas_label, count in usas_label_frequency.items():
        if count == 0:
            continue
        inverse_frequency = number_labels / count
        if log_scaled is not None:
            inverse_frequency = math.log(inverse_frequency, log_scaled)
        usas_inverse_frequency[usas_label] = inverse_frequency
    return usas_inverse_frequency



def random_negative_usas_label(
        positive_usas_labels: list[str],
        negative_usas_labels: list[str],
        label_weights: dict[str, float],
        use_weights: bool
) -> str:
    """
    Samples a random USAS tag from `label_weights` after the `positive_usas_labels` 
    and `negative_usas_labels` have been removed from the possible USAS tags 
    that could be sampled.

    Args:
        positive_usas_labels (list[str]): The list of true USAS tags
        negative_usas_labels (list[str]): The list of negative USAS tags
        label_weights (dict[str, float]): A dictionary of all relevant USAS tags 
            and how much weight should be applied to them. To NOTE is `use_weight` 
            is False the weights are never used, in addition not all USAS tags 
            have to exist in this weighting only those that you would like 
            to sample from.
        use_weights (bool): Whether to use the weighting in `label_weights` whereby 
            the higher the weight the more likely the USAS tag will be randomly 
            sampled or if `False` an even random sample of all relevant tags 
            will be applied.
    Raises:
        ValueError: If the number of USAS tags to sample from after removing 
            `positive_usas_labels` and `negative_usas_labels` tags is zero.
    Returns:
        A randomly sampled USAS tag from the USAS tags that are the key's of 
        the `label_weights` argument after the `positive_usas_labels` and 
        `negative_usas_labels` tags have been removed from the list of 
        possible tags to sample from.
    """
    labels_to_ignore = set(positive_usas_labels + negative_usas_labels)
    relevant_labels: list[str] = []
    relevant_label_cumulative_weights: list[float] = []

    total_weight_value = 0.0
    for label, weight in label_weights.items():
        if label in labels_to_ignore:
            continue
        relevant_labels.append(label)
        relevant_label_cumulative_weights.append(total_weight_value)
        total_weight_value += weight
    if len(relevant_labels) == 0:
        raise ValueError("The number of USAS tags to sample from after removing "
                         "the positive and negative USAS tags is zero, which "
                         "cannot be the case. Positive and Negative USAS tags: "
                         f"{positive_usas_labels} and {negative_usas_labels} "
                         "The dictionary of USAS tags and their weights: "
                         f"{label_weights}")
    if use_weights:
        return random.choices(relevant_labels, k=1)[0]
    else:
        return random.choices(relevant_labels, cum_weights=relevant_label_cumulative_weights, k=1)[0]



