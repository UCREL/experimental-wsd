import copy
import csv
import logging
import statistics as std_statistics
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Callable

import spacy
import torch
from pydantic import BaseModel
from transformers import AutoTokenizer

from experimental_wsd.data_processing.processed_usas_utils import (
    USASTag,
    USASTagGroup,
    load_usas_mapper,
    parse_usas_token_group,
)
from experimental_wsd.nn.token_similarity import TokenSimilarityVariableNegatives

logger = logging.getLogger(__name__)


class PyMUSASSupportedLanguages(str, Enum):
    english = "english"
    finnish = "finnish"
    chinese = "chinese"
    welsh = "welsh"


class DatasetType(str, Enum):
    gold = "gold"
    predictions = "predictions"


class TextLevel(str, Enum):
    sentence = "sentence"
    paragraph = "paragraph"
    document = "document"


class EvaluationDatasetName(str, Enum):
    benedict_english = "benedict_english"
    benedict_finnish = "benedict_finnish"
    torch_chinese = "torch_chinese"
    corcencc_welsh = "corcencc_welsh"
    icc_irish = "icc_irish"


class EvaluationText(BaseModel):
    tokens: list[str]
    labels: list[list[USASTagGroup]]
    label_offsets: list[tuple[int, int]]
    lemmas: list[str] | None = None
    pos_tags: list[str] | None = None

    def flatten_labels(self) -> list[list[USASTag]]:
        labels: list[list[USASTag]] = []
        for label_groups in self.labels:
            token_labels = []
            for label_group in label_groups:
                token_labels.extend(label_group.tags)
            labels.append(token_labels)
        return labels

    @staticmethod
    def get_label_set(label_groups: list[USASTagGroup]) -> set[str]:
        label_set: set[str] = set()
        for label_group in label_groups:
            for label in label_group.tags:
                label_set.add(label.tag)
        return label_set


class NeuralInferenceModel:
    def __init__(
        self,
        model_path: Path,
        usas_mapper_path: Path,
        usas_tags_to_filter_out: set[str] | None,
    ) -> None:
        logger.debug(f"Neural model path: {model_path}")
        logger.debug(f"USAS mapper path: {usas_mapper_path}")

        usas_tag_to_description_mapper: dict[str, str] | None = None
        self.usas_tags_to_filter_out = usas_tags_to_filter_out

        if usas_mapper_path.suffix == ".yaml":
            usas_tag_to_description_mapper = load_usas_mapper(
                usas_mapper_path, self.usas_tags_to_filter_out
            )
        elif usas_mapper_path.suffix == ".json":
            import json

            with usas_mapper_path.open("r", encoding="utf-8") as mapper_fp:
                usas_tag_to_description_mapper = json.load(mapper_fp)
                if self.usas_tags_to_filter_out is not None:
                    for filter_tag in self.usas_tags_to_filter_out:
                        del usas_tag_to_description_mapper[filter_tag]
        self.usas_tag_to_description_mapper = usas_tag_to_description_mapper
        self.model = TokenSimilarityVariableNegatives.load_from_checkpoint(
            str(model_path)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.base_model_name, add_prefix_space=True
        )

        usas_embedded_descriptions_tensor: torch.Tensor | None = None
        usas_index_to_tag: dict[int, str] = {}
        self.model.eval()
        with torch.inference_mode(mode=True):
            usas_embedded_descriptions = []

            for index, usas_tag_description in enumerate(
                usas_tag_to_description_mapper.items()
            ):
                usas_tag, usas_description = usas_tag_description
                tokenized_usas_description = self.tokenizer(
                    usas_description,
                    truncation=False,
                    padding=False,
                    return_tensors="pt",
                )
                description_input_ids = tokenized_usas_description.input_ids.to(
                    self.model.device
                ).unsqueeze(0)
                description_attention_mask = (
                    tokenized_usas_description.attention_mask.to(
                        self.model.device
                    ).unsqueeze(0)
                )
                definition_embedding = self.model.label_definition_encoding(
                    description_input_ids, description_attention_mask
                )
                usas_embedded_descriptions.append(definition_embedding)
                usas_index_to_tag[index] = usas_tag
            usas_embedded_descriptions_tensor = torch.vstack(usas_embedded_descriptions)
            NUM_DESC, DESC_BATCH, EMBEDDING_DIM = (
                usas_embedded_descriptions_tensor.shape
            )
            usas_embedded_descriptions_tensor = usas_embedded_descriptions_tensor.view(
                DESC_BATCH, NUM_DESC, EMBEDDING_DIM
            )
        assert isinstance(usas_embedded_descriptions_tensor, torch.Tensor)
        self.usas_embedded_descriptions_tensor = usas_embedded_descriptions_tensor
        self.usas_index_to_tag = usas_index_to_tag

    def inference(
        self, tokens: list[str], token_indexes: list[tuple[int, int]], top_n: int
    ) -> list[list[USASTagGroup]]:
        self.model.eval()
        prediction_labels: list[list[USASTagGroup]] = []
        with torch.inference_mode(mode=True):
            tokenized_text = self.tokenizer(
                tokens,
                truncation=False,
                padding=False,
                return_tensors="pt",
                is_split_into_words=True,
            )
            print(tokenized_text)
            if tokenized_text.input_ids.shape[1] > self.tokenizer.model_max_length:
                raise ValueError("Text token length too large for model.")
            text_input_ids = tokenized_text.input_ids.to(self.model.device)
            text_attention_mask = tokenized_text.attention_mask.to(self.model.device)
            text_embedding = self.model.text_encoding(
                text_input_ids, text_attention_mask
            )

            for start_end_indexes in token_indexes:
                token_offset_indexes = set(range(*start_end_indexes))
                text_word_ids_mask = []
                for word_id in tokenized_text.word_ids():
                    if word_id is None:
                        text_word_ids_mask.append(0)
                    elif word_id in token_offset_indexes:
                        text_word_ids_mask.append(1)
                    else:
                        text_word_ids_mask.append(0)
                text_word_ids_mask = torch.tensor(text_word_ids_mask, dtype=torch.long)
                text_word_ids_mask = text_word_ids_mask.unsqueeze(0).to(
                    device=self.model.device
                )
                if text_word_ids_mask.sum() == 0:
                    raise ValueError(
                        "Cannot find the token offsets in the given sample."
                    )
                token_embedding = self.model.token_encoding_using_text_encoding(
                    text_embedding, text_word_ids_mask
                )
                label_similarity_score = self.model.token_label_similarity(
                    self.usas_embedded_descriptions_tensor, token_embedding
                )[0]
                top_n_sorted_label_similarity_score = (
                    torch.argsort(label_similarity_score, descending=True)[:top_n]
                    .cpu()
                    .tolist()
                )
                predicted_usas_tags = [
                    self.usas_index_to_tag[top_n_index]
                    for top_n_index in top_n_sorted_label_similarity_score
                ]
                prediction_usas_tag_groups = [
                    USASTagGroup(tags=[USASTag(tag=predicted_usas_tag)])
                    for predicted_usas_tag in predicted_usas_tags
                ]
                prediction_labels.append(prediction_usas_tag_groups)
        return prediction_labels

    def get_post_tagger_inference(
        self, top_n: int
    ) -> Callable[[list[str], tuple[int, int], list[str]], list[str]]:
        def post_tagger_inference(
            text_tokens: list[str],
            token_offsets: tuple[int, int],
            tagger_label_predictions: list[USASTagGroup],
        ) -> list[USASTagGroup]:
            inference_output: list[USASTagGroup] | None = None
            default_return = tagger_label_predictions
            if not tagger_label_predictions:
                inference_output = self.inference(text_tokens, [token_offsets], top_n)[
                    0
                ]
            elif len(tagger_label_predictions) == 0:
                inference_output = self.inference(text_tokens, [token_offsets], top_n)[
                    0
                ]
            elif len(tagger_label_predictions) == 1:
                tagger_label_prediction_tags = [
                    tag.tag for tag in tagger_label_predictions[0].tags
                ]
                if tagger_label_prediction_tags == ["Z99"]:
                    inference_output = self.inference(
                        text_tokens, [token_offsets], top_n
                    )[0]
                else:
                    return default_return
            else:
                return default_return

            assert isinstance(inference_output, list)
            return inference_output
            # return [usas_tag.tag for usas_tag_group in inference_output for usas_tag in usas_tag_group.tags]

        return post_tagger_inference

    # def get_post_tagger_reranking(self) -> Callable[[list[str], tuple[int, int], list[str]], list[str]]:

    #    def post_tagger_reranking(text_tokens: list[str],
    #                              token_offsets: tuple[int, int],
    #                              tagger_label_predictions: list[str]
    #                              ) -> list[str]:
    #        if not tagger_label_predictions:
    #            return tagger_label_predictions
    #        elif len(tagger_label_predictions) == 1:
    #            return tagger_label_predictions
    #        inference_output = self.inference(text_tokens, [token_offsets], top_n=-1)[0]
    #        tagger_label_predictions_set = set(tagger_label_predictions)
    #        ranked_predictions = []
    #        for usas_tag_group in inference_output:
    #            for usas_tag in usas_tag_group.tags:
    #                tag = usas_tag.tag
    #                if tag in tagger_label_predictions_set:
    #                    ranked_predictions.append(tag)
    #        return ranked_predictions
    #    return post_tagger_reranking


class EvaluationDataset(BaseModel):
    texts: dict[str, list[EvaluationText]]
    name: EvaluationDatasetName
    text_level: TextLevel
    usas_tags_to_filter_out: set[str] | None
    dataset_type: DatasetType

    def get_first_label_only(self) -> "EvaluationDataset":
        all_filtered_texts: dict[str, list[EvaluationText]] = {}
        label_groups_per_token_distribution = self.get_statistics()[
            "label_groups_per_token_distribution"
        ]
        if (
            len(label_groups_per_token_distribution) == 1
            and label_groups_per_token_distribution.get(1, 0) > 0
        ):
            all_filtered_texts = self.texts
        else:
            for text_name, texts in self.texts.items():
                filtered_texts: list[EvaluationText] = []
                for text in texts:
                    assert isinstance(text, EvaluationText)
                    new_tokens = copy.deepcopy(text.tokens)
                    new_lemmas = copy.deepcopy(text.lemmas)
                    new_pos_tags = copy.deepcopy(text.pos_tags)
                    new_labels = []
                    new_label_offsets = copy.deepcopy(text.label_offsets)
                    for label in text.labels:
                        if len(label) == 1:
                            new_labels.append(copy.deepcopy(label))
                        else:
                            new_labels.append([copy.deepcopy(label[0])])
                    filtered_texts.append(
                        EvaluationText(
                            tokens=new_tokens,
                            lemmas=new_lemmas,
                            pos_tags=new_pos_tags,
                            labels=new_labels,
                            label_offsets=new_label_offsets,
                        )
                    )
                all_filtered_texts[text_name] = filtered_texts

        return EvaluationDataset(
            texts=all_filtered_texts,
            name=self.name,
            text_level=self.text_level,
            usas_tags_to_filter_out=self.usas_tags_to_filter_out,
            dataset_type=self.dataset_type,
        )

    def get_evaluation_texts(self) -> list[EvaluationText]:
        evaluation_texts: list[EvaluationText] = []
        for texts in self.texts.values():
            evaluation_texts.extend(texts)
        return evaluation_texts

    def get_statistics(self) -> dict[str, int | float | dict[int, int]]:
        text_token_count: list[int] = []
        text_label_count: list[int] = []
        label_lengths: list[int] = []
        tags_per_token_distribution: dict[int, int] = defaultdict(lambda: 0)
        tag_groups_per_token_distribution: dict[int, int] = defaultdict(lambda: 0)
        number_texts = 0

        for text in self.get_evaluation_texts():
            token_count = len(text.tokens)
            text_token_count.append(token_count)

            label_count = len(text.labels)
            text_label_count.append(label_count)

            for label_groups in text.labels:
                tag_groups_per_token_distribution[len(label_groups)] += 1
                for label_group in label_groups:
                    assert isinstance(label_group, USASTagGroup)
                    tags_per_token_distribution[len(label_group.tags)] += 1

            for label_offsets in text.label_offsets:
                label_lengths.append(len(range(*label_offsets)))
            number_texts += 1

        mean_token_count = std_statistics.mean(text_token_count)
        total_token_count = sum(text_token_count)
        mean_label_count = std_statistics.mean(text_label_count)
        total_label_count = sum(text_label_count)
        average_label_length = std_statistics.mean(label_lengths)

        return {
            "mean_token_count": mean_token_count,
            "total_token_count": total_token_count,
            "mean_token_label_count": mean_label_count,
            "total_token_label_count": total_label_count,
            "average_label_length": average_label_length,
            "number_texts": number_texts,
            "label_groups_per_token_distribution": dict(
                tag_groups_per_token_distribution
            ),
            "labels_per_token_distribution": dict(tags_per_token_distribution),
        }

    def get_label_distribution(self) -> dict[str, int]:
        label_distribution = Counter()
        for text in self.get_evaluation_texts():
            for label_groups in text.labels:
                for label_group in label_groups:
                    assert isinstance(label_group, USASTagGroup)
                    label_distribution.update(
                        [usas_tag.tag for usas_tag in label_group.tags]
                    )
        return dict(label_distribution)

    def rule_based_inference(
        self,
        usas_mapper_path: Path,
        language: PyMUSASSupportedLanguages,
        post_tagger_function: Callable[
            [list[str], tuple[int, int], list[USASTagGroup]], list[USASTagGroup]
        ]
        | None = None,
    ) -> "EvaluationDataset":
        """
        Returns:
            tuple["EvaluationDataset", int]: The EvaluationDataset with the
                labels representing the predictions from the rule based tagger.
        """
        prediction_texts: dict[list[EvaluationText]] = {}
        logger.debug(f"Starting {language} rule based inference")

        def filter_pymusas_tags(pymusas_tags: list[str]) -> list[USASTagGroup]:
            parsed_usas_tag_groups = parse_usas_token_group(pymusas_tags, None)
            filtered_usas_tag_groups: list[USASTagGroup] = []
            for usas_tag_group in parsed_usas_tag_groups:
                assert isinstance(usas_tag_group, USASTagGroup)
                filtered_usas_tags: list[USASTag] = []
                for usas_tag in usas_tag_group.tags:
                    new_usas_tag = usas_tag.tag
                    if usas_tag.tag == "D":
                        continue
                    if usas_tag.tag == "P":
                        new_usas_tag = "P1"
                    if usas_tag.tag == "A1.1":
                        new_usas_tag = "A1.1.1"
                    if usas_tag.tag == "S4.1.2.1":
                        new_usas_tag = "S4"
                    if usas_tag.tag == "M3.3":
                        new_usas_tag = "M3"
                    if usas_tag.tag == "S":
                        continue
                    filtered_usas_tags.append(USASTag(tag=new_usas_tag))
                filtered_usas_tag_groups.append(USASTagGroup(tags=filtered_usas_tags))
            return filtered_usas_tag_groups

        nlp = None
        if language == PyMUSASSupportedLanguages.english:
            import en_core_web_trf

            nlp = en_core_web_trf.load(exclude=["parser", "ner"])
            english_tagger_pipeline = spacy.load("en_dual_none_contextual")
            nlp.add_pipe("pymusas_rule_based_tagger", source=english_tagger_pipeline)
        elif language == PyMUSASSupportedLanguages.finnish:
            import fi_core_news_lg

            nlp = fi_core_news_lg.load(
                exclude=["tagger", "parser", "attribute_ruler", "ner"]
            )
            finnish_tagger_pipeline = spacy.load("fi_single_upos2usas_contextual")
            nlp.add_pipe("pymusas_rule_based_tagger", source=finnish_tagger_pipeline)
        elif language == PyMUSASSupportedLanguages.chinese:
            import zh_core_web_trf

            nlp = zh_core_web_trf.load(exclude=["parser", "ner"])
            chinese_tagger_pipeline = spacy.load("cmn_dual_upos2usas_contextual")
            nlp.add_pipe("pymusas_rule_based_tagger", source=chinese_tagger_pipeline)
        elif language == PyMUSASSupportedLanguages.welsh:
            nlp = spacy.load("cy_dual_basiccorcencc2usas_contextual")

        for text_name, texts in self.texts.items():
            evaluation_texts: list[EvaluationText] = []
            for text_index, text in enumerate(texts):
                prediction_tokens = copy.deepcopy(text.tokens)
                prediction_labels: list[list[USASTagGroup]] = []
                prediction_label_offsets: list[tuple[int, int]] = []

                token_spaces = [True] * len(prediction_tokens)
                doc_kwargs = {"words": prediction_tokens, "spaces": token_spaces}
                if text.pos_tags:
                    doc_kwargs["tags"] = text.pos_tags
                if text.lemmas:
                    doc_kwargs["lemmas"] = text.lemmas

                doc = spacy.tokens.Doc(nlp.vocab, **doc_kwargs)
                output_doc = nlp(doc)

                for usas_tag_offsets in text.label_offsets:
                    usas_token_offset_indexes = list(range(*usas_tag_offsets))
                    if len(usas_token_offset_indexes) != 1:
                        raise ValueError(
                            "Can only evaluate rule based system "
                            "with labels that cover only one token"
                        )
                    token_offset_index = usas_token_offset_indexes[0]
                    pymusas_tags = output_doc[token_offset_index]._.pymusas_tags
                    filtered_pymusas_tags = filter_pymusas_tags(pymusas_tags)
                    if post_tagger_function is not None:
                        filtered_pymusas_tags = post_tagger_function(
                            prediction_tokens, usas_tag_offsets, filtered_pymusas_tags
                        )
                    # prediction_labels.append(parse_usas_token_group(filtered_pymusas_tags, valid_usas_tags))
                    prediction_labels.append(filtered_pymusas_tags)
                    prediction_label_offsets.append(usas_tag_offsets)

                evaluation_texts.append(
                    EvaluationText(
                        tokens=prediction_tokens,
                        labels=prediction_labels,
                        label_offsets=prediction_label_offsets,
                    )
                )
            prediction_texts[text_name] = evaluation_texts

        logger.debug("Finished rule based inference")

        return EvaluationDataset(
            texts=prediction_texts,
            name=self.name,
            text_level=self.text_level,
            usas_tags_to_filter_out=self.usas_tags_to_filter_out,
            dataset_type=DatasetType.predictions,
        )

    def neural_inference(
        self,
        model: NeuralInferenceModel,
        top_n: int,
    ) -> "EvaluationDataset":
        prediction_texts: dict[list[EvaluationText]] = {}

        for text_name, texts in self.texts.items():
            evaluation_texts: list[EvaluationText] = []
            for text_index, text in enumerate(texts):
                prediction_tokens = copy.deepcopy(text.tokens)
                prediction_label_offsets: list[tuple[int, int]] = copy.deepcopy(
                    text.label_offsets
                )
                prediction_labels: list[list[USASTagGroup]] = model.inference(
                    prediction_tokens, prediction_label_offsets, top_n
                )
                evaluation_texts.append(
                    EvaluationText(
                        tokens=prediction_tokens,
                        labels=prediction_labels,
                        label_offsets=prediction_label_offsets,
                    )
                )
            prediction_texts[text_name] = evaluation_texts
        logger.debug("Finished neural inference")

        return EvaluationDataset(
            texts=prediction_texts,
            name=self.name,
            text_level=self.text_level,
            usas_tags_to_filter_out=self.usas_tags_to_filter_out,
            dataset_type=DatasetType.predictions,
        )


def parse_benedict_english(
    dataset_path: Path,
    usas_mapper_path: Path,
    usas_tags_to_filter_out: list[str] | None,
) -> EvaluationDataset:
    """
    Parses the Benedict English corpus into the Evaluation Dataset format, for
    easy evaluation of USAS WSD models.

    All `PUNC` tags are ignored, note the token it refers to will still exist
    in the tokens of the returned object.

    Args:
        dataset_path (Path): Path to the Benedict English corpus.
        usas_mapper_path (Path): Path to YAML object that maps USAS tags to
            their description.
        usas_tags_to_filter_out (list[str] | None): A list of USAS tags to
            filter out of the complete list of USAS tags defined by the USAS
            mapper. This is used to define what USAS tags should be in the
            evaluation dataset, e.g. Z99 this HOWEVER will not
            mean that the evaluation dataset filters these tags out rather it
            will raise a ValueError if it finds one of these invalid tags.
    Returns:
        EvaluationDataset: The parsed and formatted dataset.
    Raises:
        ValueError: If no underscore is found in an annotated token.
        ValueError: If the token with the tag is an empty token.
        ValueError: If an expected token can be parsed as a USAS tag.
        ValueError: If more than one underscore is found, only one should be found
            which should be between the token and the USAS tag.
        ValueError: If a USAS tag cannot be parsed, this could be due to it not
            being a valid USAS tag.
    """

    logger.info(f"Parsing the Benedict English dataset found at: {dataset_path}")
    logger.info(f"USAS mapper path: {usas_mapper_path}")
    logger.info(f"USAS tags being filtered out: {usas_tags_to_filter_out}")
    usas_tags_to_filter_out_set = set(usas_tags_to_filter_out)
    valid_usas_tags = set(
        load_usas_mapper(usas_mapper_path, usas_tags_to_filter_out_set).keys()
    )
    evaluation_texts: list[EvaluationText] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_fp:
        for line_index, line in enumerate(dataset_fp):
            line = line.strip()
            logger.debug(f"Line index: {line_index}")
            token_tags = line.split(" ")
            logger.debug(f"Number of tokens in line: {len(token_tags)}")

            text_tokens: list[str] = []
            labels: list[list[USASTagGroup]] = []
            label_offsets: list[tuple[int, int]] = []

            for token_index, token_tag in enumerate(token_tags):
                if "_" not in token_tag:
                    raise ValueError(
                        f"Error no underscore {line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                    )
                token_split = token_tag.split("_")
                token = token_split[0]

                if not token:
                    raise ValueError(
                        "The token with the USAS label is empty: "
                        f"{line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                    )

                token_is_a_tag = True
                try:
                    parse_usas_token_group([token], valid_usas_tags)
                except ValueError:
                    token_is_a_tag = False
                if token_is_a_tag:
                    raise ValueError(
                        f"Error expected token is a tag: {line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                    )

                if len(token_split) > 2:
                    raise ValueError(
                        "Error too many underscores, expect only one "
                        "underscore between the token and the tag "
                        f"{line_index}/{token_index} "
                        f"(line_index/token_index): {token_tag}"
                    )

                tag_groups: list[USASTagGroup] | None = None
                token, tag = token_split
                if tag == "PUNC":
                    pass
                elif token == "." and tag == ".":
                    tag = "PUNC"
                elif token == "," and tag == ",":
                    tag = "PUNC"
                elif token == "!" and tag == "!":
                    tag = "PUNC"
                elif token == "-" and tag == "-":
                    tag = "PUNC"
                else:
                    try:
                        tag_groups = parse_usas_token_group([tag], valid_usas_tags)
                    except ValueError as parsing_value_error:
                        logger.exception(
                            f"Error cannot parse the USAS tag {line_index}/{token_index} (line_index/token_index): {token_tag} {token} {tag}"
                        )
                        raise parsing_value_error

                text_tokens.append(token)
                if tag == "PUNC":
                    continue

                no_tag_groups = False
                if tag_groups is None:
                    no_tag_groups = True
                elif len(tag_groups) == 0:
                    no_tag_groups = True

                if no_tag_groups:
                    raise ValueError(
                        "Error cannot parse the USAS tag "
                        f"{line_index}/{token_index} "
                        f"(line_index/token_index): {token_tag}"
                    )

                labels.append(tag_groups)
                label_offsets.append((token_index, token_index + 1))

            evaluation_texts.append(
                EvaluationText(
                    tokens=text_tokens, labels=labels, label_offsets=label_offsets
                )
            )

    logger.info("Finished parsing the Benedict English dataset")
    return EvaluationDataset(
        texts={EvaluationDatasetName.benedict_english.value: evaluation_texts},
        name=EvaluationDatasetName.benedict_english,
        text_level=TextLevel.sentence,
        usas_tags_to_filter_out=usas_tags_to_filter_out_set,
        dataset_type=DatasetType.gold,
    )


def parse_benedict_finnish(
    dataset_path: Path,
    usas_mapper_path: Path,
    usas_tags_to_filter_out: list[str] | None,
) -> EvaluationDataset:
    """
    Parses the Benedict Finnish corpus into the Evaluation Dataset format, for
    easy evaluation of USAS WSD models.

    All `PUNC` tags are ignored, note the token it refers to will still exist
    in the tokens of the returned object.

    NOTE: Unlike the English Benedict dataset this dataset contains an `_i`
    at the end of token tag to indicate it is part of a MWE which is different
    to the normal convention of `iXX.Y.Z`. The dataset parsing at the moment does not
    support MWE but this is a note to remind us for the future when we do
    want to evaluate and support MWEs.

    Args:
        dataset_path (Path): Path to the Benedict Finnish corpus.
        usas_mapper_path (Path): Path to YAML object that maps USAS tags to
            their description.
        usas_tags_to_filter_out (list[str] | None): A list of USAS tags to
            filter out of the complete list of USAS tags defined by the USAS
            mapper. This is used to define what USAS tags should be in the
            evaluation dataset, e.g. Z99 this HOWEVER will not
            mean that the evaluation dataset filters these tags out rather it
            will raise a ValueError if it finds one of these invalid tags.
    Returns:
        EvaluationDataset: The parsed and formatted dataset.
    Raises:
        ValueError: If no underscore is found in an annotated token.
        ValueError: If the token with the tag is an empty token.
        ValueError: If an expected token can be parsed as a USAS tag.
        ValueError: If more than one underscore is found, only one should be found
            which should be between the token and the USAS tag.
        ValueError: If a USAS tag cannot be parsed, this could be due to it not
            being a valid USAS tag.
    """

    logger.info(f"Parsing the Benedict Finnish dataset found at: {dataset_path}")
    logger.info(f"USAS mapper path: {usas_mapper_path}")
    logger.info(f"USAS tags being filtered out: {usas_tags_to_filter_out}")
    usas_tags_to_filter_out_set = set(usas_tags_to_filter_out)
    valid_usas_tags = set(
        load_usas_mapper(usas_mapper_path, usas_tags_to_filter_out_set).keys()
    )

    evaluation_texts: list[EvaluationText] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_fp:
        for line_index, line in enumerate(dataset_fp):
            line = line.strip()
            logger.debug(f"Line index: {line_index}")
            token_tags = line.split(" ")
            logger.debug(f"Number of tokens in line: {len(token_tags)}")

            text_tokens: list[str] = []
            labels: list[list[USASTagGroup]] = []
            label_offsets: list[tuple[int, int]] = []

            for token_index, token_tag in enumerate(token_tags):
                if "_" not in token_tag:
                    if token_tag == ".":
                        token_tag = "._PUNC"
                    elif token_tag == ",":
                        token_tag = ",_PUNC"
                    elif token_tag == ":":
                        token_tag = ":_PUNC"
                    elif token_tag == "!":
                        token_tag = "!_PUNC"
                    elif token_tag == "(":
                        token_tag = "(_PUNC"
                    elif token_tag == ")":
                        token_tag = ")_PUNC"
                    elif token_tag == "?":
                        token_tag = "?_PUNC"
                    elif token_tag == "-":
                        token_tag = "-_PUNC"
                    elif token_tag == '"':
                        token_tag = '"_PUNC'
                    else:
                        raise ValueError(
                            f"Error no underscore {line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                        )

                token_split = token_tag.split("_")
                token = token_split[0]

                if not token:
                    raise ValueError(
                        "The token with the USAS label is empty: "
                        f"{line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                    )

                token_is_a_tag = True
                try:
                    parse_usas_token_group([token], valid_usas_tags)
                except ValueError:
                    token_is_a_tag = False
                if token_is_a_tag:
                    raise ValueError(
                        f"Error expected token is a tag: {line_index}/{token_index} (line_index/token_index): `{token_tag}`"
                    )

                token = token_split[0]
                tags = token_split[1:]
                last_tag_index = len(tags) - 1

                if tags[last_tag_index] == "i":
                    tags = tags[:-1]

                text_tokens.append(token)
                if tags == ["PUNC"]:
                    continue

                tag_groups: list[USASTagGroup] | None = None
                try:
                    tag_groups = parse_usas_token_group(tags, valid_usas_tags)
                except ValueError as parsing_value_error:
                    logger.exception(
                        f"Error cannot parse the USAS tag {line_index}/{token_index} (line_index/token_index): {token_tag} {token}"
                    )
                    raise parsing_value_error

                no_tag_groups = False
                if tag_groups is None:
                    no_tag_groups = True
                elif len(tag_groups) == 0:
                    no_tag_groups = True

                if no_tag_groups:
                    raise ValueError(
                        "Error cannot parse the USAS tag "
                        f"{line_index}/{token_index} "
                        f"(line_index/token_index): {token_tag}"
                    )
                labels.append(tag_groups)
                label_offsets.append((token_index, token_index + 1))

            evaluation_texts.append(
                EvaluationText(
                    tokens=text_tokens, labels=labels, label_offsets=label_offsets
                )
            )

    logger.info("Finished parsing the Benedict Finnish dataset")
    return EvaluationDataset(
        texts={EvaluationDatasetName.benedict_finnish.value: evaluation_texts},
        name=EvaluationDatasetName.benedict_finnish,
        text_level=TextLevel.sentence,
        usas_tags_to_filter_out=usas_tags_to_filter_out_set,
        dataset_type=DatasetType.gold,
    )


def parse_torch_chinese(
    dataset_path: Path,
    usas_mapper_path: Path,
    usas_tags_to_filter_out: list[str] | None,
) -> EvaluationDataset:
    """
    Parses the ToRCH2019 A26 Chinese corpus into the Evaluation Dataset format for
    easy evaluation of USAS WSD models.

    All `PUNCT` tags are ignored, note the token it refers to will still exist
    in the tokens of the returned object.

    Args:
        dataset_path (Path): Path to the ToRCH2019 A26 Chinese corpus.
        usas_mapper_path (Path): Path to YAML object that maps USAS tags to
            their description.
        usas_tags_to_filter_out (list[str] | None): A list of USAS tags to
            filter out of the complete list of USAS tags defined by the USAS
            mapper. This is used to define what USAS tags should be in the
            evaluation dataset, e.g. Z99 this HOWEVER will not
            mean that the evaluation dataset filters these tags out rather it
            will raise a ValueError if it finds one of these invalid tags.
    Returns:
        EvaluationDataset: The parsed and formatted dataset.
    Raises:
        ValueError: If no underscore is found in an annotated token.
        ValueError: If the token with the tag is an empty token.
        ValueError: If an expected token can be parsed as a USAS tag.
        ValueError: If more than one underscore is found, only one should be found
            which should be between the token and the USAS tag.
        ValueError: If a USAS tag cannot be parsed, this could be due to it not
            being a valid USAS tag.
    """

    def label_string_to_labels(label_string: str) -> list[str]:
        label_string = label_string.strip()
        labels: list[str] | None = None
        if "；" in label_string:
            labels = label_string.split("；")
        else:
            labels = label_string.split(";")
        cleaned_labels = []
        for label in labels:
            label = label.strip()
            if label:
                cleaned_labels.append(label)
        return cleaned_labels

    logger.info(f"Parsing the ToRCH2019 A26 Chinese dataset found at: {dataset_path}")
    logger.info(f"USAS mapper path: {usas_mapper_path}")
    logger.info(f"USAS tags being filtered out: {usas_tags_to_filter_out}")
    usas_tags_to_filter_out_set = set(usas_tags_to_filter_out)
    valid_usas_tags = set(
        load_usas_mapper(usas_mapper_path, usas_tags_to_filter_out_set).keys()
    )

    Z99_count = 0
    quantifier_row_indexes = set(
        [
            23,
            53,
            88,
            92,
            111,
            148,
            165,
            191,
            252,
            285,
            321,
            389,
            535,
            559,
            620,
            680,
            791,
            820,
            834,
            885,
            914,
            941,
            1026,
            1036,
            1049,
            1102,
            1109,
            1113,
            1117,
            1125,
            1129,
            1136,
            1162,
            1174,
            1199,
        ]
    )

    evaluation_texts: list[EvaluationText] = []
    sentence_bool_mapper = {"False": False, "True": True}

    with dataset_path.open("r", encoding="utf-8", newline="") as dataset_fp:
        dataset_csv_reader = csv.DictReader(dataset_fp)

        text_tokens: list[str] = []
        labels: list[list[USASTagGroup]] = []
        label_offsets: list[tuple[int, int]] = []
        token_index = 0
        for row_index, dataset_row in enumerate(dataset_csv_reader, start=2):
            token = dataset_row["Token"].strip()
            corrected_usas = label_string_to_labels(dataset_row["Corrected-USAS"])
            predicted_usas = label_string_to_labels(dataset_row["Predicted-USAS"])
            is_end_of_sentence = sentence_bool_mapper[dataset_row["sentence-break"]]

            token_is_a_tag = True
            try:
                parse_usas_token_group([token], valid_usas_tags)
            except ValueError:
                token_is_a_tag = False
            if token_is_a_tag:
                raise ValueError(
                    f"Error expected token is a tag: {row_index} "
                    f"(row index): {dataset_row}"
                )

            if not corrected_usas and predicted_usas == ["PUNCT"]:
                pass
            elif corrected_usas == ["Z99"]:
                Z99_count += 1
            elif row_index == 78 and corrected_usas == ["A1"]:
                pass
            elif row_index == 118 and corrected_usas == ["H1", "5.1"]:
                pass
            elif row_index == 1241 and corrected_usas == ["S3.1", "02"]:
                pass
            elif row_index == 1265 and corrected_usas == ["N4", "Z99"]:
                pass
            elif row_index == 1457 and corrected_usas == ["E4"]:
                pass
            elif row_index == 1450 and corrected_usas == ["Q1.1", "Q3", "S1.1"]:
                pass
            elif row_index == 1544 and corrected_usas == ["Z99", "T1.1"]:
                pass
            elif row_index == 1553 and corrected_usas == ["Z99", "T1.1"]:
                pass
            elif row_index == 1560 and corrected_usas == ["Z99", "T1.1"]:
                pass
            elif row_index == 1705 and corrected_usas == ["N99"]:
                pass
            elif row_index == 1706 and corrected_usas == ["N99"]:
                pass
            elif row_index == 1721 and corrected_usas == ["S2", "S7"]:
                pass
            elif row_index == 1764 and corrected_usas == ["E2", "E4"]:
                pass
            elif row_index == 1768 and corrected_usas == ["N99"]:
                pass
            elif row_index == 1770 and corrected_usas == ["E1", "E4"]:
                pass
            elif row_index == 1944 and corrected_usas == ["N3.2", "N13.2"]:
                pass
            elif row_index == 2156 and corrected_usas == ["E1", "E4"]:
                pass
            elif row_index == 2172 and corrected_usas == ["Z99", "T1.1"]:
                pass
            elif row_index == 2283 and corrected_usas == ["Z99", "T1.1"]:
                pass
            elif row_index == 663 and not corrected_usas:
                pass
            else:
                if not corrected_usas:
                    raise ValueError(
                        f"No gold label USAS tags for row index: "
                        f"{row_index} with the following data: {dataset_row}"
                    )

                if row_index in quantifier_row_indexes and corrected_usas == ["N"]:
                    corrected_usas = ["N5"]

                tag_groups: list[USASTagGroup] | None = None
                try:
                    tag_groups = parse_usas_token_group(corrected_usas, valid_usas_tags)
                except ValueError as parsing_value_error:
                    logger.exception(
                        f"Error cannot parse the USAS tag {row_index} "
                        f"(row_index): {dataset_row}"
                    )
                    raise parsing_value_error

                no_tag_groups = False
                if tag_groups is None:
                    no_tag_groups = True
                elif len(tag_groups) == 0:
                    no_tag_groups = True

                if no_tag_groups:
                    raise ValueError(
                        f"Error cannot parse the USAS tag {row_index}"
                        f"(row_index): {dataset_row}"
                    )
                labels.append(tag_groups)
                label_offsets.append((token_index, token_index + 1))

            token_index += 1
            text_tokens.append(token)

            if is_end_of_sentence:
                evaluation_texts.append(
                    EvaluationText(
                        tokens=text_tokens, labels=labels, label_offsets=label_offsets
                    )
                )
                token_index = 0
                text_tokens = []
                labels = []
                label_offsets = []
                is_end_of_sentence = False
    if len(evaluation_texts) != 46:
        raise ValueError(
            "The ToRCH 2019 A26 dataset should contain 46 sentences "
            f"but this version contains: {len(evaluation_texts)}"
        )
    logger.info(f"Ignored Z99 labels, found: {Z99_count:,} label tokens.")
    logger.info("Finished parsing the ToRCH2019 A26 Chinese dataset")
    return EvaluationDataset(
        texts={EvaluationDatasetName.torch_chinese.value: evaluation_texts},
        name=EvaluationDatasetName.torch_chinese,
        text_level=TextLevel.sentence,
        usas_tags_to_filter_out=usas_tags_to_filter_out_set,
        dataset_type=DatasetType.gold,
    )


def parse_corcencc_welsh(
    dataset_path: Path,
    usas_mapper_path: Path,
    usas_tags_to_filter_out: list[str] | None,
) -> EvaluationDataset:
    """
    Parses the CorCenCC Welsh corpus into the Evaluation Dataset format for
    easy evaluation of USAS WSD models.

    All `PUNCT` tags are ignored, note the token it refers to will still exist
    in the tokens of the returned object.

    Args:
        dataset_path (Path): Path to the CorCenCC corpus.
        usas_mapper_path (Path): Path to YAML object that maps USAS tags to
            their description.
        usas_tags_to_filter_out (list[str] | None): A list of USAS tags to
            filter out of the complete list of USAS tags defined by the USAS
            mapper. This is used to define what USAS tags should be in the
            evaluation dataset, e.g. Z99 this HOWEVER will not
            mean that the evaluation dataset filters these tags out rather it
            will raise a ValueError if it finds one of these invalid tags.
    Returns:
        EvaluationDataset: The parsed and formatted dataset.
    Raises:
        ValueError: If no underscore is found in an annotated token.
        ValueError: If the token with the tag is an empty token.
        ValueError: If an expected token can be parsed as a USAS tag.
        ValueError: If more than one underscore is found, only one should be found
            which should be between the token and the USAS tag.
        ValueError: If a USAS tag cannot be parsed, this could be due to it not
            being a valid USAS tag.
    """

    def label_string_to_labels(label_string: str) -> list[str]:
        label_string = label_string.strip()
        labels: list[str] | None = None
        if "；" in label_string:
            labels = label_string.split("；")
        else:
            labels = label_string.split(";")
        cleaned_labels = []
        for label in labels:
            label = label.strip()
            if label:
                cleaned_labels.append(label)
        return cleaned_labels

    logger.info(f"Parsing the CorCenCC dataset found at: {dataset_path}")
    logger.info(f"USAS mapper path: {usas_mapper_path}")
    logger.info(f"USAS tags being filtered out: {usas_tags_to_filter_out}")
    usas_tags_to_filter_out_set = set(usas_tags_to_filter_out)
    valid_usas_tags = set(
        load_usas_mapper(usas_mapper_path, usas_tags_to_filter_out_set).keys()
    )

    Z99_count = 0

    evaluation_texts: list[EvaluationText] = []

    with dataset_path.open("r", encoding="utf-8") as dataset_fp:
        for line_index, line in enumerate(dataset_fp):
            text_tokens: list[str] = []
            lemmas: list[str] = []
            pos_tags: list[str] = []
            labels: list[list[USASTagGroup]] = []
            label_offsets: list[tuple[int, int]] = []

            line = line.strip()
            if not line:
                continue

            all_token_data = line.split()
            for token_index, token_data in enumerate(all_token_data):
                token, lemma, _, _, _, pos_tag, usas_tag = token_data.split("|")
                token_is_a_tag = True
                try:
                    if token == "S4C":
                        token_is_a_tag = False
                    else:
                        parse_usas_token_group([token], valid_usas_tags)
                except ValueError:
                    token_is_a_tag = False
                if token_is_a_tag:
                    raise ValueError(
                        f"Error expected token is a tag: {line_index}/{token_index} (line_index/token_index): `{token_data}`"
                    )

                if usas_tag == "Z99":
                    Z99_count += 1
                elif usas_tag == "PUNCT":
                    pass
                elif (
                    line_index == 19
                    and token_index == 2
                    and token == "swyddogaethau"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 19
                    and token_index == 20
                    and token == "swyddogaethau"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 23
                    and token_index == 8
                    and token == "adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 23
                    and token_index == 37
                    and token == "welliannau"
                    and usas_tag == "!ERR"
                ):
                    pass
                elif (
                    line_index == 26
                    and token_index == 9
                    and token == "rôl"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 27
                    and token_index == 15
                    and token == "adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 29
                    and token_index == 16
                    and token == "gwelliannau"
                    and usas_tag == "!ERR"
                ):
                    pass
                elif (
                    line_index == 35
                    and token_index == 2
                    and token == "adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 40
                    and token_index == 7
                    and token == "adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 43
                    and token_index == 8
                    and token == "adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 46
                    and token_index == 26
                    and token == "allweddol"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 63
                    and token_index == 18
                    and token == "gwelliannau"
                    and usas_tag == "!ERR"
                ):
                    pass
                elif (
                    line_index == 67
                    and token_index == 37
                    and token == "gweithle"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 80
                    and token_index == 8
                    and token == "hollbwysig"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 83
                    and token_index == 18
                    and token == "gyffredinol"
                    and usas_tag == "A1"
                ):
                    pass
                elif (
                    line_index == 86
                    and token_index == 15
                    and token == "gweithredwyr"
                    and usas_tag == "I3/S2mf"
                ):
                    pass
                elif (
                    line_index == 74
                    and token_index == 3
                    and token == "ddyletswydd"
                    and usas_tag == "I3/S7"
                ):
                    pass
                elif (
                    line_index == 21
                    and token_index == 15
                    and token == "ddyletswydd"
                    and usas_tag == "I3/S7"
                ):
                    pass
                elif (
                    line_index == 94
                    and token_index == 4
                    and token == "benodir"
                    and usas_tag == "S7/X6"
                ):
                    pass
                elif (
                    line_index == 96
                    and token_index == 1
                    and token == "rôl"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 104
                    and token_index == 14
                    and token == "swyddogol"
                    and usas_tag == "A11/A10"
                ):
                    pass
                elif (
                    line_index == 115
                    and token_index == 17
                    and token == "frasamcanu"
                    and usas_tag == "X5-"
                ):
                    pass
                elif (
                    line_index == 131
                    and token_index == 5
                    and token == "hollbwysig"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 137
                    and token_index == 25
                    and token == "negeseuon"
                    and usas_tag == "Q1/Y2"
                ):
                    pass
                elif (
                    line_index == 151
                    and token_index == 3
                    and token == "math"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 161
                    and token_index == 17
                    and token == "swyddogol"
                    and usas_tag == "A11/A10"
                ):
                    pass
                elif (
                    line_index == 195
                    and token_index == 15
                    and token == "brif"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 196
                    and token_index == 13
                    and token == "gwerthfawr"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 217
                    and token_index == 4
                    and token == "thema"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 223
                    and token_index == 29
                    and token == "Themâu"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 223
                    and token_index == 45
                    and token == "themâu"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 237
                    and token_index == 22
                    and token == "brif"
                    and usas_tag == "A11/A4.2/S7.1"
                ):
                    pass
                elif (
                    line_index == 242
                    and token_index == 8
                    and token == "brif"
                    and usas_tag == "A11/A4.2/S7.1"
                ):
                    pass
                elif (
                    line_index == 248
                    and token_index == 9
                    and token == "arbennig"
                    and usas_tag == "A4.2/A11"
                ):
                    pass
                elif (
                    line_index == 252
                    and token_index == 5
                    and token == "prif"
                    and usas_tag == "A11/S7.1"
                ):
                    pass
                elif (
                    line_index == 253
                    and token_index == 17
                    and token == "cyffredinol"
                    and usas_tag == "A1"
                ):
                    pass
                elif (
                    line_index == 253
                    and token_index == 20
                    and token == "prif"
                    and usas_tag == "A11/S7.1"
                ):
                    pass
                elif (
                    line_index == 255
                    and token_index == 14
                    and token == "hyd"
                    and usas_tag == "T.13"
                ):
                    pass
                elif (
                    line_index == 263
                    and token_index == 7
                    and token == "isaf"
                    and usas_tag == "N.37"
                ):
                    pass
                elif (
                    line_index == 264
                    and token_index == 21
                    and token == "arbennig"
                    and usas_tag == "A4.2/A11"
                ):
                    pass
                elif (
                    line_index == 270
                    and token_index == 6
                    and token == "prif"
                    and usas_tag == "S7.1/A11/A14"
                ):
                    pass
                elif (
                    line_index == 270
                    and token_index == 7
                    and token == "swyddogaeth"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 273
                    and token_index == 3
                    and token == "teipoleg"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 281
                    and token_index == 22
                    and token == "swyddogol"
                    and usas_tag == "A11/S7.1"
                ):
                    pass
                elif (
                    line_index == 282
                    and token_index == 5
                    and token == "Arbennig"
                    and usas_tag == "A4.2/A11"
                ):
                    pass
                elif (
                    line_index == 293
                    and token_index == 22
                    and token == "prif"
                    and usas_tag == "A14/A11"
                ):
                    pass
                elif (
                    line_index == 295
                    and token_index == 9
                    and token == "yrfa"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 295
                    and token_index == 17
                    and token == "gweithiodd"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 299
                    and token_index == 3
                    and token == "arbennig"
                    and usas_tag == "A4.2/A11"
                ):
                    pass
                elif (
                    line_index == 301
                    and token_index == 10
                    and token == "gyflwynydd"
                    and usas_tag == "S2/I3/Q4"
                ):
                    pass
                elif (
                    line_index == 304
                    and token_index == 18
                    and token == "gyrfa"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 308
                    and token_index == 8
                    and token == "dal"
                    and usas_tag == "T.13"
                ):
                    pass
                elif (
                    line_index == 309
                    and token_index == 14
                    and token == "eicon"
                    and usas_tag == "A1.8/A11"
                ):
                    pass
                elif (
                    line_index == 311
                    and token_index == 12
                    and token == "prif"
                    and usas_tag == "A11/S7.1"
                ):
                    pass
                elif (
                    line_index == 311
                    and token_index == 22
                    and token == "prif"
                    and usas_tag == "A11/S7.1"
                ):
                    pass
                elif (
                    line_index == 315
                    and token_index == 35
                    and token == "drobwynt"
                    and usas_tag == "A11/A2.1"
                ):
                    pass
                elif (
                    line_index == 334
                    and token_index == 18
                    and token == "gydweithio"
                    and usas_tag == "I3/S5"
                ):
                    pass
                elif (
                    line_index == 363
                    and token_index == 9
                    and token == "unigrwydd"
                    and usas_tag == "E4-/S5-"
                ):
                    pass
                elif (
                    line_index == 365
                    and token_index == 28
                    and token == "unigrwydd"
                    and usas_tag == "E4-/S5-"
                ):
                    pass
                elif (
                    line_index == 366
                    and token_index == 19
                    and token == "fath"
                    and usas_tag == "A4"
                ):
                    pass
                elif (
                    line_index == 394
                    and token_index == 16
                    and token == "frenhiniaeth"
                    and usas_tag == "S7.1+/S.1F"
                ):
                    pass
                elif (
                    line_index == 406
                    and token_index == 26
                    and token == "enwocaf"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 415
                    and token_index == 3
                    and token == "penyd"
                    and usas_tag == "A1.1.1/E4-"
                ):
                    pass
                elif (
                    line_index == 420
                    and token_index == 24
                    and token == "chwareli"
                    and usas_tag == "I3/W3"
                ):
                    pass
                elif (
                    line_index == 424
                    and token_index == 8
                    and token == "Dirwasgiad"
                    and usas_tag == "E4-/I1-/G1.2-"
                ):
                    pass
                elif (
                    line_index == 434
                    and token_index == 29
                    and token == "statws"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 437
                    and token_index == 9
                    and token == "frenhines"
                    and usas_tag == "S7.1+/S.1F"
                ):
                    pass
                elif (
                    line_index == 444
                    and token_index == 7
                    and token == "seremonïol"
                    and usas_tag == "Q2.2/S7.1/A11+"
                ):
                    pass
                elif (
                    line_index == 448
                    and token_index == 24
                    and token == "statws"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 453
                    and token_index == 15
                    and token == "allweddol"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 460
                    and token_index == 2
                    and token == "gyrfa"
                    and usas_tag == "I3"
                ):
                    pass
                elif (
                    line_index == 470
                    and token_index == 25
                    and token == "hollbwysig"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 476
                    and token_index == 10
                    and token == "bennaf"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 482
                    and token_index == 19
                    and token == "gyfarfod"
                    and usas_tag == "S1.1.3/Q.2"
                ):
                    pass
                elif (
                    line_index == 484
                    and token_index == 0
                    and token == "Cyllidir"
                    and usas_tag == "Q1/Q2/I1"
                ):
                    pass
                elif (
                    line_index == 484
                    and token_index == 4
                    and token == "bennaf"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 484
                    and token_index == 8
                    and token == "Adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 486
                    and token_index == 6
                    and token == "ddyletswydd"
                    and usas_tag == "I3/S7.1"
                ):
                    pass
                elif (
                    line_index == 486
                    and token_index == 29
                    and token == "Newyddion"
                    and usas_tag == "!ERR"
                ):
                    pass
                elif (
                    line_index == 491
                    and token_index == 8
                    and token == "Adran"
                    and usas_tag == "N5.1/I3"
                ):
                    pass
                elif (
                    line_index == 491
                    and token_index == 11
                    and token == "Cyfathrebu"
                    and usas_tag == "Q1/Q2"
                ):
                    pass
                elif (
                    line_index == 495
                    and token_index == 49
                    and token == "sail"
                    and usas_tag == "Q2/X4"
                ):
                    pass
                elif (
                    line_index == 497
                    and token_index == 23
                    and token == "swyddogol"
                    and usas_tag == "A10/A11/Q2"
                ):
                    pass
                elif (
                    line_index == 499
                    and token_index == 5
                    and token == "swyddogol"
                    and usas_tag == "A10/A11/Q2"
                ):
                    pass
                elif (
                    line_index == 502
                    and token_index == 10
                    and token == "seiliedig"
                    and usas_tag == "A11"
                ):
                    pass
                elif (
                    line_index == 513
                    and token_index == 8
                    and token == "Swyddfa"
                    and usas_tag == "H1/I3"
                ):
                    pass
                elif (
                    line_index == 514
                    and token_index == 1
                    and token == "cyfarfod"
                    and usas_tag == "S1.1.3/Q.2"
                ):
                    pass
                elif (
                    line_index == 514
                    and token_index == 7
                    and token == "galwodd"
                    and usas_tag == "Q2.2/Q1"
                ):
                    pass
                elif (
                    line_index == 514
                    and token_index == 19
                    and token == "gyffredinol"
                    and usas_tag == "A1"
                ):
                    pass
                elif (
                    line_index == 519
                    and token_index == 4
                    and token == "Swyddfa"
                    and usas_tag == "H1/I3"
                ):
                    pass
                else:
                    if (
                        line_index == 3
                        and token_index == 10
                        and token == "gweithio"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 18
                        and token_index == 17
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 37
                        and token_index == 16
                        and token == "gweithio"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 38
                        and token_index == 18
                        and token == "gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 57
                        and token_index == 5
                        and token == "gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 62
                        and token_index == 25
                        and token == "swyddi"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 101
                        and token_index == 40
                        and token == "swyddi"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 112
                        and token_index == 15
                        and token == "broses"
                        and usas_tag == "A1"
                    ):
                        usas_tag = "A1.1.1"
                    elif (
                        line_index == 133
                        and token_index == 12
                        and token == "weithio"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 147
                        and token_index == 9
                        and token == "ddefnydd"
                        and usas_tag == "A.1.5.1"
                    ):
                        usas_tag = "A1.5.1"
                    elif (
                        line_index == 159
                        and token_index == 25
                        and token == "ddefnyddio"
                        and usas_tag == "A.1.5.1"
                    ):
                        usas_tag = "A1.5.1"
                    elif (
                        line_index == 191
                        and token_index == 15
                        and token == "broses"
                        and usas_tag == "A1"
                    ):
                        usas_tag = "A1.1.1"
                    elif (
                        line_index == 225
                        and token_index == 7
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 227
                        and token_index == 8
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 248
                        and token_index == 20
                        and token == "cyn-fyfyrwyr"
                        and usas_tag == "T.1.1.1/S2/P1"
                    ):
                        usas_tag = "T1.1.1/S2/P1"
                    elif (
                        line_index == 259
                        and token_index == 21
                        and token == "ddyletswyddau"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 286
                        and token_index == 24
                        and token == "gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 295
                        and token_index == 4
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 326
                        and token_index == 24
                        and token == "eleni"
                        and usas_tag == "T.1.1.2"
                    ):
                        usas_tag = "T1.1.2"
                    elif (
                        line_index == 329
                        and token_index == 4
                        and token == "eleni"
                        and usas_tag == "T.1.1.2"
                    ):
                        usas_tag = "T1.1.2"
                    elif (
                        line_index == 333
                        and token_index == 1
                        and token == "eleni"
                        and usas_tag == "T.1.1.2"
                    ):
                        usas_tag = "T1.1.2"
                    elif (
                        line_index == 335
                        and token_index == 23
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 336
                        and token_index == 0
                        and token == "Gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 337
                        and token_index == 2
                        and token == "gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 341
                        and token_index == 20
                        and token == "waith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 345
                        and token_index == 12
                        and token == "nodweddu"
                        and usas_tag == "A4"
                    ):
                        usas_tag = "A4.1"
                    elif (
                        line_index == 351
                        and token_index == 4
                        and token == "gwaith"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 381
                        and token_index == 0
                        and token == "Geirdarddiad"
                        and usas_tag == "Q2/T.1.1.1"
                    ):
                        usas_tag = "Q2/T1.1.1"
                    elif (
                        line_index == 391
                        and token_index == 43
                        and token == "gyfoes"
                        and usas_tag == "T.1.1.2"
                    ):
                        usas_tag = "T1.1.2"
                    elif (
                        line_index == 397
                        and token_index == 15
                        and token == "weithiodd"
                        and usas_tag == "I3"
                    ):
                        usas_tag = "I3.1"
                    elif (
                        line_index == 511
                        and token_index == 5
                        and token == "croesawyd"
                        and usas_tag == "Q2.2/S.1.2.4"
                    ):
                        usas_tag = "Q2.2/S1.2.4"
                    elif (
                        line_index == 512
                        and token_index == 0
                        and token == "Dywedodd"
                        and usas_tag == "Q.21"
                    ):
                        usas_tag = "Q2.1"
                    elif (
                        line_index == 578
                        and token_index == 3
                        and token == "peiriant"
                        and usas_tag == "A.1.1.1"
                    ):
                        usas_tag = "A1.1.1"

                    tag_groups: list[USASTagGroup] | None = None
                    try:
                        tag_groups = parse_usas_token_group([usas_tag], valid_usas_tags)
                    except ValueError as parsing_value_error:
                        logger.exception(
                            f"Error cannot parse the USAS tag {line_index}/{token_index} (line_index/token_index): "
                            f"{token_data}"
                        )
                        raise parsing_value_error

                    no_tag_groups = False
                    if tag_groups is None:
                        no_tag_groups = True
                    elif len(tag_groups) == 0:
                        no_tag_groups = True

                    if no_tag_groups:
                        raise ValueError(
                            f"Error cannot parse the USAS tag {line_index}/{token_index} (line_index/token_index): "
                            f"{token_data}"
                        )
                    labels.append(tag_groups)
                    label_offsets.append((token_index, token_index + 1))
                text_tokens.append(token)
                pos_tags.append(pos_tag)
                lemmas.append(lemma)
            evaluation_texts.append(
                EvaluationText(
                    tokens=text_tokens,
                    lemmas=lemmas,
                    pos_tags=pos_tags,
                    labels=labels,
                    label_offsets=label_offsets,
                )
            )
    logger.info(f"Ignored Z99 labels, found: {Z99_count:,} label tokens.")
    logger.info("Finished parsing the CorCenCC Welsh dataset")
    return EvaluationDataset(
        texts={EvaluationDatasetName.corcencc_welsh.value: evaluation_texts},
        name=EvaluationDatasetName.corcencc_welsh,
        text_level=TextLevel.sentence,
        usas_tags_to_filter_out=usas_tags_to_filter_out_set,
        dataset_type=DatasetType.gold,
    )


def parse_icc_irish(
    dataset_path: Path,
    usas_mapper_path: Path,
    usas_tags_to_filter_out: list[str] | None,
    predictions: bool,
    gold_evaluation_texts: dict[str, list[EvaluationText]] | None = None,
    post_tagger_function: Callable[
        [list[str], tuple[int, int], list[USASTagGroup]], list[USASTagGroup]
    ]
    | None = None,
) -> EvaluationDataset:
    """
    Prases either the manually annotated dataset or pre-computed predictions
    of the Internation Comparable Corpus (ICC) Irish corpus into the
    Evaluation Dataset format, for easy evaluation of USAS WSD models.

    Pre-computed predictions are expected to be in the same format as the
    manually annotated dataset whereby the `USAS` field should contain the
    USAS tag predictions.

    For the manually annotated dataset: All `PUNCT` tags associated to the
    UPOS field are ignored, as well as Z99 USAS tags, and any other punctuation
    symbols.

    Args:
        dataset_path (Path): Path to the ICC Irish corpus. This should be a Path
            to the folder that contains many TSV files.
        usas_mapper_path (Path): Path to YAML object that maps USAS tags to
            their description.
        usas_tags_to_filter_out (list[str] | None): A list of USAS tags to
            filter out of the complete list of USAS tags defined by the USAS
            mapper. This is used to define what USAS tags should be in the
            evaluation dataset, e.g. Z99 this HOWEVER will not
            mean that the evaluation dataset filters these tags out rather it
            will raise a ValueError if it finds one of these invalid tags.
        predictions (bool): Whether the data that is being parsed are predictions
            or not.
        gold_evaluation_texts (dict[str, list[EvaluationText]] | None) If
            predictions is True then this should contain the manually annotated
            dictionary of evaluation texts, this is so that the predictions can
            skip any labels it will not be evaluated on. Default None.
    Returns:
        EvaluationDataset: The parsed and formatted dataset.
    Raises:
        ValueError: If no underscore is found in an annotated token.
        ValueError: If the token with the tag is an empty token.
        ValueError: If an expected token can be parsed as a USAS tag.
        ValueError: If more than one underscore is found, only one should be found
            which should be between the token and the USAS tag.
        ValueError: If a USAS tag cannot be parsed, this could be due to it not
            being a valid USAS tag.
    """

    logger.info(f"Parsing the ICC Irish dataset found at: {dataset_path}")
    logger.info(f"USAS mapper path: {usas_mapper_path}")
    logger.info(f"USAS tags being filtered out: {usas_tags_to_filter_out}")
    if predictions:
        logger.info("Parsing the predictions")
        if gold_evaluation_texts is None:
            raise ValueError(
                "As the `prediction` argument is `True` the "
                "argument gold_evaluation_texts should contain "
                "a value and not None."
            )
    else:
        logger.info("Parsing the manually annotated dataset")
        if gold_evaluation_texts is not None:
            raise ValueError(
                "As the `prediction` argument is `False` the "
                "argument gold_evaluation_texts should be None."
            )
    usas_tags_to_filter_out_set = set(usas_tags_to_filter_out)
    valid_usas_tags = set(
        load_usas_mapper(usas_mapper_path, usas_tags_to_filter_out_set).keys()
    )
    if predictions:
        valid_usas_tags.add("Z99")
        valid_usas_tags.add("X4")
        valid_usas_tags.add("S7")
        valid_usas_tags.add("S1")
    evaluation_name_texts: dict[str, list[EvaluationText]] = {}

    expected_file_names = set(
        [
            "ICC-GA-WB0-003_Baoismhachnamh.tsv",
            "ICC-GA-WE0-001v2-Gaelscéal.tsv",
            "ICC-GA-WE0-011-comhar-alt-2020-04-101076.tsv",
            "ICC-GA-WF0-001.tsv",
            "ICC-GA-WF0-003.tsv",
            "ICC-GA-WPH-001-the_wire.tsv",
            "ICC-GA-WPH-003-george_orwell.tsv",
            "ICC-GA-WR0-001-foinse-alt-2009-11-18.tsv",
            "ICC-GA-WR0-020-foinse-alt-2013-09-19.tsv",
            "ICC-GA-WR0-021-tuairisc.tsv",
        ]
    )
    Z99_count = 0
    for data_file in dataset_path.iterdir():
        if data_file.name not in expected_file_names:
            raise ValueError(
                f"The file: {data_file} is not one of the "
                "expected files for this dataset. Expected file "
                "names for this dataset are the following: "
                f"{expected_file_names}"
            )
        logging.debug(f"Processing file: {data_file.name}")
        with data_file.open("r", encoding="utf-8", newline="") as tsv_file:
            text_tokens: list[str] = []
            labels: list[list[USASTagGroup]] = []
            label_offsets: list[tuple[int, int]] = []

            tsv_reader = csv.DictReader(
                tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE
            )

            fields = tsv_reader.fieldnames
            logging.debug(f"Fields: {fields}")

            gold_label_offsets_set: set[tuple[int, int]] | None = None
            if predictions:
                a_gold_evaluation_texts = gold_evaluation_texts[data_file.stem]
                if len(a_gold_evaluation_texts) != 1:
                    raise ValueError(
                        "The manually annotated data contains more "
                        "text files for this data file "
                        f"{data_file.stem}"
                    )
                a_gold_evaluation_texts = a_gold_evaluation_texts[0]
                gold_label_offsets = a_gold_evaluation_texts.label_offsets
                gold_label_offsets_set = set(gold_label_offsets)
                if len(gold_label_offsets) != len(gold_label_offsets_set):
                    raise ValueError(
                        "The manual annotation labels offsets overlap "
                        "with each other which we currently do not "
                        "support."
                    )

            for row_index, row in enumerate(tsv_reader):
                logging.debug(f"Processing row index: {row_index}")

                token = row["TOKEN"]

                if not token:
                    raise ValueError(
                        "The token is empty which should not be the case: "
                        f"{row_index} (row index): `{row}`"
                    )

                token_is_a_tag = True
                try:
                    parse_usas_token_group([token], valid_usas_tags)
                except ValueError:
                    token_is_a_tag = False
                if token_is_a_tag:
                    raise ValueError(
                        f"Error expected token is a tag: {row_index} (row index): "
                        f"`{row}`"
                    )

                tag_groups: list[USASTagGroup] | None = None

                UPOS_value = row.get("UPOS", "")
                USAS_tag = row["USAS"]

                if UPOS_value == "PUNCT" and not predictions:
                    pass
                elif USAS_tag == "Z99" and not predictions:
                    Z99_count += 1
                elif token == "." and not predictions:
                    pass
                elif token == "!" and not predictions:
                    pass
                elif token == "-" and not predictions:
                    pass
                elif token == "," and not predictions:
                    pass
                elif token == "(" and not predictions:
                    pass
                elif token == ")" and not predictions:
                    pass
                elif token == "'" and not predictions:
                    pass
                elif token == ";" and not predictions:
                    pass
                elif token == "…" and not predictions:
                    pass
                elif token == "—" and not predictions:
                    pass
                elif token == '"' and not predictions:
                    pass
                elif token == "?" and not predictions:
                    pass
                elif (
                    data_file.name == "ICC-GA-WPH-003-george_orwell.tsv"
                    and row_index == 112
                    and USAS_tag == "S7"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WB0-003_Baoismhachnamh.tsv"
                    and row_index == 193
                    and USAS_tag == "X4"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WB0-003_Baoismhachnamh.tsv"
                    and row_index == 905
                    and USAS_tag == "X4"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WB0-003_Baoismhachnamh.tsv"
                    and row_index == 1910
                    and USAS_tag == "G5"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WPH-001-the_wire.tsv"
                    and row_index == 108
                    and USAS_tag == "G1.2/S7"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WPH-001-the_wire.tsv"
                    and row_index == 110
                    and USAS_tag == "X5"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-003.tsv"
                    and row_index == 2470
                    and USAS_tag == "E4+"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WE0-001v2-Gaelscéal.tsv"
                    and row_index == 13
                    and USAS_tag == "A11 "
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WE0-001v2-Gaelscéal.tsv"
                    and row_index == 74
                    and USAS_tag == "A11 "
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-001.tsv"
                    and row_index == 306
                    and USAS_tag == "N2.7+"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-001.tsv"
                    and row_index == 405
                    and USAS_tag == "X5"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-001.tsv"
                    and row_index == 596
                    and USAS_tag == "S1 L2"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-001.tsv"
                    and row_index == 1041
                    and USAS_tag == "B6"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WF0-001.tsv"
                    and row_index == 1275
                    and USAS_tag == "E4-"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WR0-020-foinse-alt-2013-09-19.tsv"
                    and row_index == 215
                    and USAS_tag == "E4+"
                    and not predictions
                ):
                    pass
                elif (
                    data_file.name == "ICC-GA-WR0-020-foinse-alt-2013-09-19.tsv"
                    and row_index == 712
                    and USAS_tag == "G5"
                    and not predictions
                ):
                    pass
                elif (
                    USAS_tag == "S2mfc S1"
                    and row_index == 663
                    and data_file.name == "ICC-GA-WR0-001-foinse-alt-2009-11-18.tsv"
                    and not predictions
                ):
                    pass
                elif (
                    USAS_tag == "S2mfc S1"
                    and row_index == 1549
                    and data_file.name == "ICC-GA-WR0-001-foinse-alt-2009-11-18.tsv"
                    and not predictions
                ):
                    pass
                elif (
                    predictions
                    and (row_index, row_index + 1) not in gold_label_offsets_set
                ):
                    pass
                else:
                    if predictions and USAS_tag == "Q4.3 /T1 X7+ F1 K4 O2 P1 Y2":
                        USAS_tag = "Q4.3/T1 X7+ F1 K4 O2 P1 Y2"
                    if predictions and USAS_tag == "Q4.3 /T1 X7+ K4 P1 F1 Y2 O2":
                        USAS_tag = "Q4.3/T1 X7+ K4 P1 F1 Y2 O2"
                    if predictions and USAS_tag == "Q4.3 /T1 X7+ P1 Y2 O2 K4 F1":
                        USAS_tag = "Q4.3/T1 X7+ P1 Y2 O2 K4 F1"
                    if (
                        data_file.name == "ICC-GA-WF0-003.tsv"
                        and row_index == 626
                        and USAS_tag == "A.1.1.1"
                        and not predictions
                    ):
                        USAS_tag = "A1.1.1"
                    if (
                        data_file.name == "ICC-GA-WR0-020-foinse-alt-2013-09-19.tsv"
                        and row_index == 2049
                        and USAS_tag == "A.1.1.1"
                        and not predictions
                    ):
                        USAS_tag = "A1.1.1"
                    try:
                        USAS_tag = USAS_tag.split()
                        tag_groups = parse_usas_token_group(
                            USAS_tag, valid_usas_tags=valid_usas_tags
                        )
                    except ValueError as parsing_value_error:
                        logger.exception(
                            f"Error cannot parse the USAS tag {row_index} (row index): {row} (file name): {data_file.stem}"
                        )
                        raise parsing_value_error

                    no_tag_groups = False
                    if tag_groups is None:
                        no_tag_groups = True
                    elif len(tag_groups) == 0:
                        no_tag_groups = True

                    if no_tag_groups:
                        raise ValueError(
                            "Error cannot parse the USAS tag "
                            f"{row_index} (row index): {row}"
                        )

                    labels.append(tag_groups)
                    label_offsets.append((row_index, row_index + 1))
                text_tokens.append(token)

            if predictions and post_tagger_function is not None:
                tmp_labels = []
                tmp_label_offsets = []
                for label, label_offset in zip(labels, label_offsets):
                    new_tags = post_tagger_function(text_tokens, label_offset, label)
                    tmp_labels.append(new_tags)
                    tmp_label_offsets.append(label_offset)
                labels = tmp_labels
                label_offsets = tmp_label_offsets
                # tmp_labels = []
                # tmp_label_offsets = []
                # for label, label_offset in zip(labels, label_offsets):
                #    if len(label) == 1 and label[0].tags[0].tag.lower() == "z99":
                #        new_tags = post_tagger_function(text_tokens, label_offset, ["Z99"])
                #        tmp_labels.append(parse_usas_token_group(new_tags, valid_usas_tags))
                #    else:
                #        tmp_labels.append(label)
                #    tmp_label_offsets.append(label_offset)
                # labels = tmp_labels
                # label_offsets = tmp_label_offsets

            evaluation_name_texts[data_file.stem] = [
                EvaluationText(
                    tokens=text_tokens, labels=labels, label_offsets=label_offsets
                )
            ]
    logger.info(f"Ignored Z99 labels, found: {Z99_count:,} label tokens.")
    logger.info("Finished parsing the ICC Irish dataset")

    dataset_type = DatasetType.gold
    if predictions:
        dataset_type = DatasetType.predictions
    return EvaluationDataset(
        texts=evaluation_name_texts,
        name=EvaluationDatasetName.icc_irish,
        text_level=TextLevel.paragraph,
        usas_tags_to_filter_out=usas_tags_to_filter_out_set,
        dataset_type=dataset_type,
    )
