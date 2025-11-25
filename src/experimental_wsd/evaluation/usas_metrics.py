import logging

from experimental_wsd.evaluation.usas_datasets import EvaluationDataset, EvaluationText

logger = logging.getLogger(__name__)


def top_n_accuracy(
    gold_dataset: EvaluationDataset,
    prediction_dataset: EvaluationDataset,
    n: int,
    strict: bool,
) -> float:
    if gold_dataset.name != prediction_dataset.name:
        raise ValueError(
            "The Gold and Prediction datasets must come from the "
            "same original dataset which is determined by the name "
            f"attribute, Gold name: {gold_dataset.name} and "
            f"Prediction name: {prediction_dataset.name}"
        )
    gold_dataset_statistics = gold_dataset.get_statistics()
    prediction_dataset_statistics = prediction_dataset.get_statistics()

    if (
        gold_dataset_statistics["number_texts"]
        != prediction_dataset_statistics["number_texts"]
    ):
        raise ValueError(
            "The Gold and Prediction datasets have a different "
            "number of texts which should not be the case. "
            "Number of Gold and Predictions texts: "
            f"{gold_dataset_statistics['number_texts']:,} "
            f"{prediction_dataset_statistics['number_texts']:,} "
        )

    if (
        gold_dataset_statistics["total_token_count"]
        != prediction_dataset_statistics["total_token_count"]
    ):
        raise ValueError(
            "The Gold and Prediction datasets have a different "
            "number of tokens which should not be the case. "
            "Number of Gold and Predictions tokens: "
            f"{gold_dataset_statistics['total_token_count']:,} "
            f"{prediction_dataset_statistics['total_token_count']:,} "
        )

    if len(gold_dataset.texts) != len(prediction_dataset.texts):
        raise ValueError(
            "The number of evaluation texts is different between "
            f"gold: {len(gold_dataset.texts)} and "
            f"predictions: {len(prediction_dataset.texts)}"
        )
    gold_text_names = set(gold_dataset.texts.keys())
    prediction_text_names = set(prediction_dataset.texts.keys())
    if gold_text_names != prediction_text_names:
        raise ValueError(
            "The names of the evaluation texts are not the same "
            f"between gold: {gold_text_names} and "
            f"predictions: {prediction_text_names}"
        )

    evaluation_text_index = 0
    tp = 0
    fp = 0
    for evaluation_text_name in gold_text_names:
        for gold_evaluation_text, prediction_evaluation_text in zip(
            gold_dataset.texts[evaluation_text_name],
            prediction_dataset.texts[evaluation_text_name],
        ):
            if gold_evaluation_text.tokens != prediction_evaluation_text.tokens:
                raise ValueError(
                    "The Gold and Prediction tokens are different for "
                    f"a given text. For text name: {evaluation_text_name} "
                    f"text index: {evaluation_text_index}"
                    "Gold text tokens: "
                    f"{gold_evaluation_text.tokens} and Prediction text "
                    f"tokens {prediction_evaluation_text.tokens} for "
                )

            if (
                gold_evaluation_text.label_offsets
                != prediction_evaluation_text.label_offsets
            ):
                raise ValueError(
                    "The Gold and Prediction label offsets are different for "
                    f"a given text. For text name: {evaluation_text_name} "
                    f"text index: {evaluation_text_index}"
                    "Gold label offsets: "
                    f"{gold_evaluation_text.label_offsets} and Prediction label "
                    f"offsets {prediction_evaluation_text.label_offsets} for "
                )

            if len(gold_evaluation_text.labels) != len(
                prediction_evaluation_text.labels
            ):
                raise ValueError(
                    "The number of Gold and Prediction label groups are different for "
                    "a given text. For"
                    f"text name: {evaluation_text_name} "
                    f"text index: {evaluation_text_index}"
                    "Number of Gold label groups: "
                    f"{len(gold_evaluation_text.labels)} and number of Prediction "
                    f"label groups {len(prediction_evaluation_text.labels)} for "
                )

            for gold_label_groups, prediction_label_groups in zip(
                gold_evaluation_text.labels, prediction_evaluation_text.labels
            ):
                if len(gold_label_groups) != 1:
                    raise ValueError(
                        "The number of gold label groups we assume "
                        "is currently 1 as this is what we support "
                        f"evaluation text name: {evaluation_text_name} "
                        f"content of the text: {gold_evaluation_text} "
                        f"evaluation text index: {evaluation_text_index}"
                    )
                top_n_prediction_label_groups = prediction_label_groups[:n]
                if n == -1:
                    top_n_prediction_label_groups = prediction_label_groups
                if not strict:
                    gold_label_set = EvaluationText.get_label_set(gold_label_groups)
                    top_n_label_set = EvaluationText.get_label_set(
                        top_n_prediction_label_groups
                    )

                    correct_label_predictions = gold_label_set.intersection(
                        top_n_label_set
                    )
                    if correct_label_predictions:
                        tp += 1
                    else:
                        fp += 1
                else:
                    gold_label_group = gold_label_groups[0]
                    correct = False
                    for prediction_label_group in top_n_prediction_label_groups:
                        if gold_label_group.tags_equal(prediction_label_group):
                            correct = True
                            break
                    if correct:
                        tp += 1
                    else:
                        # print("GOLD:")
                        # gold_pred_tags = "/".join([tag.tag for tag in gold_label_group.tags])
                        # print(f"`{gold_pred_tags}`")
                        # print("PREDICTIONS:")
                        # for prediction_label_group in top_n_prediction_label_groups:
                        #    pred_tags = "/".join([tag.tag for tag in prediction_label_group.tags])
                        #    print(f"`{pred_tags}`")
                        # print()
                        fp += 1

            evaluation_text_index += 1

    if tp == 0 and fp == 0:
        return 0.0
    else:
        return tp / (tp + fp)
