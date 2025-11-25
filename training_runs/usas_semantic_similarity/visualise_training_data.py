import logging
from collections import defaultdict
from enum import Enum
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import datasets  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import typer  # noqa: E402
from matplotlib import pyplot  # noqa: E402
from typing_extensions import Annotated  # noqa: E402

from experimental_wsd.config import (  # noqa: E402
    DATA_PROCESSING_DIR,
    MosaicoCoreUSAS,
    USASMapper,
)
from experimental_wsd.data_processing import processed_usas_utils  # noqa: E402
from experimental_wsd.data_processing.shared_token_utils import (  # noqa: E402
    remove_duplicate_list_of_list_entries_while_maintaining_order,
)

logger = logging.getLogger(__name__)


def get_top_level_usas_tags(all_usas_tags: list[str]) -> list[str]:
    top_level_tags = set()
    for tag in all_usas_tags:
        top_level_tags.add(tag[0])
    return sorted(top_level_tags)


def aggregate_label_statistics(
    label_statistics: dict[str, int | float],
) -> dict[str, int | float]:
    aggregated_statistics = {}
    for label, statistic in label_statistics.items():
        top_level_label = label[0]
        top_level_statistic = statistic + aggregated_statistics.get(top_level_label, 0)
        aggregated_statistics[top_level_label] = top_level_statistic
    total_statistics = sum(label_statistics.values())
    aggregated_total_statistics = sum(aggregated_statistics.values())
    if total_statistics != aggregated_total_statistics:
        raise ValueError(
            f"The total aggregated statistics: {aggregated_total_statistics:,.3f} "
            "should match the original non-aggregated statistics: "
            f"{total_statistics:,.3f}"
        )
    return aggregated_statistics


class DatasetName(str, Enum):
    english_mosaico = "english_mosaico"


def get_document_statistics(hf_dataset: datasets.Dataset) -> dict[str, int]:
    document_ids = set()
    sentence_count = 0
    token_count = 0
    labelled_tokens = 0
    total_labels = 0
    for sample in hf_dataset:
        document_ids.add(sample["id"]["document_id"])
        sentence_count += 1
        token_count += len(sample["tokens"])
        labelled_tokens += len(sample["usas"])
        total_labels += sum([len(label) for label in sample["usas"]])
    document_statistics = {
        "No. Documents": len(document_ids),
        "No. Sentences": sentence_count,
        "No. Tokens": token_count,
        "No. Labelled Tokens": labelled_tokens,
        "No. Labels": total_labels,
        "Avg. Labels per Token": round(total_labels / labelled_tokens, 2),
    }
    return document_statistics


def main(
    dataset_name: Annotated[
        DatasetName, typer.Argument(help="The name of the dataset to visualise")
    ],
    number_cpus: Annotated[
        int, typer.Argument(help="Number of CPUs to use process the data")
    ],
    dataset_statistics_output_file: Annotated[
        Path,
        typer.Argument(
            writable=True,
            readable=False,
            help="File path to write the dataset statistics to in Latex format.",
        ),
    ],
    heatmap_output_file: Annotated[
        Path,
        typer.Argument(
            writable=True, readable=False, help="File path to write the heatmap to."
        ),
    ],
    filter_out_labels: Annotated[
        Optional[list[str]],
        typer.Option("-f", help="The semantic labels to filter out of the dataset"),
    ] = None,
    no_heatmap_titles: Annotated[
        bool, typer.Option("-h", help="No titles on the heatmap")
    ] = False,
) -> None:
    logger.info(f"Dataset name: {dataset_name.value}")
    logger.info(f"Number CPUs to process the data with: {number_cpus}")
    if filter_out_labels:
        logger.info("Filtering out the following labels:")
        for label in filter_out_labels:
            logger.info(f"{label}")
    else:
        logger.info("Not filtering out any labels.")
    logger.info(
        f"Writing the general dataset statistics in Latex format too: {dataset_statistics_output_file}"
    )
    logger.info(
        f"Writing the heatmap of the label distribution of the training data too: {heatmap_output_file}"
    )
    logger.info(f"Titles on the heatmap: {not no_heatmap_titles}")

    usas_tag_to_description_mapper = processed_usas_utils.load_usas_mapper(
        USASMapper, filter_out_labels
    )
    if dataset_name == DatasetName.english_mosaico:
        train_data_file_paths = MosaicoCoreUSAS.train
        validation_file_path = MosaicoCoreUSAS.validation
        test_file_path = MosaicoCoreUSAS.test
        all_file_paths = [*train_data_file_paths, validation_file_path, test_file_path]
        overwrite_data = False
        process_file_paths_arguments = [
            (
                file_path,
                DATA_PROCESSING_DIR,
                f"variable_mosaico_usas_{index}",
                overwrite_data,
            )
            for index, file_path in enumerate(all_file_paths)
        ]

        processed_file_paths: list[Path] | None = None
        with Pool(number_cpus) as pool:
            processed_file_paths = pool.starmap(
                processed_usas_utils.process_file,
                process_file_paths_arguments,
                chunksize=1,
            )
        processed_file_paths_str = [
            str(file_path.resolve()) for file_path in processed_file_paths
        ]
        # training_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[:8])
        training_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[0]
        )
        # validation_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[8])
        validation_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[8]
        )["train"].take(20000)
        # test_dataset = datasets.load_dataset("json", data_files=processed_file_paths_str[9])
        test_dataset = datasets.load_dataset(
            "json", data_files=processed_file_paths_str[9]
        )["train"].take(20000)

        usas_dataset = datasets.DatasetDict(
            {
                "train": training_dataset["train"],
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )
        usas_dataset_de_duplicated = usas_dataset.map(
            remove_duplicate_list_of_list_entries_while_maintaining_order,
            fn_kwargs={"key": "usas", "tags_to_filter_out": filter_out_labels},
            num_proc=number_cpus,
        )
        dataset_split_statistics = defaultdict(dict)
        for dataset_name, dataset in usas_dataset_de_duplicated.items():
            for field_name, field_value in get_document_statistics(dataset).items():
                dataset_split_statistics[field_name][dataset_name] = field_value
        logger.info(
            f"Writing general dataset statistics to file: {dataset_statistics_output_file}"
        )
        pd.DataFrame(dataset_split_statistics).to_latex(dataset_statistics_output_file)
        logger.info("Finished writing statistics to file.")

        training_label_statistics = processed_usas_utils.get_usas_label_statistics(
            usas_dataset_de_duplicated["train"], usas_tag_to_description_mapper
        )
        for label, count in training_label_statistics.items():
            if count == 0:
                logger.info(f"Label that does not occur in the training data: {label}")

        training_log_inverse_label_statistics = (
            processed_usas_utils.usas_inverse_label_statistics(
                training_label_statistics, log_scaled=2
            )
        )
        training_inverse_label_statistics = (
            processed_usas_utils.usas_inverse_label_statistics(
                training_label_statistics, log_scaled=None
            )
        )

        label_statistics = {"Label": [], "Relative Frequency": [], "Distribution": []}
        aggregated_distribution_to_statistic = {
            "Original": aggregate_label_statistics(training_label_statistics),
            "Inverse": aggregate_label_statistics(training_inverse_label_statistics),
            "Log Inverse": aggregate_label_statistics(
                training_log_inverse_label_statistics
            ),
        }
        top_level_usas_tags = get_top_level_usas_tags(
            list(usas_tag_to_description_mapper.keys())
        )
        for label in top_level_usas_tags:
            for (
                distribution_name,
                distribution,
            ) in aggregated_distribution_to_statistic.items():
                label_statistics["Label"].append(label)
                label_statistics["Distribution"].append(distribution_name)
                distribution_total = sum(distribution.values())
                label_count = distribution.get(label, 0.0)
                normalised_count = label_count / distribution_total
                label_statistics["Relative Frequency"].append(normalised_count)

        # label_statistics_df = pd.DataFrame(label_statistics)

        # so.Plot(label_statistics_df, y="Label", x="Relative Frequency", color="Distribution").add(so.Bar(), so.Stack()).save("test.svg")

        original_counts = aggregate_label_statistics(training_label_statistics)
        label_statistics = {"Label": [], "Frequency": []}

        for label in top_level_usas_tags:
            label_statistics["Label"].append(label)
            label_statistics["Frequency"].append(original_counts[label])
        # label_statistics_df = pd.DataFrame(label_statistics)
        # so.Plot(label_statistics_df, y="Label", x="Frequency").add(so.Bar()).save("original.svg")

        label_statistics = {
            "Label": [],
            "Original (Count)": [],
            "Inverse": [],
            "Log Inverse": [],
        }
        heatmap_statistics = defaultdict(dict)

        distribution_to_statistic = {
            "Original": training_label_statistics,
            "Inverse": training_inverse_label_statistics,
            "Log Inverse": training_log_inverse_label_statistics,
        }

        dist_values = defaultdict(lambda: 0)
        for label in usas_tag_to_description_mapper.keys():
            label_statistics["Label"].append(label)
            label_distribution_values = 0
            for distribution_name, distribution in distribution_to_statistic.items():
                distribution_count = distribution.get(label, 0.0)
                percentage = distribution_count / sum(distribution.values())
                dist_values[distribution_name] += percentage
                percentage_count = f"{percentage:.4f} ({distribution_count:,})"
                if distribution_name == "Original":
                    label_statistics["Original (Count)"].append(percentage_count)
                else:
                    label_statistics[distribution_name].append(percentage)
                heatmap_statistics[label][distribution_name] = percentage
                label_distribution_values += percentage
            heatmap_statistics[label]["Average"] = label_distribution_values / len(
                distribution_to_statistic
            )

        # label_statistics_df = pd.DataFrame(label_statistics)
        # label_statistics_df.to_latex("label_statistics.tex")

        # Remove outliers from the heatmap colours
        percentage_values = sorted(
            [
                value
                for _, distribution_values in heatmap_statistics.items()
                for _, value in distribution_values.items()
            ]
        )
        fifth_percentile_index = int((len(percentage_values) / 100) * 5)
        percentile_95 = percentage_values[
            len(percentage_values) - 1 - fifth_percentile_index
        ]
        logger.info(
            f"Heatmap relative percentage range: {percentage_values[0]:.3f} - {percentile_95:.3f}"
        )
        normalised_heatmap_statistics = defaultdict(dict)
        for label, distribution_value in heatmap_statistics.items():
            for distribution_name, value in distribution_value.items():
                normalised_value = min(value, percentile_95)
                normalised_heatmap_statistics[label][distribution_name] = (
                    normalised_value
                )

        distribution_stat = defaultdict(lambda: 0)
        for _, distribution_name_stat in heatmap_statistics.items():
            for distribution_name, stat in distribution_name_stat.items():
                distribution_stat[distribution_name] += stat
        logger.info(f"Distribution statistics for heatmap: {distribution_stat}")
        logger.info(f"Distribution statistics for heatmap: {dist_values}")

        page_dims = (4.5, 24)
        fig, ax = pyplot.subplots(figsize=page_dims)
        sns.heatmap(
            pd.DataFrame(normalised_heatmap_statistics).T,
            ax=ax,
            annot=pd.DataFrame(heatmap_statistics).T,
            annot_kws={"fontsize": "xx-small"},
        )  # , cbar_kws={"ticks":[0,0.2,0.4,0.6,0.8,1.0], "extend": "max"})
        ax.tick_params(labelsize="xx-small")
        if not no_heatmap_titles:
            ax.set_ylabel("USAS Labels", fontsize="small")
            ax.set_xlabel("Distribution", fontsize="small")
            ax.set_title(
                "Probability of a USAS label within the training dataset\nfor each distribution",
                fontsize="medium",
            )
        fig.tight_layout()
        fig.savefig(heatmap_output_file)
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    typer.run(main)
