import os
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath, computed_field, field_validator


class WSDDataDirectory(BaseModel):
    """
    Given the top level directory it computes the instance attributes
    `data` and `gold` which are file paths to the data and gold datasets for the
    given top level directory datasets.

    This class performs a validation step to ensure that the top level directory
    only contains two files, of which these two files should be:
    - XML file that contains the data.
    - txt file that contains the WSD gold labels.

    Instance Attributes:

        directory (DirectoryPath): The top level directory containing the two
            files that represent this dataset.
        data (Path): The data file path in XML format for this dataset,
            contains all of the data except the WSD gold labels, it should
            contain per token a reference to the gold label.
        gold (Path): The file path to the txt file containing the WSD gold labels.
    """

    directory: DirectoryPath

    @field_validator("directory", mode="after")
    def validate_data_directory(directory: DirectoryPath) -> DirectoryPath:
        file_types_counter = Counter()
        for _file in directory.iterdir():
            file_types_counter.update([_file.suffix.lower()])

        duplicates_file_types = set()
        for file_type, count in file_types_counter.items():
            if count > 1:
                duplicates_file_types.add(file_type)

        duplicate_error_string = (
            f"Found files with the same file type within: {directory} "
            "this should not be the case. File types associated with more than "
            f"one file: {duplicates_file_types}."
        )
        if duplicates_file_types:
            raise ValueError(duplicate_error_string)

        if len(file_types_counter) != 2:
            raise ValueError(
                f"The number of files in the directory {directory} "
                "should only be 2, 1 txt file and 1 XML file. "
                f"File types found: {file_types_counter}"
            )
        if ".xml" not in file_types_counter:
            raise ValueError(
                f"The number of files in the directory {directory} "
                "should only be 2, 1 txt file and 1 XML file. "
                f"File types found: {file_types_counter}"
            )
        if ".txt" not in file_types_counter:
            raise ValueError(
                f"The number of files in the directory {directory} "
                "should only be 2, 1 txt file and 1 XML file. "
                f"File types found: {file_types_counter}"
            )
        return directory

    @computed_field
    @property
    def data(self) -> FilePath:
        for _file in self.directory.iterdir():
            if ".xml" in _file.suffix.lower():
                return _file

    @computed_field
    @property
    def gold(self) -> FilePath:
        for _file in self.directory.iterdir():
            if ".txt" in _file.suffix.lower():
                return _file


class MaruEnglishConfig(BaseModel):
    """
    Given the top level directory for the English Hard Maru et al. 2022 WSD
    dataset, the datasets created within this work are found.

    To Note

    Instance attributes:

        data_directory (WSDDataDirectory): The top level directory for the
            English Hard Maru et al. 2022 WSD dataset.
        fortitude (WSDDataDirectory): The 42D dataset, found using `data_directory`.
            A new test set taken from the BNC whereby the ground truth does
            not occur in SemCor and the first sense within WordNet is not the
            sense of the target term. The samples of texts come from 42
            different domains according to the domains
            defined in BabelNet version 4.
        all_amended (WSDDataDirectory): The ALL amended dataset, found using `data_directory`.
            This is the corrected version of the Raganato et al 2017 ALL dataset.
        hard_en (WSDDataDirectory): The hard en dataset, found using `data_directory`.
            The samples from All, S10, and 42D that none of the current 7 state
            of the art systems could get correct.
        soft_en (WSDDataDirectory): The soft en dataset, found using `data_directory`.
            The samples from All, S10, and 42D that at least 1 of the 7 current
            state of the art systems could get correct.
        s_10 (WSDDataDirectory): The S10 amended dataset, found using `data_directory`.
            Corrected version of SemEval 2010 task 17.
    """

    data_directory: DirectoryPath

    @computed_field
    @property
    def fortitude(self) -> WSDDataDirectory:
        return WSDDataDirectory(directory=Path(self.data_directory, "42D"))

    @computed_field
    @property
    def all_amended(self) -> WSDDataDirectory:
        return WSDDataDirectory(directory=Path(self.data_directory, "ALLamended"))

    @computed_field
    @property
    def hard_en(self) -> WSDDataDirectory:
        return WSDDataDirectory(directory=Path(self.data_directory, "hardEN"))

    @computed_field
    @property
    def soft_en(self) -> WSDDataDirectory:
        return WSDDataDirectory(directory=Path(self.data_directory, "softEN"))

    @computed_field
    @property
    def s_10(self) -> WSDDataDirectory:
        return WSDDataDirectory(directory=Path(self.data_directory, "S10amended"))


class RaganatoEnglishConfig(BaseModel):
    """
    Given the top level directory for the English Raganato et al. 2017 WSD
    dataset, the datasets used within this work are found.

    Instance attributes:

        data_directory (Path): The top level directory for the
            English Raganato et al. 2017 WSD dataset.
        all (Path): The ALL dataset. Contains all of the evaluation datasets.
        semeval_2007 (WSDDataDirectory): Evaluation dataset. SemEval 2007 dataset. This is
            also typically used for validation.
        semeval_2013 (WSDDataDirectory): Evaluation dataset. SemEval 2013 dataset. This is
            typically only used for testing.
        semeval_2015 (WSDDataDirectory): Evaluation dataset. SemEval 2015 dataset. This is
            typically only used for testing.
        senseval_2 (WSDDataDirectory): Evaluation dataset. SensEval 2 dataset. This is
            typically only used for testing.
        senseval_3 (WSDDataDirectory): Evaluation dataset. SensEval 3 dataset. This is
            typically only used for testing.
        semcor (WSDDataDirectory): Training dataset. SemCor dataset. This is
            typically only used for training.
        semcor_omsti (WSDDataDirectory): Training dataset. SemCor and OMSTI datasets.
            This is typically only used for training.
    """

    data_directory: DirectoryPath

    @computed_field
    @property
    def all(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "ALL")
        )

    @computed_field
    @property
    def semeval_2007(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "semeval2007")
        )

    @computed_field
    @property
    def semeval_2013(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "semeval2013")
        )

    @computed_field
    @property
    def semeval_2015(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "semeval2015")
        )

    @computed_field
    @property
    def senseval_2(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "senseval2")
        )

    @computed_field
    @property
    def senseval_3(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Evaluation_Datasets", "senseval3")
        )

    @computed_field
    @property
    def semcor(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Training_Corpora", "SemCor")
        )

    @computed_field
    @property
    def semcor_omsti(self) -> WSDDataDirectory:
        return WSDDataDirectory(
            directory=Path(self.data_directory, "Training_Corpora", "SemCor+OMSTI")
        )


MaruEnglish: MaruEnglishConfig | None = None
RaganatoEnglish: RaganatoEnglishConfig | None = None

if os.environ.get("ENGLISH_MARU_HARD"):
    MaruEnglish = MaruEnglishConfig(data_directory=os.environ.get("ENGLISH_MARU_HARD"))

if os.environ.get("ENGLISH_RAGANATO"):
    RaganatoEnglish = RaganatoEnglishConfig(
        data_directory=os.environ.get("ENGLISH_RAGANATO")
    )

# Used to set where the processed data should be stored.
DATA_PROCESSING_DIR = Path(
    os.environ.get(
        "EXPERIMENTAL_WSD_DATA_PROCESSING_DIR",
        Path(Path.home(), ".cache", "experimental_wsd"),
    )
)
DATA_PROCESSING_DIR.mkdir(exist_ok=True)
