import os
from pathlib import Path

from pydantic import BaseModel, DirectoryPath, computed_field


class MaruEnglishConfig(BaseModel):
    """
    Given the top level directory for the English Hard Maru et al. 2022 WSD
    dataset, the datasets created within this work are found.

    Instance attributes:

        data_directory (Path): The top level directory for the
            English Hard Maru et al. 2022 WSD dataset.
        fortitude (Path): The 42D dataset, found using `data_directory`.
            A new test set taken from the BNC whereby the ground truth does
            not occur in SemCor and the first sense within WordNet is not the
            sense of the target term. The samples of texts come from 42
            different domains according to the domains
            defined in BabelNet version 4.
        all_amended (Path): The ALL amended dataset, found using `data_directory`.
            This is the corrected version of the Raganato et al 2017 ALL dataset.
        hard_en (Path): The hard en dataset, found using `data_directory`.
            The samples from All, S10, and 42D that none of the current 7 state
            of the art systems could get correct.
        soft_en (Path): The soft en dataset, found using `data_directory`.
            The samples from All, S10, and 42D that at least 1 of the 7 current
            state of the art systems could get correct.
        s_10 (Path): The S10 amended dataset, found using `data_directory`.
            Corrected version of SemEval 2010 task 17.
    """

    data_directory: DirectoryPath

    @computed_field
    @property
    def fortitude(self) -> DirectoryPath:
        return Path(self.data_directory, "42D")

    @computed_field
    @property
    def all_amended(self) -> DirectoryPath:
        return Path(self.data_directory, "ALLamended")

    @computed_field
    @property
    def hard_en(self) -> DirectoryPath:
        return Path(self.data_directory, "hardEN")

    @computed_field
    @property
    def soft_en(self) -> DirectoryPath:
        return Path(self.data_directory, "softEN")

    @computed_field
    @property
    def s_10(self) -> DirectoryPath:
        return Path(self.data_directory, "S10amended")


class RaganatoEnglishConfig(BaseModel):
    """
    Given the top level directory for the English Raganato et al. 2017 WSD
    dataset, the datasets used within this work are found.

    Instance attributes:

        data_directory (Path): The top level directory for the
            English Raganato et al. 2017 WSD dataset.
        all (Path): The ALL dataset. Contains all of the evaluation datasets.
        semeval_2007 (Path): Evaluation dataset. SemEval 2007 dataset. This is
            also typically used for validation.
        semeval_2013 (Path): Evaluation dataset. SemEval 2013 dataset. This is
            typically only used for testing.
        semeval_2015 (Path): Evaluation dataset. SemEval 2015 dataset. This is
            typically only used for testing.
        senseval_2 (Path): Evaluation dataset. SensEval 2 dataset. This is
            typically only used for testing.
        senseval_3 (Path): Evaluation dataset. SensEval 3 dataset. This is
            typically only used for testing.
        semcor (Path): Training dataset. SemCor dataset. This is
            typically only used for training.
        semcor_omsti (Path): Training dataset. SemCor and OMSTI datasets.
            This is typically only used for training.
    """

    data_directory: DirectoryPath

    @computed_field
    @property
    def all(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "ALL")

    @computed_field
    @property
    def semeval_2007(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "semeval2007")

    @computed_field
    @property
    def semeval_2013(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "semeval2013")

    @computed_field
    @property
    def semeval_2015(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "semeval2015")

    @computed_field
    @property
    def senseval_2(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "senseval2")

    @computed_field
    @property
    def senseval_3(self) -> DirectoryPath:
        return Path(self.data_directory, "Evaluation_Datasets", "senseval3")

    @computed_field
    @property
    def semcor(self) -> DirectoryPath:
        return Path(self.data_directory, "Training_Corpora", "SemCor")

    @computed_field
    @property
    def semcor_omsti(self) -> DirectoryPath:
        return Path(self.data_directory, "Training_Corpora", "SemCor+OMSTI")


MaruEnglish: MaruEnglishConfig | None = None
RaganatoEnglish: RaganatoEnglishConfig | None = None

if os.environ.get("ENGLISH_MARU_HARD"):
    MaruEnglish = MaruEnglishConfig(data_directory=os.environ.get("ENGLISH_MARU_HARD"))

if os.environ.get("ENGLISH_RAGANATO"):
    RaganatoEnglish = RaganatoEnglishConfig(
        data_directory=os.environ.get("ENGLISH_RAGANATO")
    )
