import importlib
import os
from pathlib import Path

from experimental_wsd import config


def test_global_variables(tmp_path: Path):
    """
    Tests the following global variables:
    * MaruEnglish
    * RaganatoEnglish
    """
    maru_data_dir = Path(tmp_path, "maru")
    os.mkdir(maru_data_dir)
    os.mkdir(Path(maru_data_dir, "42D"))
    Path(maru_data_dir, "42D", "42D.data.xml").touch()
    Path(maru_data_dir, "42D", "42D.gold.key.txt").touch()
    os.mkdir(Path(maru_data_dir, "ALLamended"))
    os.mkdir(Path(maru_data_dir, "hardEN"))
    os.mkdir(Path(maru_data_dir, "softEN"))
    os.mkdir(Path(maru_data_dir, "S10amended"))

    if "ENGLISH_MARU_HARD" not in os.environ:
        os.environ["ENGLISH_MARU_HARD"] = str(maru_data_dir.resolve())
        importlib.reload(config)
        assert isinstance(config.MaruEnglish, config.MaruEnglishConfig)
        config.MaruEnglish.fortitude
        assert config.MaruEnglish.fortitude.gold == Path(
            maru_data_dir, "42D", "42D.gold.key.txt"
        )
        assert config.MaruEnglish.fortitude.data == Path(
            maru_data_dir, "42D", "42D.data.xml"
        )
        del os.environ["ENGLISH_MARU_HARD"]
    else:
        importlib.reload(config)
        assert isinstance(config.MaruEnglish, config.MaruEnglishConfig)

    raganato_data_dir = Path(tmp_path, "raganato")
    os.mkdir(raganato_data_dir)
    raganato_eval_data_dir = Path(raganato_data_dir, "Evaluation_Datasets")
    os.mkdir(raganato_eval_data_dir)
    os.mkdir(Path(raganato_eval_data_dir, "ALL"))
    os.mkdir(Path(raganato_eval_data_dir, "semeval2007"))
    os.mkdir(Path(raganato_eval_data_dir, "semeval2013"))
    os.mkdir(Path(raganato_eval_data_dir, "semeval2015"))
    os.mkdir(Path(raganato_eval_data_dir, "senseval2"))
    os.mkdir(Path(raganato_eval_data_dir, "senseval3"))
    raganato_train_data_dir = Path(raganato_data_dir, "Training_Corpora")
    os.mkdir(raganato_train_data_dir)
    os.mkdir(Path(raganato_train_data_dir, "SemCor"))
    os.mkdir(Path(raganato_train_data_dir, "SemCor+OMSTI"))

    if "ENGLISH_RAGANATO" not in os.environ:
        os.environ["ENGLISH_RAGANATO"] = str(raganato_data_dir.resolve())
        importlib.reload(config)
        assert isinstance(config.RaganatoEnglish, config.RaganatoEnglishConfig)
        del os.environ["ENGLISH_RAGANATO"]
    else:
        importlib.reload(config)
        assert isinstance(config.RaganatoEnglish, config.RaganatoEnglishConfig)
