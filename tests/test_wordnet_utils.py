from pathlib import Path

import pytest
import wn

from experimental_wsd.wordnet_utils import check_lexicon_exists


def test_check_lexicon_exists(tmp_path: Path):
    # Change the default data directory so that we can test the function.
    test_lexicon = "omw-nn:1.4"
    wn.config.data_directory = str(tmp_path)
    with pytest.raises(ValueError):
        check_lexicon_exists(test_lexicon)

    wn.download(test_lexicon)
    check_lexicon_exists(test_lexicon)
