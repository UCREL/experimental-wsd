from pathlib import Path

import pytest
import wn

from experimental_wsd.wordnet_utils import (
    check_lexicon_exists,
    get_all_senses,
    get_negative_wordnet_sense_ids,
    get_random_sense_id,
    get_random_synset,
)

EN_LEXICON = "omw-en:1.4"
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand="")


class FakeLexicon:
    def __init__(self):
        pass

    def synsets(self) -> list[str]:
        return []

    def describe(self) -> str:
        return ""


def test_check_lexicon_exists(tmp_path: Path):
    # Change the default data directory so that we can test the function.

    original_config_data_path = wn.config.data_directory
    wn.config.data_directory = str(tmp_path)
    with pytest.raises(ValueError):
        check_lexicon_exists(EN_LEXICON)

    wn.download(EN_LEXICON)
    check_lexicon_exists(EN_LEXICON)
    wn.config.data_directory = original_config_data_path


def test_get_random_synset():
    with pytest.raises(ValueError):
        get_random_synset(FakeLexicon())

    random_synset = get_random_synset(ENGLISH_WN)
    assert isinstance(random_synset, wn.Synset)

    all_synsets = ENGLISH_WN.synsets()
    expected_synset = all_synsets.pop()

    assert expected_synset == get_random_synset(ENGLISH_WN, set(all_synsets))

    all_synsets.append(expected_synset)

    with pytest.raises(ValueError):
        get_random_synset(ENGLISH_WN, all_synsets)


def test_get_all_senses():
    expected_be_noun_senses = ["omw-en-14631295-n"]
    expected_be_verb_senses = [
        "omw-en-02604760-v",
        "omw-en-02616386-v",
        "omw-en-02655135-v",
        "omw-en-02603699-v",
        "omw-en-02749904-v",
        "omw-en-02664769-v",
        "omw-en-02620587-v",
        "omw-en-02445925-v",
        "omw-en-02697725-v",
        "omw-en-02268246-v",
        "omw-en-02614181-v",
        "omw-en-02744820-v",
        "omw-en-02702508-v",
    ]
    expected_be_all_senses = expected_be_noun_senses + expected_be_verb_senses
    assert expected_be_all_senses == get_all_senses(ENGLISH_WN, "be", None)
    assert expected_be_noun_senses == get_all_senses(ENGLISH_WN, "be", "n")
    assert expected_be_verb_senses == get_all_senses(ENGLISH_WN, "be", "v")

    expected_be_verb_senses.remove("omw-en-02702508-v")
    assert expected_be_verb_senses == get_all_senses(
        ENGLISH_WN,
        "be",
        "v",
        senses_to_ignore=set(["omw-en-02702508-v", "omw-en-14631295-n"]),
    )

    assert [] == get_all_senses(ENGLISH_WN, "madeupword", None)


def test_get_random_sense_id():
    with pytest.raises(ValueError):
        get_random_sense_id(FakeLexicon())

    random_sense_id = get_random_sense_id(ENGLISH_WN)
    assert isinstance(random_sense_id, str)

    all_sense_ids = [synset.id for synset in ENGLISH_WN.synsets()]
    expected_sense_id = all_sense_ids.pop()

    assert expected_sense_id == get_random_sense_id(ENGLISH_WN, set(all_sense_ids))

    all_sense_ids.append(expected_sense_id)

    with pytest.raises(ValueError):
        get_random_sense_id(ENGLISH_WN, all_sense_ids)


@pytest.mark.parametrize("positive_sense_id_is_list", [True, False])
@pytest.mark.parametrize("get_random_sense", [True, False])
def test_get_negative_wordnet_sense_ids(
    get_random_sense: bool, positive_sense_id_is_list: bool
):
    lemma_key = "lemma"
    pos_tag_key = "pos"
    sense_id_key = "label_id"
    sense_id_value = "omw-en-02614181-v"
    if positive_sense_id_is_list:
        sense_id_value = [sense_id_value]

    expected_negative_verb_senses = [
        "omw-en-02604760-v",
        "omw-en-02616386-v",
        "omw-en-02655135-v",
        "omw-en-02603699-v",
        "omw-en-02749904-v",
        "omw-en-02664769-v",
        "omw-en-02620587-v",
        "omw-en-02445925-v",
        "omw-en-02697725-v",
        "omw-en-02268246-v",
        "omw-en-02744820-v",
        "omw-en-02702508-v",
    ]

    test_sample = {lemma_key: "be", pos_tag_key: "v", sense_id_key: sense_id_value}

    assert expected_negative_verb_senses == get_negative_wordnet_sense_ids(
        test_sample,
        sense_id_key,
        ENGLISH_WN,
        lemma_key,
        pos_tag_key,
        get_random_sense=get_random_sense,
    )

    expected_negative_senses = [
        "omw-en-14631295-n",
        "omw-en-02604760-v",
        "omw-en-02616386-v",
        "omw-en-02655135-v",
        "omw-en-02603699-v",
        "omw-en-02749904-v",
        "omw-en-02664769-v",
        "omw-en-02620587-v",
        "omw-en-02445925-v",
        "omw-en-02697725-v",
        "omw-en-02268246-v",
        "omw-en-02744820-v",
        "omw-en-02702508-v",
    ]
    test_sample[pos_tag_key] = None
    assert expected_negative_senses == get_negative_wordnet_sense_ids(
        test_sample,
        sense_id_key,
        ENGLISH_WN,
        lemma_key,
        pos_tag_key,
        get_random_sense=get_random_sense,
    )

    laptop_sense_id = "omw-en-03642806-n"
    if positive_sense_id_is_list:
        laptop_sense_id = [laptop_sense_id]
    test_sample = {
        lemma_key: "laptop",
        pos_tag_key: None,
        sense_id_key: laptop_sense_id,
    }

    negative_sense_ids = get_negative_wordnet_sense_ids(
        test_sample,
        sense_id_key,
        ENGLISH_WN,
        lemma_key,
        pos_tag_key,
        get_random_sense=get_random_sense,
    )

    if get_random_sense:
        assert 1 == len(negative_sense_ids)
        assert negative_sense_ids[0] != laptop_sense_id
    else:
        assert 0 == len(negative_sense_ids)
