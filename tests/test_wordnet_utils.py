from pathlib import Path

import pytest
import wn

from experimental_wsd.wordnet_utils import (
    check_lexicon_exists,
    get_all_senses,
    get_definition,
    get_negative_wordnet_sense_ids,
    get_normalised_mwe_lemma_for_wordnet,
    get_random_sense,
    get_random_sense_id,
)

EN_LEXICON = "omw-en:1.4"
ENGLISH_WN = wn.Wordnet(lexicon=EN_LEXICON, expand="")


class FakeSynset:
    def __init__(self):
        pass

    def definition(self) -> str:
        return ""


class FakeSense:
    def __init__(self):
        pass

    def synset(self) -> FakeSynset:
        return FakeSynset()


class FakeLexicon:
    def __init__(self):
        pass

    def sense(self, sense_id: str) -> FakeSense:
        return FakeSense()

    def senses(self) -> list[str]:
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


def test_get_all_senses():
    expected_be_noun_senses = ["omw-en-Be-14631295-n"]
    expected_be_verb_senses = [
        "omw-en-be-02604760-v",
        "omw-en-be-02616386-v",
        "omw-en-be-02655135-v",
        "omw-en-be-02603699-v",
        "omw-en-be-02749904-v",
        "omw-en-be-02664769-v",
        "omw-en-be-02620587-v",
        "omw-en-be-02445925-v",
        "omw-en-be-02697725-v",
        "omw-en-be-02268246-v",
        "omw-en-be-02614181-v",
        "omw-en-be-02744820-v",
        "omw-en-be-02702508-v",
    ]
    expected_be_all_senses = expected_be_noun_senses + expected_be_verb_senses
    assert expected_be_all_senses == get_all_senses(ENGLISH_WN, "be", None)
    assert expected_be_noun_senses == get_all_senses(ENGLISH_WN, "be", "n")
    assert expected_be_verb_senses == get_all_senses(ENGLISH_WN, "be", "v")

    expected_be_verb_senses.remove("omw-en-be-02603699-v")
    assert expected_be_verb_senses == get_all_senses(
        ENGLISH_WN,
        "be",
        "v",
        senses_to_ignore=set(["omw-en-be-02603699-v", "omw-en-14631295-n"]),
    )

    assert [] == get_all_senses(ENGLISH_WN, "madeupword", None)


def test_get_random_sense_id():
    with pytest.raises(ValueError):
        get_random_sense_id(FakeLexicon())

    random_sense_id = get_random_sense_id(ENGLISH_WN)
    assert isinstance(random_sense_id, str)

    all_sense_ids = [sense.id for sense in ENGLISH_WN.senses()]
    expected_sense_id = all_sense_ids.pop()

    assert expected_sense_id == get_random_sense_id(ENGLISH_WN, set(all_sense_ids))

    all_sense_ids.append(expected_sense_id)

    with pytest.raises(ValueError):
        get_random_sense_id(ENGLISH_WN, set(all_sense_ids))


def test_get_random_sense():
    with pytest.raises(ValueError):
        get_random_sense(FakeLexicon())

    random_sense = get_random_sense(ENGLISH_WN)
    assert isinstance(random_sense, wn.Sense)

    all_senses = ENGLISH_WN.senses()
    expected_sense = all_senses.pop()

    assert expected_sense == get_random_sense(ENGLISH_WN, set(all_senses))

    all_senses.append(expected_sense)

    with pytest.raises(ValueError):
        get_random_sense(ENGLISH_WN, set(all_senses))


@pytest.mark.parametrize("get_random_sense", [True, False])
def test_get_negative_wordnet_sense_ids(get_random_sense: bool):
    lemma = "be"
    pos_tag = "v"
    sense_id = ["omw-en-be-02603699-v", "omw-en-be-02604760-v"]

    expected_negative_verb_senses = [
        "omw-en-be-02616386-v",
        "omw-en-be-02655135-v",
        "omw-en-be-02749904-v",
        "omw-en-be-02664769-v",
        "omw-en-be-02620587-v",
        "omw-en-be-02445925-v",
        "omw-en-be-02697725-v",
        "omw-en-be-02268246-v",
        "omw-en-be-02614181-v",
        "omw-en-be-02744820-v",
        "omw-en-be-02702508-v",
    ]

    assert expected_negative_verb_senses == get_negative_wordnet_sense_ids(
        lemma,
        pos_tag,
        sense_id,
        ENGLISH_WN,
        get_random_sense=get_random_sense,
    )

    expected_negative_senses = [
        "omw-en-Be-14631295-n",
        "omw-en-be-02616386-v",
        "omw-en-be-02655135-v",
        "omw-en-be-02749904-v",
        "omw-en-be-02664769-v",
        "omw-en-be-02620587-v",
        "omw-en-be-02445925-v",
        "omw-en-be-02697725-v",
        "omw-en-be-02268246-v",
        "omw-en-be-02614181-v",
        "omw-en-be-02744820-v",
        "omw-en-be-02702508-v",
    ]

    pos_tag = None
    assert expected_negative_senses == get_negative_wordnet_sense_ids(
        lemma,
        pos_tag,
        sense_id,
        ENGLISH_WN,
        get_random_sense=get_random_sense,
    )

    laptop_sense_id = ["omw-en-laptop-03642806-n"]
    lemma = "laptop"
    pos_tag = "n"

    negative_sense_ids = get_negative_wordnet_sense_ids(
        lemma,
        pos_tag,
        laptop_sense_id,
        ENGLISH_WN,
        get_random_sense=get_random_sense,
    )

    if get_random_sense:
        assert 1 == len(negative_sense_ids)
        assert negative_sense_ids[0] != laptop_sense_id
    else:
        assert 0 == len(negative_sense_ids)


def test_get_normalised_mwe_lemma_for_wordnet():
    assert "stone fruit" == get_normalised_mwe_lemma_for_wordnet("stone_fruit")
    assert "stone fruit" == get_normalised_mwe_lemma_for_wordnet("stone fruit")
    assert "stonefruit" == get_normalised_mwe_lemma_for_wordnet("stonefruit")
    assert "stone" == get_normalised_mwe_lemma_for_wordnet("stone")
    assert "" == get_normalised_mwe_lemma_for_wordnet("")


def test_get_definition():
    new_york_sense_id = "omw-en-New_York-09119277-n"
    expected_definition = (
        "the largest city in New York State and in the United States; "
        "located in southeastern New York at the mouth of the Hudson river; "
        "a major financial and cultural center"
    )
    assert expected_definition == get_definition(new_york_sense_id, ENGLISH_WN)

    with pytest.raises(ValueError):
        get_definition(new_york_sense_id, FakeLexicon())
