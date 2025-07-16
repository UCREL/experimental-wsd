from experimental_wsd.pos_constants import (
    UNI_DEP_TO_WORDNET_POS_TAGS,
    UniversalDepPOSTags,
    WordNetPOSTags,
)


def test_universal_dep_pos_tags():
    # pyrefly: ignore[no-matching-overload]
    assert 17 == len(set(UniversalDepPOSTags))
    # pyrefly: ignore[bad-argument-type]
    for tag in UniversalDepPOSTags:
        assert tag.name != tag.value
        assert tag.name.lower() == tag.value


def test_word_net_pos_tags():
    # pyrefly: ignore[no-matching-overload]
    assert 5 == len(set(WordNetPOSTags))
    # pyrefly: ignore[bad-argument-type]
    for tag in WordNetPOSTags:
        assert tag.name == tag.value
        assert tag.name.lower() == tag.value


def test_uni_dep_to_word_net_pos_tags():
    assert 5 == len(UNI_DEP_TO_WORDNET_POS_TAGS)
    
    assert UNI_DEP_TO_WORDNET_POS_TAGS["noun"] == "n"
    assert UNI_DEP_TO_WORDNET_POS_TAGS["verb"] == "v"
    assert UNI_DEP_TO_WORDNET_POS_TAGS["adj"] == "a"
    assert UNI_DEP_TO_WORDNET_POS_TAGS["adv"] == "r"
    assert UNI_DEP_TO_WORDNET_POS_TAGS["propn"] == "n"
