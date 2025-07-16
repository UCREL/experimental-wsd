"""
Contains classes and Global variables relating to Part Of Speech tags.
"""

import enum


class UniversalDepPOSTags(enum.StrEnum):
    """
    The Universal Dependency POS tags:
    https://universaldependencies.org/u/pos/
    """

    ADJ = enum.auto()
    ADP = enum.auto()
    ADV = enum.auto()
    AUX = enum.auto()
    CCONJ = enum.auto()
    DET = enum.auto()
    INTJ = enum.auto()
    NOUN = enum.auto()
    NUM = enum.auto()
    PART = enum.auto()
    PRON = enum.auto()
    PROPN = enum.auto()
    PUNCT = enum.auto()
    SCONJ = enum.auto()
    SYM = enum.auto()
    VERB = enum.auto()
    X = enum.auto()


class WordNetPOSTags(enum.StrEnum):
    """
    The 5 POS tags that Princeton WordNet uses:

    n = Noun
    v = Verb
    a = Adjective
    r = Adverb
    s = Adjective Satellite
    https://wordnet.princeton.edu/documentation/senseidx5wn
    https://wordnet.princeton.edu/documentation/wndb5wn
    """

    n = enum.auto()
    v = enum.auto()
    a = enum.auto()
    r = enum.auto()
    s = enum.auto()


# Not sure how to match the WordNet POS tag
# Adjective Satellite to the Universal Dependencies POS tags:
UNI_DEP_TO_WORDNET_POS_TAGS = {
    UniversalDepPOSTags.NOUN: WordNetPOSTags.n,
    UniversalDepPOSTags.VERB: WordNetPOSTags.v,
    UniversalDepPOSTags.ADJ: WordNetPOSTags.a,
    UniversalDepPOSTags.ADV: WordNetPOSTags.r,
    UniversalDepPOSTags.PROPN: WordNetPOSTags.n,
}

# Should match `NOUN` to UniversalDepPOSTags.PROPN as
# the SemCor `NOUN` includes proper nouns.
SEMCOR_TO_UNI_DEP_POS_TAGS = {
    ".": UniversalDepPOSTags.PUNCT,
    "ADJ": UniversalDepPOSTags.ADJ,
    "ADP": UniversalDepPOSTags.ADP,
    "ADV": UniversalDepPOSTags.ADV,
    "CONJ": UniversalDepPOSTags.CCONJ,
    "DET": UniversalDepPOSTags.DET,
    "NOUN": UniversalDepPOSTags.NOUN,
    "NUM": UniversalDepPOSTags.NUM,
    "PRON": UniversalDepPOSTags.PRON,
    "PRT": UniversalDepPOSTags.PART,
    "VERB": UniversalDepPOSTags.VERB,
    "X": UniversalDepPOSTags.X,
}
