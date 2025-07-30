"""
WordNet (https://github.com/goodmami/wn) helper functions.
"""

import random

import wn


def check_lexicon_exists(lexicon_specifier: str) -> None:
    """
    Ensures that the lexicon associated to the given
    specifier, e.g. `omw-en:1.4` has been downloaded. If the lexicon cannot be
    found it raises a ValueError.

    Note to download a lexicon with a given specifier ():
    `uv run python -m wn download omw-en:1.4`

    A list of allowed lexicons and their specifiers can be found at the following
    URL:
    https://github.com/goodmami/wn?tab=readme-ov-file#available-wordnets

    By default the WordNet lexicons are downloaded too:
    ```
    $HOME/.wn_data
    ```
    This directory can be changed through the following attribute:
    `wn.config.data_directory`

    Returns:
        None: This happens when the lexicon can be found/it does exist and can
            be used.
    Raises:
        ValueError: If the lexicon with the given specifier has not been
            downloaded.
    """

    downloaded_lexicons = wn.lexicons()
    for lexicon in downloaded_lexicons:
        lexicon_name = f"{lexicon.id}:{lexicon.version}"
        if lexicon_name == lexicon_specifier:
            return
    raise ValueError(
        "Cannot find the lexicon with the following specifier: "
        f"{lexicon_specifier} in the list of lexicons "
        f"downloaded: {downloaded_lexicons} "
        "Please download the specified lexicon: "
        f"uv run python -m wn download {lexicon_specifier}"
    )


def get_random_sense_id(
    word_net_lexicon: wn.Wordnet, senses_to_ignore: set[str] | None = None
) -> str:
    """
    Given a WordNet lexicon it returns a random sense ID, e.g. "omw-en-02604760-v".

    Args:
        word_net_lexicon (wn.Wordnet): A Wordnet lexicon to get Synset data from.
        senses_to_ignore: (set[str] | None = None): Sense ids that should not
            be returned, if None then no filtering is applied. Default None.
    Returns:
        str: Returns a random sense ID, e.g. "omw-en-02604760-v"

    Raises:
        ValueError: If the Wordnet lexicon contains no Synsets.
        ValueError: If the senses to ignore contains all of the senses that
            are in the given Wordnet lexicon.
        ValueError: If the synsets to ignore contains all of the synsets that are
            in the given Wordnet lexicon.
    """
    all_word_net_synsets = word_net_lexicon.synsets()
    if len(all_word_net_synsets) == 0:
        raise ValueError(
            f"The WordNet lexicon: {word_net_lexicon.describe()} contains no Synsets."
        )
    all_word_net_sense_ids: list[str] = []
    sense_ids_to_choose_from: set[str] = set()
    for synset in all_word_net_synsets:
        sense_id = synset.id
        all_word_net_sense_ids.append(sense_id)
        if senses_to_ignore is not None:
            if sense_id in senses_to_ignore:
                continue
            sense_ids_to_choose_from.add(sense_id)

    if senses_to_ignore is not None:
        if len(sense_ids_to_choose_from) == 0:
            raise ValueError(
                "The senses to ignore has removed all of the possible "
                "senses to choose from, given the following lexicon: "
                f"{word_net_lexicon.describe()}"
            )
        all_word_net_sense_ids = list(sense_ids_to_choose_from)

    random_sense_id = random.choice(all_word_net_sense_ids)
    return random_sense_id


def get_random_synset(
    word_net_lexicon: wn.Wordnet, synsets_to_ignore: set[wn.Synset] | None = None
) -> wn.Synset:
    """
    Given a WordNet lexicon it returns a random synset.

    NOTE: a synset can become a sense ID by accessing it's attribute `id`, e.g.
    `a_synset.id`

    Args:
        word_net_lexicon (wn.Wordnet): A Wordnet lexicon to get Synset data from.
        synsets_to_ignore (set[wn.Synset] | None): Synsets that should not be
            returned. If None then no filtering will happen. Default None.
    Returns:
        wn.Synset: Returns a random synset.

    Raises:
        ValueError: If the Wordnet lexicon contains no Synsets.
        ValueError: If the synsets to ignore contains all of the synsets that are
            in the given Wordnet lexicon.
    """
    all_word_net_synsets = word_net_lexicon.synsets()
    if len(all_word_net_synsets) == 0:
        raise ValueError(
            f"The WordNet lexicon: {word_net_lexicon.describe()} contains no Synsets."
        )

    if synsets_to_ignore is not None:
        all_word_net_synset_set = set(all_word_net_synsets)
        synsets_to_choose_from = all_word_net_synset_set.difference(synsets_to_ignore)
        if len(synsets_to_choose_from) == 0:
            raise ValueError(
                "The synsets to ignore has removed all of the possible "
                "synsets to choose from, given the following lexicon: "
                f"{word_net_lexicon.describe()}"
            )
        all_word_net_synsets = list(synsets_to_choose_from)

    random_synset = random.choice(all_word_net_synsets)
    return random_synset


def get_all_senses(
    word_net_lexicon: wn.Wordnet,
    lemma: str,
    pos_tag: str | None,
    senses_to_ignore: set[str] | None = None,
) -> list[str]:
    """
    Given a lemma and optional POS tag return all possible senses from the
    given Wordnet lexicon, the senses are returned in the order of 1st
    matched word and all of it's senses then 2nd matched word and it's senses
    until the last matched word and it's senses. Therefore the first sense
    should be the most likely sense according to WordNet.

    The function removes any possible duplicate senses.

    Args:
        word_net_lexicon (wn.Wordnet): The Wordnet lexicon to get the sense data
            from.
        lemma (str): The lemma of the text
        pos_tag (str | None): The POS tag of the text, can be None if not known.
        senses_to_ignore (set[str] | None): Sense ids that should not be returned
            in the list, if None then it will return all sense ids. Default None.
    Returns:
        list[str]: All of the senses that are linked to this lemma and POS tag
            in the given WordNet in most likely sense order. Example list:
            ["omw-en-02604760-v", "omw-en-02616386-v"]
    """
    senses: list[str] = []
    senses_set: set[str] = set()

    if senses_to_ignore is None:
        senses_to_ignore = set()

    word_net_words = word_net_lexicon.words(lemma, pos_tag)

    for word in word_net_words:
        for word_synset in word.synsets():
            sense_id = word_synset.id
            if sense_id in senses_set:
                continue
            if sense_id in senses_to_ignore:
                continue

            senses_set.add(sense_id)
            senses.append(sense_id)

    return senses


def get_negative_wordnet_sense_ids(
    lemma: str,
    pos_tag: str | None,
    sense_id: str,
    word_net_lexicon: wn.Wordnet,
    get_random_sense: bool = False,
) -> list[str]:
    """
    Given a data sample containing at least the following key-values:
    * `sense_id_key` (str | list[str]) - The positive word net sense ID for the sample.
        This can be either a single string or a list of strings.
    * `lemma_key` (str) - The lemma of the sample
    * `pos_tag_key` (str | None) - The POS tag of the sample which can be None.

    It will return all of the negative Wordnet sense ids for this sample based on
    all of the senses that are associated to the (lemma, POS tag) which are not
    the positive word net sense ID.

    Args:
        sample (dict[str, str | None]): The sample to get word net sense ids
            for.
        sense_id_key (str): The key in the sample that is associated to the
            positive/correct sense ID. The positive/correct sense ID can be
            a single string or a list of strings.
        word_net_lexicon (wn.Wordnet): Wordnet lexicon to find the senses
            of the given lemma and POS tag.
        lemma_key (str): The key in the sample that is associated to the
            lemma. Default is "lemma".
        pos_tag_key (str): The key in the sample that is associated to the
            POS tag value. Default is "pos_tag".
        get_random_sense (bool): If True for non-ambiguous lemma and pos tags
            a random sense ID is created as negative sense ID, else when False
            no negative sense ID will be given for that sample, e.g. returns an
            empty list. Default False.
    Returns:
        list[str]: The negative Wordnet sense IDs in Wordnet order, meaning the
            first sense ID should be the most likely for the given (lemma, POS tag).
    """

    positive_sense_id = sense_id

    negative_sense_ids_to_ignore = set([positive_sense_id])
    negative_sense_ids = get_all_senses(
        word_net_lexicon, lemma, pos_tag, senses_to_ignore=negative_sense_ids_to_ignore
    )

    if negative_sense_ids:
        return negative_sense_ids

    if get_random_sense:
        random_sense_id = get_random_sense_id(
            word_net_lexicon, negative_sense_ids_to_ignore
        )
        return [random_sense_id]

    return negative_sense_ids


def get_normalised_wordnet_mwe(mwe: str) -> str:
    """
    """
    if not "_" in mwe:
        return mwe
    return " ".join(mwe.split("_")) 
