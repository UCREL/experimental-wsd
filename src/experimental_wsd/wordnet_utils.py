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
    Given a WordNet lexicon it returns a random sense ID, e.g. "omw-en--apos-hood-08641944-n".

    Args:
        word_net_lexicon (wn.Wordnet): A Wordnet lexicon to get Sense data from.
        senses_to_ignore: (set[str] | None = None): Sense ids that should not
            be returned, if None then no filtering is applied. Default None.
    Returns:
        str: Returns a random sense ID, e.g. "omw-en--apos-hood-08641944-n"

    Raises:
        ValueError: If the Wordnet lexicon contains no Senses.
        ValueError: If the senses to ignore contains all of the senses that
            are in the given Wordnet lexicon.
        ValueError: If the senses to ignore contains all of the senses that are
            in the given Wordnet lexicon.
    """
    all_word_net_senses = word_net_lexicon.senses()
    if len(all_word_net_senses) == 0:
        raise ValueError(
            f"The WordNet lexicon: {word_net_lexicon.describe()} contains no Senses."
        )
    all_word_net_sense_ids: list[str] = []
    sense_ids_to_choose_from: set[str] = set()
    for sense in all_word_net_senses:
        sense_id = sense.id
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


def get_random_sense(
    word_net_lexicon: wn.Wordnet, senses_to_ignore: set[wn.Sense] | None = None
) -> wn.Sense:
    """
    Given a WordNet lexicon it returns a random Wordnet Sense.

    This is the same as `get_random_sense_id` but instead of using the ID of the
    `wn.Sense` we are using the `wn.Sense` objects themselves.

    NOTE: a Sense can become a sense ID by accessing it's attribute `id`, e.g.
    `word_net_lexicon.sense("omw-en--apos-hood-08641944-n").id`

    Args:
        word_net_lexicon (wn.Wordnet): A Wordnet lexicon to get Sense data from.
        senses_to_ignore (set[wn.Sense] | None): Senses that should not be
            returned. If None then no filtering will happen. Default None.
    Returns:
        wn.Sense: Returns a random Sense.

    Raises:
        ValueError: If the Wordnet lexicon contains no Senses.
        ValueError: If the senses to ignore contains all of the senses that are
            in the given Wordnet lexicon.
    """
    senses_ids_to_ignore: set[str] | None = None
    if senses_to_ignore is not None:
        senses_ids_to_ignore = set()
        for sense_to_ignore in senses_to_ignore:
            senses_ids_to_ignore.add(sense_to_ignore.id)

    random_sense_id = get_random_sense_id(
        word_net_lexicon, senses_to_ignore=senses_ids_to_ignore
    )
    return word_net_lexicon.sense(random_sense_id)


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
            Example of a Sense ID `omw-en-carrousel-02966372-n`
    Returns:
        list[str]: All of the senses that are linked to this lemma and POS tag
            in the given WordNet in most likely sense order. Example list:
            ["omw-en-be-02445925-v", "omw-en-be-02697725-v"]
    """
    tmp_senses = word_net_lexicon.senses(lemma, pos_tag)

    if senses_to_ignore is None:
        senses_to_ignore = set()
    # Used to remove duplicates
    tmp_sense_set = set()
    senses: list[str] = []
    for sense in tmp_senses:
        sense_id = sense.id
        if sense_id in tmp_sense_set:
            continue
        if sense_id in senses_to_ignore:
            continue
        senses.append(sense_id)
        tmp_sense_set.add(sense_id)
    return senses


def get_negative_wordnet_sense_ids(
    lemma: str,
    pos_tag: str | None,
    sense_id: str,
    word_net_lexicon: wn.Wordnet,
    get_random_sense: bool = False,
) -> list[str]:
    """
    Given a lemma, optional POS tag, and positive sense ID, it will return all
    of the negative Wordnet sense ids for this sample based on all of the
    senses that are associated to the (lemma, POS tag) which are not
    the positive word net sense ID.

    Args:
        lemma (str): The lemma of the text
        pos_tag (str | None): The POS tag of the text, can be None if not known.
        sense_id (str): The positive Wordnet sense id which should not be part
            of the sense ids returned as negatives. Example would be
            `omw-en-carrousel-02966372-n`
        word_net_lexicon (wn.Wordnet): Wordnet lexicon to find the senses
            of the given lemma and POS tag.
        get_random_sense (bool): If True for non-ambiguous lemma and pos tags
            a random sense ID is created as negative sense ID, else when False
            no negative sense ID will be given for that sample, e.g. returns an
            empty list. In essence if all Senses that are returned for the
            given lemma and POS tag is the `sense_id` and this is True, then
            a random sense ID is created as a negative sense ID. Default False.
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


def get_normalised_mwe_lemma_for_wordnet(mwe_lemma: str) -> str:
    """
    Given a Multi Word Expression (MWE) lemma that is likely to have come from the
    SemCor dataset, lemma not token, it will normalise it so that it can be
    found within Wordnet.

    In essence this function replaces all `_` with a whitespace token.

    Args:
        mwe_lemma (str): The Multi Word Expression (MWE) lemma to be normalised.
    Returns:
        str: The normalised MWE.
    """
    if "_" not in mwe_lemma:
        return mwe_lemma
    return mwe_lemma.replace("_", " ")
