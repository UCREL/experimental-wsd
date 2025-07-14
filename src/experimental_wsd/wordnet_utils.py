"""
WordNet (https://github.com/goodmami/wn) helper functions. 
"""

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
        lexicon_name = f'{lexicon.id}:{lexicon.version}'
        if lexicon_name == lexicon_specifier:
            return
    raise ValueError('Cannot find the lexicon with the following specifier: '
                     f'{lexicon_specifier} in the list of lexicons '
                     f'downloaded: {downloaded_lexicons} '
                     'Please download the specified lexicon: '
                     f'uv run python -m wn download {lexicon_specifier}')