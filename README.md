# Experimental WSD
Experiments in Word Sense Disambiguation.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the python package and project manager, e.g. instead of pip, poetry, etc. To install uv see the following [guide](https://docs.astral.sh/uv/getting-started/installation/).

``` bash
uv init --no-package
```

To add a python package

``` bash
uv add PYTHON_PACKAGE
```

To activate a Python repl with all of the project requirements:

``` bash
uv run python
```

To export to a common lock file as determined by [PEP 751](https://peps.python.org/pep-0751/), `pylock.toml` which is not currently fully supported by uv hence the `uv.lock` file :

``` bash
uv export --format pylock.toml -o pylock.toml
```

You can also export to requirements.txt format like so:

``` bash
uv export --format requirements.txt -o requirements.txt
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc. For type checking instead of mypy we are using [pyrefly](https://github.com/facebook/pyrefly), in the future we might move to [ty](https://github.com/astral-sh/ty).

To run the linting and type checking:

``` bash
make check
```

### Tests

To run the tests:

``` bash
make test
```

### Environment variables

To download the [WSL dataset](#word-sense-linking-wsl-bejgu-et-al-2024) this requires you to have access to it and for you to have created a [HuggingFace access token](https://huggingface.co/docs/hub/en/datasets-polars-auth), once you have access to the dataset and have created a HuggingFace access token, add the token to the `HF_TOKEN` variable within [./.env](./.env), e.g.:

``` bash
HF_TOKEN=xxxxxxx
```

## Data

To download all of the training and evaluation data to [./data](./data) listed below run the following:

``` bash
cd data
bash download_data.sh
```

For WordNet 3.0 we can use the [Python library wn](https://github.com/goodmami/wn) and to be compatible with previous work we will use the [Princeton WordNet 3.0](https://wordnet.princeton.edu/) and [NLTK](https://www.nltk.org/howto/wordnet.html) version which is [OMW English Wordnet based on WordNet 3.0](https://github.com/omwn/omw-data).

### English Raganato 2017 WSD

**Note** that a corrected version of this dataset was released with [Maru et al. 2022](https://aclanthology.org/2022.acl-long.324/) of which we have this dataset as part of our training and evaluation dataset, see [section English Hard Maru 2022 WSD](#english-hard-maru-2022-wsd), but we keep this non-corrected version as many of the latest systems use this dataset (see [Blevins and Zettlemoyer 2020](https://aclanthology.org/2020.acl-main.95/), [Barba et al. 2021](https://aclanthology.org/2021.naacl-main.371/), and [Zhang et al. 2022](https://aclanthology.org/2022.coling-1.357.pdf)).

This has come from [Raganato et al. 2017](https://aclanthology.org/E17-1010.pdf) (website of the dataset [http://lcl.uniroma1.it/wsdeval/home](http://lcl.uniroma1.it/wsdeval/home)) it downloads the following training and test datasets too [./data/WSD_Evaluation_Framework/](./data/WSD_Evaluation_Framework/):

- Training
    * SemCor
    * OMSTI
- Test
    * Senseval-2
    * Senseval-3
    * SemEval-07
    * SemEval-13
    * SemEval-15

The results from the combined test sets is called `ALL EN` in [Conia et al. 2024](https://aclanthology.org/2024.naacl-long.442/) and `ALL` in [Raganato et al. 2017](https://aclanthology.org/E17-1010.pdf). In some method based WSD papers the SemEval-7 dataset is used as the development/validation set, see [Barba et al. 2021](https://aclanthology.org/2021.naacl-main.371.pdf) and [Zhang et al. 2022](https://aclanthology.org/2022.coling-1.357/)

### English Hard Maru 2022 WSD

[Maru et al. 2022](https://aclanthology.org/2022.acl-long.324/) whereby the data can be found on the related [GitHub](https://github.com/SapienzaNLP/wsd-hard-benchmark). The datasets can be found in [./data/english-hard-wsd/](./data/english-hard-wsd/). It only contains evaluation/test datasets which are the following:

* Allamended - Corrected version of [Raganato et al. 2017](https://aclanthology.org/E17-1010.pdf) but with the removal of SemEval 2007 as this is normally used as a development dataset.
* S10amended - Corrected version of SemEval 2010 task 17. 
* 42D - A new test set taken from the BNC whereby the ground truth does not occur in SemCor and the first sense within WordNet is not the sense of the target term. The samples of texts come from 42 different domains according to the domains defined in BabelNet version 4.
* hardEN - The samples from All, S10, and 42D that none of the current 7 state of the art systems could get correct.
* softEN - The samples from All, S10, and 42D that at least 1 of the 7 current state of the art systems could get correct.

### Multilingual XL-WSD 2021

This has come from [Pasini et al. 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17609) (website of the dataset [https://sapienzanlp.github.io/xl-wsd/](https://sapienzanlp.github.io/xl-wsd/)) it downloads the following training and test datasets too [./data/xl-wsd/](./data/xl-wsd/). The dataset covers 18 languages including English using the BabelNet version 4 as the sense inventory. The English evaluation dataset is an extension of [Raganato et al. 2017](https://aclanthology.org/E17-1010.pdf) to include SemEval 2010 task 17 as well as the coarse grained WSD task data from SemEval 2007 task 7. All test dataset have been manually annotated whereas the training datasets from languages other than English have been created using English training data and machine translation.


### Word Sense Linking (WSL) Bejgu et al. 2024

Requires permission to access this dataset, but it is still free, you have to request permission through HuggingFace at the dataset's [data page](https://huggingface.co/datasets/Babelscape/wsl). 