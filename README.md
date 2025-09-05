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

When training models we make use of a cache directory which by default is set too:

``` bash
$HOME/.cache/experimental_wsd
```

This cache directory path can be overridden by setting it within the environment variable `EXPERIMENTAL_WSD_DATA_PROCESSING_DIR` for instance you could add the following to the .env and then make sure you run `from dotenv import load_dotenv; load_dotenv()` in your Python script to ensure that the environment variable is loaded:

``` bash
EXPERIMENTAL_WSD_DATA_PROCESSING_DIR="A PATH"
```

In addition it is helpful to set the following, this avoids a lot of logging messages asking us to set this environment variable, this relates to how HuggingFace runs the tokenizer (I cannot find any official documentation on this variable, therefore set it to True assuming that it will run the tokenization process in parallel which should be quicker).
``` bash
TOKENIZERS_PARALLELISM=true
```

## Data

To download all of the training and evaluation data to [./data](./data) listed below run the following:

``` bash
cd data
bash download_data.sh
```

For WordNet 3.0 we can use the [Python library wn](https://github.com/goodmami/wn) and to be compatible with previous work we will use the [Princeton WordNet 3.0](https://wordnet.princeton.edu/) and [NLTK](https://www.nltk.org/howto/wordnet.html) version which is [OMW English Wordnet based on WordNet 3.0](https://github.com/omwn/omw-data).

The scripts adds the following environment variables too [./.env](./.env) **Note that the paths will be absolute paths rather than relative as the example shows**:
``` bash
XL_WSD_PATH=data/xl-wsd
ENGLISH_MARU_HARD=data/english-hard-wsd
ENGLISH_RAGANATO=data/WSD_Evaluation_Framework
```

Whereby:
- `XL_WSD_PATH` is the folder path to the [XL-WSD dataset.](#multilingual-xl-wsd-2021)
- `ENGLISH_MARU_HARD` is the folder path to the [English Hard Maru 2022 WSD dataset.](#english-hard-maru-2022-wsd)
- `ENGLISH_RAGANATO` is the folder path to the [English Raganato 2017 WSD dataset.](#english-raganato-2017-wsd)

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

### Mosaico Core USAS data

We have tagged the [Mosaico Core English Wikipedia](https://aclanthology.org/2024.naacl-long.442.pdf) dataset with the CLAWS and English USAS taggers. The CLAWS tagger is a rule based tokenizer, POS, and sentence boundary detection tagger. The English USAS tagger is a rule based lemmatiser and semantic tagger, whereby the semantic tags come from the USAS tagset. The USAS tagger is the C version of the USAS tagger.

This dataset contains 10 JSONL formatted files, of which we expect these files to be in the following folder:
``` bash
./data/mosaico_core_usas
```

Please add the absolute (not relative) path to this directory in the `./env` file like so:

``` bash
MOSAICO_CORE_USAS=/workspaces/experimental-wsd/data/mosaico_core_usas
```

Please add the absolute (not relative) path to the `usas_mapper.yaml` file which should be in the [./data folder](./data) in the `./env` file like so:

``` bash
USAS_MAPPER=/workspaces/experimental-wsd/data/usas_mapper.yaml
```


## WSD Training


To get a predicted good Learning Rate ([using a Learning Rate Finder from Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.tuner.tuning.Tuner.html#lightning.pytorch.tuner.tuning.Tuner.lr_find)) and a recommended maximum batch size:

``` bash
uv run training_runs/semantic_similarity/get_token_similarity_variables_negatives_lr_batch_sizes.py --config training_runs/semantic_similarity/variable_negatives_configs/base_config.yaml
```
Take the values that come out of this are a very rough guide, e.g. with the batch size it would probably be best to choose a value half the size, e.g. 64 if it recommends 128. For the learning rate it is best to use it as a guide and go for a learning rate similar too or identical too a learning rate in a paper that is training a model similar to yours, this guide produced will mainly tell you if it is similar to other learning rate scores which I think is good.

``` bash
uv run $(pwd)/training_runs/semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit --config $(pwd)/training_runs/semantic_similarity/variable_negatives_configs/base_config.yaml --config $(pwd)/training_runs/semantic_similarity/variable_negatives_configs/jhu_clsp_ettin_encoder_17m.yaml
```





### Running on HEX

#### Setup

1. Git clone this repository
2. Create a Python virtual environment: `python3 -m venv venv` and then enter the virtual environment `source venv/bin/activate`
3. Change the `pyproject.toml` file so that the following two lines are removed `en-core-web-trf` and `en-core-web-trf = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0.tar.gz" }` as for some reason `pip` could not install this repository with those two lines in the file.
4. Install the python packages `pip install -e .`
5. Before downloading the required data change the script `./data/download_data.sh` so that it uses `python` rather than `uv run python` and then follow the standard instruction of `cd data && bash download_data.sh`

#### Finding Learning Rate and Recommended Batch Size

##### WSD Token Semantic Similarity with variable negatives

This is an example using the DeBERTa V3 base model however it can be adjusted to any other base model, e.g. BERT, ModernBERT, etc by changing the `--config` file.

``` bash
sbatch slurm_runs/semantic_similarity/variable_negatives/test_lr_batch_size.sh --config $(pwd)/training_runs/semantic_similarity/variable_negatives_configs/deberta_v3_base.yaml
```

This will then generate the following file `./log/error/semantic_similarity_variable_negatives_test_lr_batch_size.log` that will contain a suggested Learning Rate as well as the list of all learning rates tested with their associated loss scores in ascending order of loss value, e.g.

``` bash
INFO:root:LR: 3.630780547701014e-08    Loss: 15.414904287434045
INFO:root:LR: 5.75439937337157e-07    Loss: 15.77297642139897
INFO:root:LR: 1.0964781961431852e-07    Loss: 15.816144917654551
INFO:root:LR: 8.317637711026709e-07    Loss: 15.953934407975186
INFO:root:LR: 6.918309709189366e-07    Loss: 15.959775399719017
INFO:root:LR: 6.3095734448019305e-06    Loss: 15.962526348079619
INFO:root:LR: 4.786300923226383e-07    Loss: 16.01712950715485
INFO:root:LR: 3.0199517204020163e-06    Loss: 16.14076463083797
INFO:root:LR: 1.2022644346174132e-06    Loss: 16.175092202026857
INFO:root:LR: 1e-06    Loss: 16.19031309975706
INFO:root:LR: 5.248074602497728e-06    Loss: 16.28327231364075
INFO:root:LR: 1.445439770745928e-06    Loss: 16.38478278176444
INFO:root:LR: 1.7378008287493761e-06    Loss: 16.386512021698884
INFO:root:LR: 2.089296130854039e-06    Loss: 16.42428246479376
INFO:root:LR: 2.5118864315095797e-06    Loss: 16.559806156732773
INFO:root:LR: 2.51188643150958e-08    Loss: 16.6148078646217
INFO:root:LR: 3.630780547701014e-06    Loss: 16.628934864966823
INFO:root:LR: 4.365158322401661e-06    Loss: 16.63886725934056
INFO:root:LR: 3.019951720402016e-08    Loss: 17.10791306356275
INFO:root:LR: 1.4454397707459274e-08    Loss: 17.30391533905722
INFO:root:LR: 2.0892961308540398e-08    Loss: 18.898985447038516
INFO:root:LR: 1e-08    Loss: 19.18319355357775
INFO:root:Suggested LR: 7.585775750291837e-08, loss at this LR: 14.533493115188765
```

For the learning rate it is best to use it as a guide and go for a learning rate similar too or identical too a learning rate in a paper that is training a model similar to yours, this guide produced will mainly tell you if it is similar to other learning rate scores which I think is good.

It will also state the largest batch size as well as the batch size it test up to:

```bash
`Trainer.fit` stopped: `max_steps=50` reached.
Batch size 2 succeeded, trying batch size 4
`Trainer.fit` stopped: `max_steps=50` reached.
Batch size 4 succeeded, trying batch size 8
`Trainer.fit` stopped: `max_steps=50` reached.
Batch size 8 succeeded, trying batch size 16
Batch size 16 failed, trying batch size 8
Finished batch size finder, will continue with full run using batch size 8
Restoring states from the checkpoint path at /mnt/nfs/homes/mooreap1/experimental-wsd/.scale_batch_size_ecc0c8e8-6ba5-426f-963f-e717de5a9ec6.ckpt
Restored all states from the checkpoint at /mnt/nfs/homes/mooreap1/experimental-wsd/.scale_batch_size_ecc0c8e8-6ba5-426f-963f-e717de5a9ec6.ckpt
INFO:root:Largest batch size: 8
```

Take the values that come out of this are a very rough guide, e.g. with the batch size it would probably be best to choose a value half the size, e.g. 4 if it recommends 8.

#### Pre-Processing data

Not all of the datasets are large enough to require a separate script for pre-processing the data but the ones listed below are, when pre-processing any of these datasets it is best to have a large compute node with many CPUs and 10's of GBs of RAM, the pre-processing scripts do not require a GPU.

##### USAS Token Semantic Similarity with variable negatives

This is an example of how to pre-process the data with the base model `jhu-clsp/ettin-encoder-17m`

``` bash
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/ettin-encoder-17m usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_17_m --num-cpus-pre-processing 15
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/ettin-encoder-68m usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_68m --num-cpus-pre-processing 15
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py FacebookAI/xlm-roberta-base usas_semantic_similarity_variable_nagative_FacebookAI_xlm_roberta_base --num-cpus-pre-processing 15
```

``` bash
srun -p cpu-48h python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/ettin-encoder-17m --num-cpus-pre-processing 5
srun -p cpu-48h python ./training_runs/usas_semantic_similarity/pre_process_dataset.py FacebookAI/xlm-roberta-base --num-cpus-pre-processing 5
```

``` bash
uv run python ./training_runs/usas_semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit \
--config ./training_runs/usas_semantic_similarity/variable_negatives_configs/base_config.yaml \
--config ./training_runs/usas_semantic_similarity/variable_negatives_configs/jhu_clsp_ettin_encoder_17m.yaml --model.learning_rate 1e-5 --data.dataset_folder_name usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_17_m
```

``` bash
sbatch slurm_runs/semantic_similarity/usas_variable_negatives/jhu_clsp_ettin_68m.sh --model.learning_rate 1e-5 --data.dataset_folder_name usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_68m
sbatch slurm_runs/semantic_similarity/usas_variable_negatives/xlmr_base.sh --model.learning_rate 1e-5 --data.dataset_folder_name usas_semantic_similarity_variable_nagative_FacebookAI_xlm_roberta_base
```


``` bash
uv run python ./training_runs/usas_semantic_similarity/evaluate.py ./lightning_logs/usas_jhu_clsp_ettin_encoder_17m/version_14/checkpoints/last.ckpt
uv run python ./training_runs/usas_semantic_similarity/evaluate.py ./lightning_logs/usas_jhu_clsp_ettin_encoder_17m/version_15/checkpoints/last.ckpt
```


#### Training Models

##### WSD Token Semantic Similarity with variable negatives

This is will train 4 DeBERTa V3 base models each with a different learning rate:
``` bash
sbatch slurm_runs/semantic_similarity/variable_negatives/deberta_v3_base.sh
```