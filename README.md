# Neural Semantic Tagger for USAS

This repository contains the code to train the PyMUSAS Neural tagger, a BERT/transformer based semantic tagger that outputs token level semantic tags from the [USAS tagset](https://ucrel.lancs.ac.uk/usas/usas_guide.pdf).

The scripts described in this README allow you to train a variation of the [Bi-Encoder Model (BEM) from Blevins and Zettlemoyer 2020](https://aclanthology.org/2020.acl-main.95.pdf) a Word Sense Disambiguation (WSD) model. The only difference between the original and this version is that this version ties the weights of the context and gloss encoder. The model is trained to find the most relevant gloss/description for a given contextualised token that is to be disambiguated. The description comes from the semantic tagset, which in this case is USAS, whereby the description describes a semantic tag.


## Setup

**Note** we assume you are running this code on a computer/machine that is using a Nvidia GPU, the dev-container assumes a Nvidia GPU. The local setup might work without a GPU but it will be very slow to train the BERT based models.

You can either use the dev container with your favourite editor, e.g. VSCode. Or you can create your setup locally below we demonstrate both.

In both cases they share the same tools, of which these tools are:
* [uv](https://docs.astral.sh/uv/) for Python packaging and development
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.

### Dev Container

A [dev container](https://containers.dev/) uses a docker container to create the required development environment, the Dockerfile we use for this dev container can be found at [./.devcontainer/Dockerfile](./.devcontainer/Dockerfile). To run it locally it requires docker to be installed, you can also run it in a cloud based code editor, for a list of supported editors/cloud editors see [the following webpage.](https://containers.dev/supporting)

To run for the first time on a local VSCode editor (a slightly more detailed and better guide on the [VSCode website](https://code.visualstudio.com/docs/devcontainers/tutorial)):
1. Ensure docker is running.
2. Ensure the VSCode [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension is installed in your VSCode editor.
3. Open the command pallete `CMD + SHIFT + P` and then select `Dev Containers: Rebuild and Reopen in Container`

You should now have everything you need to develop, `uv`, `make`, for VSCode various extensions like `Pylance`, etc.

If you have any trouble see the [VSCode website.](https://code.visualstudio.com/docs/devcontainers/tutorial).

### Local

To run locally first ensure you have the following tools installted locally:
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python packaging and development. (version `0.9.6`)
* [make](https://www.gnu.org/software/make/) (OPTIONAL) for automation of tasks, not strictly required but makes life easier.
  * Ubuntu: `apt-get install make`
  * Mac: [Xcode command line tools](https://mac.install.guide/commandlinetools/4) includes `make` else you can use [brew.](https://formulae.brew.sh/formula/make)
  * Windows: Various solutions proposed in this [blog post](https://earthly.dev/blog/makefiles-on-windows/) on how to install on Windows, inclduing `Cygwin`, and `Windows Subsystem for Linux`.

When developing on the project you will want to install the Python package locally in editable format with all the extra requirements, this can be done like so:

```bash
uv sync
```

### Linting

Linting and formatting with [ruff](https://docs.astral.sh/ruff/) it is a replacement for tools like Flake8, isort, Black etc.

To run the linting:

``` bash
make check
```

### Tests

To run the tests:

``` bash
make test
```

### Environment variables

**Note** environment variables are stored in the `./.env` file, of which can example file could look like, do not share this file as it can/will contains sensitive information like a HuggingFace token:

``` bash
HF_TOKEN="TOKEN_VALUE"
EXPERIMENTAL_WSD_DATA_PROCESSING_DIR=/workspaces/experimental-wsd/data/processed_data
TOKENIZERS_PARALLELISM=true
MOSAICO_CORE_USAS=/workspaces/experimental-wsd/data/mosaico_core_usas
USAS_MAPPER=/workspaces/experimental-wsd/data/usas_mapper.yaml
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
bash download_usas_data.sh
# WordNet is required due to past dependencies within the code base.
uv run python -m wn download omw-en:1.4
```

### Mosaico Core USAS data

We have tagged the [Mosaico Core English Wikipedia](https://aclanthology.org/2024.naacl-long.442.pdf) dataset with the CLAWS and English USAS taggers, the code for this tagging process can be found at [https://github.com/UCREL/mosaico-usas-processing](https://github.com/UCREL/mosaico-usas-processing). The CLAWS tagger is a rule based tokenizer, POS, and sentence boundary detection tagger. The English USAS tagger is a rule based lemmatiser and semantic tagger, whereby the semantic tags come from the USAS tagset. The USAS tagger is the C version of the USAS tagger, see the tagging repository for more details.

The processed dataset has been uploaded to HuggingFace Hub as a dataset and that is where we have downloaded it from, the repository can be found at [https://huggingface.co/datasets/ucrelnlp/English-USAS-Mosaico](https://huggingface.co/datasets/ucrelnlp/English-USAS-Mosaico).

This dataset contains 10 JSONL formatted files, of which we expect these files to be in the following folder:
``` bash
./data/mosaico_core_usas
```

Please add the absolute (not relative) path to this directory in the `./env` file like so:

``` bash
MOSAICO_CORE_USAS=/workspaces/experimental-wsd/data/mosaico_core_usas
```

The USAS mapper file ([./data/usas_mapper.yaml](./data/usas_mapper.yaml)), which comes with this repository, is a mapper of a USAS semantic tag to it's description, e.g. `A1.1.1` = `Title: General actions, making etc. Description: General/abstract terms relating to an activity/action...`. This mapper file is used by the model, as with the BEM architecture it finds the most relevant description for a given token that it is disambiguating. Thus this mapper allows us to map semantic tags to their descriptions and vice versa.

Please add the absolute (not relative) path to the `usas_mapper.yaml` file which should be in the [./data folder](./data) in the `./env` file like so:

``` bash
USAS_MAPPER=/workspaces/experimental-wsd/data/usas_mapper.yaml
```


## WSD Training

The data downloaded from the [Mosaico core USAS data](#mosaico-core-usas-data) we only train on the first file `wikipedia_export.jsonl.0` this is because the dataset was large enough for our initial experiments to show that we are getting good performance without the rest of the dataset and due to computational reasons. This data is first pre-processed and then used for training. We validate on the first 20,000 sentences of the `wikipedia_export.jsonl.8` file.

We will be fine-tuning 4 BERT models:
* [jhu-clsp/ettin-encoder-17m](https://huggingface.co/jhu-clsp/ettin-encoder-17m) - English only 17 million parameter model.
* [jhu-clsp/ettin-encoder-68m](https://huggingface.co/jhu-clsp/ettin-encoder-68m) - English only 68 million parameter model.
* [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) - Multilingual 140 million parameter model.
* [jhu-clsp/mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) - Multilingual 307 million parameter model.

### Pre-Processing

The pre-processing is model specific, in the code block below are the 4 commands used to pre-process the data for the 4 models we fine tuned. In all cases we have stated we are processing the data with `15` CPUs and that the label `Z99` should be removed from the dataset. The `Z99` label is removed as this label represents the unknown category which is semantically not very useful to learn.

``` bash
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/ettin-encoder-17m usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_17_m_z99_filtered --num-cpus-pre-processing 15 --filter-out-labels Z99
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/ettin-encoder-68m usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_68_m_z99_filtered --num-cpus-pre-processing 15 --filter-out-labels Z99
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/mmBERT-small usas_semantic_similarity_variable_nagative_jhu_clsp_mmBERT_small_z99_filtered --num-cpus-pre-processing 15 --filter-out-labels Z99
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py jhu-clsp/mmBERT-base usas_semantic_similarity_variable_nagative_jhu_clsp_mmBERT_base_z99_filtered --num-cpus-pre-processing 15 --filter-out-labels Z99
```

These final datasets that are used for training and validating on are stored at `$EXPERIMENTAL_WSD_DATA_PROCESSING_DIR/machine_learning_data/usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_17_m_z99_filtered` for the `jhu-clsp/ettin-encoder-17m`.

**Note** that pre-processing can take a while and can take up a large amount of disk storage (26GB per model for the final dataset and as we use HuggingFace mapping/filtering scripts these steps can also be cached with the HuggingFace cache directory which will take up more space).

For additional information on this script get the help information:
``` bash
uv run python ./training_runs/usas_semantic_similarity/pre_process_dataset.py --help
```

### Model Training

The model training was primarily conducted on our University Slurm cluster, these Slurm scripts in essence ran the following commands:

``` bash
uv run python ./training_runs/usas_semantic_similarity/train_and_evaluate_token_similarity_variables_negatives.py fit \
--config ./training_runs/usas_semantic_similarity/variable_negatives_configs/base_config.yaml \
--config ./training_runs/usas_semantic_similarity/variable_negatives_configs/jhu_clsp_ettin_encoder_17m.yaml \
--model.learning_rate 1e-5 \
--data.dataset_folder_name usas_semantic_similarity_variable_nagative_jhu_clsp_ettin_encoder_17_m \
--ckpt_path ./lightning_logs/usas_jhu_clsp_ettin_encoder_17m/version_15/checkpoints/last.ckpt
```
This script calls the PyTorch Lightning Command Line Interface (CLI), more details can be found at [https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html). This CLI trains the model through configuration files that whereby the first configuration file can contain the standard/shared variables and the second contains model specific variables. The additional flags allows you to override existing variables and in this case with `--ckpt_path` allows us to continue training with an model that we had started to train.

For those interested our Slurm scripts can be found in the following folder [./slurm_runs/semantic_similarity/usas_variable_negatives](./slurm_runs/semantic_similarity/usas_variable_negatives).


### Evaluating on USAS data

We have reported on 5 test sets:
* Chinese
* English
* Welsh
* Finnish
* Irish

Of which the model evaluations including the statistics have been ran through the following Python notebook; [./notebooks/evaluation_dataset_visulisations/usas_evaluation_data.ipynb](./notebooks/evaluation_dataset_visulisations/usas_evaluation_data.ipynb) and the data statistics can be found in that notebook folder. This notebook does assume that the test data has been downloaded too [./data/usas_evaluation](./data/usas_evaluation) folder.


The Irish data has been processed whereby the details of this process can be found [here](https://github.com/UCREL/ciall-data?tab=readme-ov-file#generating-predictions-from-the-manually-tagged-data), once processed it is expected that the data is to be stored at [./data/usas_evaluation/icc_irish](./data/usas_evaluation/icc_irish), whereby we assume in that folder are the following folders; `predicted_sem_tag_tsv_CHKD`, and `sem_tag_tsv_CHKD`.



## Visulisations

### Training data heatmap

To generate the heatmap of the training data as well as the general dataset statistics of the training and validation data (it also outputs the test data statistics but this is test data we never used in the end), run the following:

``` bash
uv run python training_runs/usas_semantic_similarity/visualise_training_data.py english_mosaico 10 ./statistics_and_visualisations/general_dataset_statistics.tex ./statistics_and_visualisations/training_data_label_distribution_heatmap.svg -f Z99 -h
```

The command in essence will use 10 CPU for pre-processing the data, if it is needed, output the dataset statistics too `./statistics_and_visualisations/general_dataset_statistics.tex`, output the heatmap of the distribution of lables within the training data too `./statistics_and_visualisations/training_data_label_distribution_heatmap.svg` and will ensure that the `Z99` label is not included in the dataset (this label does not semantically mean anything and is not used during training).

You can get more help with this script by running:

``` bash
uv run python training_runs/usas_semantic_similarity/visualise_training_data.py --help
```

## General scripts

Note: none of these scripts are required to train or evaluate the USAS models.

### Automatically Translating the USAS tagset

``` bash
uv run ./scripts/translate_usas_mapper.py ./data/usas_mapper.yaml finnish ./data/finnish_usas_mapper.json
```

### Adding sentence breaks to the Chinese Torch evaluation data

``` bash
uv run ./scripts/python process_zh.py data/usas_evaluation/ToRCH2019_A26_chinese.csv data/usas_evaluation/ToRCH2019_A26_sentence_breaks_chinese.csv
```