# experimental-wsd
Experiments in WSD



## Data

To download all of the training and evaluation data to [./data](./data) listed below run the following:

``` bash
cd data
bash download_data.sh
```

For WordNet 3.0 we can use the [Python library wn](https://github.com/goodmami/wn) and to be compatible with previous work we will use the [Princeton WordNet 3.0](https://wordnet.princeton.edu/) and [NLTK](https://www.nltk.org/howto/wordnet.html) version which is [OMW English Wordnet based on WordNet 3.0](https://github.com/omwn/omw-data).

### English WSD

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

### Multilingual WSD

This has come from [Pasini et al. 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17609) (website of the dataset [https://sapienzanlp.github.io/xl-wsd/](https://sapienzanlp.github.io/xl-wsd/)) it downloads the following training and test datasets too [./data/xl-wsd/](./data/xl-wsd/)

### Hard datasets

To be added, but this will come from the following paper [Maru et al. 2022](https://aclanthology.org/2022.acl-long.324/) whereby the data can be found on the related [GitHub](https://github.com/SapienzaNLP/wsd-hard-benchmark).