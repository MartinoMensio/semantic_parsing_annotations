## Installing

Clone all the submodules: `git submodule update --init --recursive`

required python2

- create a venv
- install required things by using `install.sh`

## Process

1. download the models for open-sesame and semafor: run `get_models.sh`
2. get the corpora: run `get_corpora.sh`
3. extract the sentences from the corpora: run `text_extract.py`
4. annotate with the prebuilt systems: run `annotate.sh`
5. group the annotations on the same sentences: run `aligner.py`
6. extract the TSV files for easier comparison: run `to_tabular.py`


## TODOs

automate the framenet source files download?? need:
- FrameNet: `data/fndata-1.7/luIndex.xml`
- preprocessed conll: `data/neural/fn1.7/fn1.7.fulltext.train.syntaxnet.conll`
- embeddings: `glove.6B.100d.txt`