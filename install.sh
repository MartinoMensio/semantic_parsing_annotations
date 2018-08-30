#!/usr/bin/env bash

# install the requirements

pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger wordnet
