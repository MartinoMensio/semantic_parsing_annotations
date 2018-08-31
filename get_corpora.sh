#!/usr/bin/env bash

mkdir -p data/botcycle
wget https://raw.githubusercontent.com/D2KLab/botcycle/master/nlu/data/wit_en/preprocessed/fold_train.json -O data/botcycle/source.json
mkdir -p data/atis
wget https://raw.githubusercontent.com/D2KLab/botcycle/master/nlu/data/atis/preprocessed/fold_train.json -O data/atis/source.json
mkdir -p data/nlu-benchmark
wget https://raw.githubusercontent.com/D2KLab/botcycle/master/nlu/data/nlu-benchmark/preprocessed/fold_train.json -O data/nlu-benchmark/source.json