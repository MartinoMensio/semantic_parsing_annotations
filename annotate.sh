#!/usr/bin/env bash

for DATASET in botcycle atis nlu-benchmark huric
do
    pushd open-sesame
    python -m sesame.targetid --mode predict --model_name fn1.7-pretrained-targetid --raw_input ../data/${DATASET}/sentences.txt
    python -m sesame.frameid --mode predict --model_name fn1.7-pretrained-frameid --raw_input logs/fn1.7-pretrained-targetid/predicted-targets.conll
    python -m sesame.argid --mode predict --model_name fn1.7-pretrained-argid --raw_input logs/fn1.7-pretrained-frameid/predicted-frames.conll
    popd

    cp open-sesame/logs/fn1.7-pretrained-argid/predicted-args.conll data/${DATASET}/predicted_opensesame.conll

    # also SEMAFOR
    rm -f data/${DATASET}/predicted_semafor.json
    semafor/bin/runSemafor.sh data/${DATASET}/sentences.txt ${PWD}/data/${DATASET}/predicted_semafor.json 4

    # also allennlp from http://demo.allennlp.org/predict/semantic-role-labeling
    #./allennlp.py data/${DATASET}/sentences.txt data/${DATASET}/predicted_allennlp.json
done