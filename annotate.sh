#!/usr/bin/env bash

for DATASET in botcycle atis nlu-benchmark
do
    pushd open-sesame
    python -m sesame.targetid --mode predict --model_name fn1.7-pretrained-targetid --raw_input ../data/${DATASET}/sentences.txt
    python -m sesame.frameid --mode predict --model_name fn1.7-pretrained-frameid --raw_input logs/fn1.7-pretrained-targetid/predicted-targets.conll
    python -m sesame.argid --mode predict --model_name fn1.7-pretrained-argid --raw_input logs/fn1.7-pretrained-frameid/predicted-frames.conll
    popd

    cp open-sesame/logs/fn1.7-pretrained-argid/predicted-args.conll data/${DATASET}/predicted_opensesame.conll
done