#!/usr/bin/env python

import json
import os
import plac

from pathlib2 import Path
import unicodedata

DATA_LOCATION = 'data'

def load_source(dataset_name, file_name='source.json'):
    loc = Path(DATA_LOCATION) / dataset_name / file_name
    with open(str(loc)) as f:
        dataset = json.load(f)

    return dataset

def dataset_to_sents(dataset):
    sentences = [' '.join(sample['words']) for sample in dataset['data']]

    # remove non-ascii
    sentences = [unicodedata.normalize('NFKD', unicode(s)).encode('ascii', 'ignore') for s in sentences]

    return sentences

def write_sentences(sentences, dataset_name, file_name='sentences.txt'):
    loc = Path(DATA_LOCATION) / dataset_name / file_name
    with open(str(loc), 'w') as f:
        f.write('\n'.join(sentences))

def main(dataset_name='botcycle'):
    dataset = load_source(dataset_name)
    sentences = dataset_to_sents(dataset)
    write_sentences(sentences, dataset_name)


if __name__ == '__main__':
    for dataset in ['botcycle', 'atis', 'nlu-benchmark', 'huric']:
        plac.call(main, [dataset])