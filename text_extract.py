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
    sentences = []
    for sample in dataset['data']:
        words = []
        for w in sample['words']:
            new_w = unicodedata.normalize('NFKD', unicode(w)).encode('ascii', 'ignore').lower()
            # R&B tokenization difference by open-sesame/SEMAFOR
            new_w.replace('&', 'n')
            if len(new_w) > 1 and new_w.endswith('.'):
                # this is an acronym / abbreviation that will cause mess in tokenization
                new_w = new_w[:-1]
            if not new_w:
                # add an unk if the word was only non-ascii chars. So will mantain the alignment with annotations
                new_w = 'unk'
            words.append(new_w)
        s = ' '.join(words)
        sentences.append(s)

    sentences = set(sentences)

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