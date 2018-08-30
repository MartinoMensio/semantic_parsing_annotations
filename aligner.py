#!/usr/bin/env python

import json
import os
import itertools
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from pathlib2 import Path

def read_gold(path):
    """Reads the gold annotations in the format used in botcycle (https://github.com/D2KLab/botcycle/tree/master/nlu/data)"""
    with open(str(path)) as f:
        content = json.load(f)
    result = [
        {
            'text': ' '.join(s['words']).lower(),
            'words': s['words'],
            'interpretation': {
                'source': 'GOLD',
                # a single frame per sentence in the gold format
                'class': s['intent'],
                # this is not stored in the gold format
                'lu_index': -1,
                # the IOB for FrameElements
                'IOB': s['slots']
            }
        } for s in content['data']
    ]
    # TODO remove duplicates!!! Why wit.ai allowed that??
    return result, content['meta']['intent_types']

def read_conll(path):
    """ """
    with open(str(path)) as f:
        content = f.read().strip()

    samples = [
        [
            list(t.split('\t'))
            for t in s.split('\n')
        ]
        for s in content.split('\n\n')
    ]

    result = [
        {
            'text': ' '.join([w[1] for w in s]),
            'words': [w[1] for w in s],
            'interpretation': {
                'source': 'open-sesame',
                # a single frame per sentence in the gold format
                'class': next(w[13] for w in s if w[13] != '_'),
                # this is not stored in the gold format
                'lu_index': 'TODO',
                # the IOB for FrameElements
                'IOB': [w[-1] for w in s]
            }
        } for s in samples
    ]
    classes = sorted(set([s['interpretation']['class'] for s in result]))
    return result, classes

def by_text(samples):
    """
    Given an iterable of samples, returns a dict {sentence: list(interpretations)}
    where an interpretation is something like {'frame': str, 'iob': list(IOB)}
    """
    key_fn = lambda x: x['text']
    result = {
        g[0]: {
            'words': next(g[1])['words'],
            'interpretations': [s['interpretation'] for s in g[1]],
        }
        for g in itertools.groupby(sorted(samples, key=key_fn), key_fn)
    }
    return result

def frame_mappings(interpretations_by_text, in_types, out_types):
    # build maps from names to indexes for intents and frame names
    in_types_lookup = {el[1]: el[0] for el in enumerate(in_types)}
    out_types_lookup = {el[1]: el[0] for el in enumerate(out_types)}
    matrix = np.zeros((len(in_types_lookup), len(out_types_lookup)))
    for text, annots in interpretations_by_text.items():
        ins = []
        outs = []
        for interp in annots['interpretations']:
            if interp['source'] == 'GOLD':
                ins.append(in_types_lookup[interp['class']])
            elif interp['source'] == 'open-sesame':
                outs.append(out_types_lookup[interp['class']])
        points = list(itertools.product(ins, outs))
        for in_, out_ in points:
            matrix[in_, out_] += 1

    return matrix

def print_matrix(matrix, in_types, out_types):
    # normalize by rows
    row_sums = matrix.sum(axis=1)
    new_matrix = matrix / row_sums[:, np.newaxis]
    #print(new_matrix)

    fig, ax = plt.subplots(figsize=(15, 5))
    cmap = plt.get_cmap('Greens')
    im = ax.imshow(new_matrix, cmap=cmap)

    # labels
    ax.set_xticks(np.arange(len(out_types)))
    ax.set_yticks(np.arange(len(in_types)))
    ax.set_xticklabels(out_types, rotation=90)
    ax.set_yticklabels(in_types)

    plt.show()



def main():
    data_path = Path('data') / 'botcycle'
    gold, intent_types = read_gold(data_path / 'source.json')
    open_sesame, frame_types = read_conll(data_path / 'predicted_opensesame.conll')
    all = by_text(gold + open_sesame)
    matrix = frame_mappings(all, intent_types, frame_types)
    print_matrix(matrix, intent_types, frame_types)

if __name__ == '__main__':
    main()