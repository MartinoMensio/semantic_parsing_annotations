#!/usr/bin/env python

from __future__ import print_function

import json
import os
import itertools
import numpy as np
import unicodedata

from IPython.display import HTML, display

import matplotlib
import matplotlib.pyplot as plt

from pathlib2 import Path

def read_gold(path):
    """Reads the gold annotations in the format used in botcycle (https://github.com/D2KLab/botcycle/tree/master/nlu/data)"""
    with open(str(path)) as f:
        content = json.load(f)

    # avoid non-ascii
    for s in content['data']:
        words = []
        for w in s['words']:
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
        s['words'] = words
    result = [
        {
            'text': ' '.join(s['words']),
            'words': s['words'],
            'interpretation': {
                'source': 'GOLD',
                # a single frame per sentence in the gold format
                'class': s['intent'],
                # this is not stored in the gold format
                'lu_indicator': get_lu_indicator_from_slots(s),
                # the IOB for FrameElements
                'IOB': s['slots']
            },
            'id': int(s.get('id', idx))
        } for idx, s in enumerate(content['data'])
    ]
    # remove duplicates!!! Why wit.ai allowed that??
    d = {el['text']: el for el in result}
    result = list(d.values())

    return result, np.array(content['meta']['intent_types'])

def get_lu_indicator_from_slots(sample):
    result = ['_' for _ in sample['slots']]
    lu_indexes = sample.get('lexical_unit_ids', [])
    for lu in lu_indexes:
        index = lu - sample['start_token_id']
        result[index] = sample['words'][index]
    return result

def read_conll(path, gold):
    """ """
    with open(str(path)) as f:
        content = f.read().strip()

    # gold is there to fix some issues of open-sesame
    gold_samples_by_id = {s['id']: s for s in gold}
    #for k,v in gold_samples_by_id.items():
    #    print(k, v['words'])

    samples = [
        [
            list(t.split('\t'))
            for t in s.split('\n')
        ]
        for s in content.split('\n\n')
    ]
    # open-sesame does a mess: replaces some words with UNK. Let's restore the proper word
    for s in samples:
        broken = []
        sentence = ''
        for idx, w in enumerate(s):
            sentence += ' ' + w[1]
            if w[1] == 'unk':
                broken.append(idx)
                #gold_sample = gold_samples_by_id[int(w[6])]
                #print(idx, s, gold_sample)
                #w[1] = gold_sample['words'][idx]
        if broken:
            indexes_to_compare = set(range(len(s))).difference(set(broken))
            print(sentence, 'broken', broken, 'to_compare', indexes_to_compare)
            matching_gold = [gs for gs in gold if all([s[idx][1] == gs['words'][idx] for idx in range(len(gs['words'])) if idx in indexes_to_compare]) and len(s) == len(gs['words'])]
            print(len(matching_gold), matching_gold)
            for idx in broken:
                s[idx][1] = matching_gold[0]['words'][idx]
    result = [
        {
            'text': ' '.join([w[1] for w in s]),
            'words': [w[1] for w in s],
            'interpretation': {
                'source': 'open-sesame',
                # a single frame per sentence in the gold format
                'class': next(w[13] for w in s if w[13] != '_'),
                # this is not stored in the gold format
                'lu_indicator': [w[-3] for w in s],
                # the IOB for FrameElements
                'IOB': [w[-1] for w in s]
            }
        } for s in samples
    ]
    classes = set([s['interpretation']['class'] for s in result])
    return result, classes

def read_semafor(path):
    with open(str(path)) as f:
        content = f.readlines()

    annots = []
    frame_types = []

    for l in content:
        source_annot = json.loads(l)
        words = source_annot['tokens']
        for f in source_annot['frames']:
            frame_type = f['target']['name']
            frame_types.append(frame_type)
            iob = ['O'] * len(words)
            for annotation_set in f['annotationSets']:
                #print(annotation_set)
                for fe in annotation_set['frameElements']:
                    fe_name = fe['name']
                    fe_tokens_id = range(fe['spans'][0]['start'], fe['spans'][0]['end'])
                    #print(fe_tokens_id)
                    iob[fe_tokens_id[0]] = 'B-{}'.format(fe_name)
                    for i in fe_tokens_id[1:]:
                        iob[i] = 'I-{}'.format(fe_name)
            #print(iob)
            annots.append({
                'text': ' '.join(words),
                'words': words,
                'interpretation': {
                    'source': 'SEMAFOR',
                    'class': frame_type,
                    'lu_indicator': [f['target']['spans'][0]['text'] if idx >= f['target']['spans'][0]['start'] and idx < f['target']['spans'][0]['end'] else '_' for idx, w in enumerate(words)],
                    'IOB': iob
                }
            })


    return annots, set(frame_types)

def by_text(samples):
    """
    Given an iterable of samples, returns a dict {sentence: list(interpretations)}
    where an interpretation is something like {'frame': str, 'iob': list(IOB)}
    """
    key_fn = lambda x: x['text'].lower()
    result = {
        k: {
            'words': g[0]['words'],
            'interpretations': [s['interpretation'] for s in g],
        }
        # after grouping by, the map in brackets below is built to pass from an iterator to a list so that g[0] does not consume the first iteration removing one interpretation
        for k, g in {k: list(g) for k,g in itertools.groupby(sorted(samples, key=key_fn), key_fn)}.items()
    }
    #print(result.keys())
    return result

def get_reverse_mapping(list_of_values):
    """Returns a dict with {value: index}"""
    return {el[1]: el[0] for el in enumerate(list_of_values)}

def get_alignment_matrix(interpretations_by_text, in_types, out_types, filter_source='open-sesame'):
    # build maps from names to indexes for intents and frame names. matrix is indexed [in_idx, out_idx], where in_idx and out_idx can be obtained by looking at the other two returned dicts
    in_types_lookup = get_reverse_mapping(in_types)
    out_types_lookup = get_reverse_mapping(out_types)
    matrix = np.zeros((len(in_types_lookup), len(out_types_lookup)))
    for text, annots in interpretations_by_text.items():
        ins = []
        outs = []
        for interp in annots['interpretations']:
            if interp['source'] == 'GOLD':
                ins.append(in_types_lookup[interp['class']])
            elif interp['source'] == filter_source:
                outs.append(out_types_lookup[interp['class']])
        points = list(itertools.product(ins, outs))
        for in_, out_ in points:
            matrix[in_, out_] += 1

    return matrix, in_types_lookup, out_types_lookup

def print_matrix(matrix, in_types, out_types, ax):
    # normalize by rows
    row_sums = matrix.sum(axis=1)
    new_matrix = matrix / row_sums[:, np.newaxis]
    new_matrix = np.transpose(new_matrix)
    #print(new_matrix)

    # fig, ax = plt.subplots(figsize=(len(in_types)/3, len(out_types)/3)) #figsize=(15, 5)
    cmap = plt.get_cmap('Greens')
    im = ax.imshow(new_matrix, cmap=cmap)

    # labels
    ax.set_xticks(np.arange(len(in_types)))
    ax.set_yticks(np.arange(len(out_types)))
    ax.set_xticklabels(in_types, rotation=90)
    ax.set_yticklabels(out_types)

    #plt.show()

def save_to_file(data, path):
    with open(str(path), 'w') as f:
        json.dump(data, f, indent=2)

def read_annotations_and_group(dataset_name):
    data_path = Path('data') / dataset_name
    gold, intent_types = read_gold(data_path / 'source.json')
    open_sesame, frame_types_open_sesame = read_conll(data_path / 'predicted_opensesame.conll', gold)
    semafor, frame_types_semafor = read_semafor(data_path / 'predicted_semafor.json')
    all_frame_types = frame_types_open_sesame.union(frame_types_semafor)
    grouped = by_text(gold + open_sesame + semafor)
    # TODO write one time to file, and then avoid doing every time all the grouping by
    #save_to_file(grouped, data_path / 'compared_by_sentence.json')
    return grouped, intent_types, np.array(sorted(all_frame_types))

def print_annotations(grouped_samples, max_display=None):
    for idx, sentence in enumerate(grouped_samples.values()):
        if idx > max_display:
            break
        rows = {}
        rows['WORDS'] = sentence['words']
        for annot in sentence['interpretations']:
            rows['{}_{}'.format(annot['source'], annot['class'])] = annot['IOB']
        display_sequences(rows.keys(), rows.values())

def display_sequences(row_names, sequences):
    html_str = '<table><tr>{}</tr></table>'.format(
        '</tr><tr>'.join(
            '<td><b>{}</b></td>'.format(row_name) +
            ''.join(['<td style="background-color: {};">{}</td>'.format(get_color(value), value) for value in row])
            for row_name, row in zip(row_names, sequences)
        )
    )
    display(HTML(html_str))

def get_color(value):
    if value == 'O':
        return 'rgb(255,255,255)'
    else:
        return 'rgb(0,255,0)'

def main(dataset_name='botcycle'):
    grouped, intent_types, frame_types = read_annotations_and_group(dataset_name)
    # write only in the main
    save_to_file(grouped, Path('data') / dataset_name / 'compared_by_sentence.json')

    matrix_os, intents_lookup_os, frames_lookup_os = get_alignment_matrix(grouped, intent_types, frame_types)
    matrix_se, intents_lookup_se, frames_lookup_se = get_alignment_matrix(grouped, intent_types, frame_types, 'SEMAFOR')
    print_best_n_for_each_gold_type(matrix_os, intent_types, frame_types, 'open-sesame', 3)
    print_best_n_for_each_gold_type(matrix_se, intent_types, frame_types, 'semafor', 3)
    # and also display the matrix
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(24, 16)) #figsize=(15, 5)
    print_matrix(matrix_os, intent_types, frame_types, ax1)
    print_matrix(matrix_se, intent_types, frame_types, ax2)
    plt.tight_layout()
    plt.show()

def print_best_n_for_each_gold_type(matrix, intent_types, frame_types, origin='alingnment', n=3):
    print(origin, ':')
    for idx, row in enumerate(matrix):
        # get the indexes of the top elements in the row (the most counted frames)
        best_n_idx = row.argsort()[-n:][::-1]
        # get a list of (name, count) of the frames
        best_n_with_scores = list(zip(frame_types[best_n_idx], row[best_n_idx]))
        print(intent_types[idx], '-->', best_n_with_scores)


if __name__ == '__main__':
    main('botcycle')
    main('atis')
    main('huric')
    main('nlu-benchmark')