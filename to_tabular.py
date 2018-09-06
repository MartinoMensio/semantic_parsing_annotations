#!/usr/bin/env python

import os
import json
import plac

# TODO group by frame!!!

def get_lu_and_iob(lu_indicator, iob_label):
    if lu_indicator == '_':
        return iob_label
    else:
        return 'LU+' + iob_label

def main(input_file, output_file):
    """Read the input file that has annotations grouped by sentence, and prduce a tabular format"""
    with open(input_file) as f:
        content = json.load(f)

    lines = []

    for sentence in content.values():
        line = [''] + ['{}::{}'.format(annot['source'], annot['class']) for annot in sentence['interpretations']]
        lines.append(line)
        for idx, w in enumerate(sentence['words']):
            line = [w] + [get_lu_and_iob(annot['lu_indicator'][idx], annot['IOB'][idx]) for annot in sentence['interpretations']]
            lines.append(line)
        lines.append([])

    with open(output_file, 'w') as f:
        for l in lines:
            f.write('\t'.join(l) + '\n')

if __name__ == '__main__':
    plac.call(main)