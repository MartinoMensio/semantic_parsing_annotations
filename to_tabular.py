#!/usr/bin/env python

import os
import json
import plac

from pathlib2 import Path

import xml.etree.ElementTree as ET
import re

# TODO group by frame!!!

FRAMENET_LOCATION = Path('open-sesame/data/fndata-1.7')

def get_lu_and_iob(lu_indicator, iob_label):
    if lu_indicator == '_':
        return iob_label
    else:
        return 'LU+' + iob_label

def create_tabular(input_file, output_file, lu_by_frame):
    """Read the input file that has annotations grouped by sentence, and prduce a tabular format"""
    with open(str(input_file)) as f:
        content = json.load(f)

    lines = []

    for sentence in content.values():
        line = [''] + ['{}::{}'.format(annot['source'], annot['class']) for annot in sentence['interpretations']]
        lines.append(line)
        #print(sentence)
        for idx, w in enumerate(sentence['words']):
            line = [w] + [get_lu_and_iob(annot['lu_indicator'][idx], annot['IOB'][idx]) for annot in sentence['interpretations']]
            lines.append(line)
        # see if the LU is in the ones from FrameNet for the corresponding Frame
        gold_annots = [annot for annot in sentence['interpretations'] if annot['source'] == 'GOLD']
        if not gold_annots:
            print(sentence)
        else:
            gold_frame_annot = gold_annots[0]
            if gold_frame_annot['class'] in lu_by_frame:
                lines.append(['# LU_in_frame?'] + is_lu_in_frame(lu_by_frame[gold_frame_annot['class']], gold_frame_annot['lu_indicator']))
        lines.append([])

    with open(str(output_file), 'w') as f:
        for l in lines:
            f.write('\t'.join(l) + '\n')

def is_lu_in_frame(lu_for_frame, lu_indicators):
    simplified_lus = [lu.split('.')[0] for lu in lu_for_frame]
    result = [str(lu_ind in simplified_lus) for lu_ind in lu_indicators if lu_ind != '_']
    return result

def get_lu_by_frame(framenet_location):
    frames_location = framenet_location / 'frame'
    lu_by_frame_name = {}
    for el in frames_location.glob('*.xml'):
        if not el.is_file():
            break
        with open(str(el)) as f:
            xmlstring = f.read()
        # remove namespace that makes ET usage cumbersome
        xmlstring = re.sub(r'\sxmlns="[^"]+"', '', xmlstring, count = 1)
        root = ET.fromstring(xmlstring)
        lus = [lu.attrib['name'] for lu in root.findall('lexUnit')]
        lu_by_frame_name[el.name[:-4]] = lus
    #frame_data_by_name = {el.name[:-4]: ET.parse(str(el)).getroot() for el in frames_location.glob('*.xml') if el.is_file()}
    #lu_by_frame_name = {frame_name: frame_data.findall('lexUnit') for frame_name, frame_data in frame_data_by_name.items()}
    #print(lu_by_frame_name)
    return lu_by_frame_name

def main(dataset_name):
    dataset_folder = Path('data') / dataset_name
    lu_by_frame = get_lu_by_frame(FRAMENET_LOCATION)
    create_tabular(dataset_folder / 'compared_by_sentence.json', dataset_folder / 'tabular.tsv', lu_by_frame)

if __name__ == '__main__':
    main('botcycle')
    main('atis')
    main('huric')
    main('nlu-benchmark')