#!/usr/bin/env python

"""This module performs the parsing/SRL by interrogating the demo of allenNLP"""

import json
import os
import plac
import requests
from tqdm import tqdm

def main(input_file, output_file):
    with open(input_file) as f:
        content = f.read()

    sentences = content.split('\n')

    results = []
    for s in tqdm(sentences):
        response = requests.post('http://demo.allennlp.org/predict/semantic-role-labeling', json={'sentence': s})
        annotations = response.json()['verbs']
        #print(annotations)
        for annot in annotations:
            results.append({
                'words': s.split(),
                'slots': annot['tags'],
                'intent': 'verb'
            })

    result = {
        'data': results,
        'meta': {'intent_types': ['verb']}
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    plac.call(main)