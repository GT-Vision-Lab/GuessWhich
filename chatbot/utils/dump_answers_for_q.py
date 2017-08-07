import os
import sys
import json
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm
from unidecode import unidecode

random.seed(42)
np.random.seed(42)

def cleanNonAscii(text):
    try:
        text = text.decode('ascii');

    except:
        try: text = unicode(text, encoding = 'utf-8');
        except:
            pass
        text = unidecode(text);
    return text;

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-question', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading data...'
    data = json.load(open('../data/visdial_0.5/visdial_0.5_train.json', 'r'))

    print 'Parsing...'
    questions, answers = [], []
    for i in data:
        for j in range(10):
            questions.append(i['dialog'][j]['question'])
            answers.append(i['dialog'][j]['answer'])

    print 'Filtering...'
    res = []
    for i in range(len(questions)):
        if questions[i] == args.question:
            res.append(questions[i] + '? ' + answers[i])

    res = list(set(res))
    for i in res:
        print cleanNonAscii(i)
