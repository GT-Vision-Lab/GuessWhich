import os
import json
import random
from tqdm import tqdm

import numpy as np
from unidecode import unidecode
import spacy

random.seed(32)
np.random.seed(32)

def sentence_to_vec(sentence, nlp, mode=0):
    """
    * Returns Spacy vector for input sentence
    
    INPUT
    sentence -- unicode string

    OUTPUT
    mode == 0:
        vector(word 1) + vector(word 2) + vector(word 3) + vector(remaining words)
    mode == 1:
        vector(sentence)
    """

    if mode == 0:
        vec = np.zeros((1200))
        tokens = nlp(sentence)

        for i in xrange(min(3, len(tokens))):
            vec[300*i : 300*(i+1)] = tokens[i].vector

        if len(tokens) >= 4:
            for i in xrange(len(tokens)-3):
                vec[900:] += tokens[i+3].vector

        return vec

    elif mode == 1:
        tokens = nlp(sentence)
        vec = tokens.vector

        return vec

def multi_sentence_to_vec(lst, nlp, mode=0):
    """
    * Returns Spacy vectors for list of sentences
    * Uses sentence_to_vec internally

    INPUT
    lst -- list of unicode strings
    """

    if mode == 0:
        vec = np.zeros((len(lst), 1200))
    else:
        vec = np.zeros((len(lst), 300))
    for i in tqdm(range(len(lst))):
        vec[i,:] = sentence_to_vec(lst[i], nlp, mode)

    return vec

def get_nearest_neighbors(vec, num_nearest):
    """
    * Returns nearest neighbor vectors (by L2 distance)
    * Faster if vec ONLY has unique vectors
    
    INPUT
    vec -- N x 1200 array of vectors

    OUTPUT
    N x num_nearest list of nearest neighbor indices
    """

    lst = [[] for _ in range(vec.shape[0])]
    for i in tqdm(range(vec.shape[0])):
        dist = (vec - vec[i]) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        idx = np.argsort(dist)
        lst[i] = idx[1:1+num_nearest]

    return lst

def get_nearest_neighbors_query(feat, vec, num_nearest):
    """
    * Returns nearest neighbor vectors (by L2 distance)
    
    INPUT
    feat -- N x D array of vectors
    vec -- query vector

    OUTPUT
    1 x n vector of indices sorted by distance
    """

    lst = []
    dist = (feat - vec) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    idx = np.argsort(dist)
    lst = idx[:num_nearest]

    return lst

def get_nearest_neighbors_query_with_rank(feat, vec, num_nearest, gt_index):
    """
    * Returns nearest neighbor vectors (by L2 distance)
    * Returns rank of GT

    INPUT
    same as `get_nearest_neighbors_query()`, and
    GT index of image in feat

    OUTPUT
    1 x n vector of indices sorted by distance, and
    Rank of GT image in sorted distance
    """

    lst = []
    dist = (feat - vec) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    idx = np.argsort(dist)
    lst = idx[:num_nearest]

    rank = np.where(idx == gt_index)[0][0]
    return (lst, rank, dist[gt_index])

def cleanNonAscii(text):
    try:
        text = text.decode('ascii');

    except:
        try: text = unicode(text, encoding = 'utf-8');
        except:
            pass
        text = unidecode(text);
    return text;

def printEval(ranks):
    R1 = np.sum(np.array(ranks)==1) / float(len(ranks))
    R5 =  np.sum(np.array(ranks)<=5) / float(len(ranks))
    R10 = np.sum(np.array(ranks)<=10) / float(len(ranks))
    ave = np.sum(np.array(ranks)) / float(len(ranks))
    mrr = np.sum(1/(np.array(ranks, dtype='float'))) / float(len(ranks))
    return (R1, R5, R10, ave, mrr)
