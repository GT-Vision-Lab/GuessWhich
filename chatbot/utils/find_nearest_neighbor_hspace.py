# python find_nearest_neighbor_hspace.py -json_input ../data/visdial_0.5/visdial_0.5_test.json -json_output working/qoptions_cap_qtm1_atm1_sample25k_visdial_0.5_test.json

import spacy
import json
import random
import argparse
import numpy as np
from tqdm import tqdm

from utils import *

random.seed(42)
np.random.seed(42)

if __name__ == '__main__':
    """
    build a set of unique histories (pairs of QA)
    and follow-up questions
    maintain mapping indices to orig list
    compute nearest neighbors for set
    map back to space, and return follow-up
    questions to nearest neighbor histories
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-json_output', type=str)
    parser.add_argument('-num_nearest', type=int, default=100)
    args = parser.parse_args()

    print 'Loading json...'
    data = json.load(open(args.json_input, 'r'))

    print 'Loading spacy...'
    nlp = spacy.en.English()

    print 'Building set of histories and follow-up questions...'
    imgs, caps, ques, ans = [], [], [], []
    #  max_len = 5000
    #  for i in tqdm(range(max_len)):
    for i in tqdm(range(len(data))):
        imgs.append(data[i]['image_id'])
        caps.append(data[i]['caption'])
        for j in range(10):
            q = data[i]['dialog'][j]['question']
            a = data[i]['dialog'][j]['answer']
            ques.append(q)
            ans.append(a)
    print 'Questions: %d' % len(ques)

    print 'Computing vectors...'
    mode = 1 # mode 1 for computing avg word2vec of entire sentence
    cap_vectors = multi_sentence_to_vec(caps, nlp, mode)
    ques_vectors = multi_sentence_to_vec(ques, nlp, mode)
    ans_vectors = multi_sentence_to_vec(ans, nlp, mode)

    print 'Combining vectors...'
    #  feat = np.zeros((len(ques), cap_vectors.shape[1]
                                #  + ques_vectors.shape[1]
                                #  + ans_vectors.shape[1]))
    feat = np.zeros((len(ques), cap_vectors.shape[1] + ques_vectors.shape[1]))
    print 'Feat shape'
    print feat.shape
    feat_str = []
    for i in tqdm(range(len(cap_vectors))):
        for j in range(10):
            if j == 0:
                feat[i*10][:300] = cap_vectors[i]
                feat_str.append(caps[i])
            else:
                # caption + q_t-1 + a_t-1
                #  feat[i*10+j] = np.concatenate((cap_vectors[i],
                                               #  ques_vectors[i*10+j-1],
                                               #  ans_vectors[i*10+j-1]))
                # caption + qa_t-1_mean
                feat[i*10+j] = np.concatenate((cap_vectors[i],
                                            np.mean((ques_vectors[i*10+j-1], ans_vectors[i*10+j-1]), axis=0)))
                # caption + qa_mean
                #  ques_mean = np.mean((ques_vectors[i*10:i*10+j]), axis=0)
                #  ans_mean = np.mean((ans_vectors[i*10:i*10+j]), axis=0)
                #  feat[i*10+j] = np.concatenate((cap_vectors[i],
                                               #  np.mean((ques_mean, ans_mean), axis=0)))
                # caption + q_mean + a_mean
                #  feat[i*10+j] = np.concatenate((cap_vectors[i], ques_mean, ans_mean))
                feat_str.append(caps[i] + ' ' + ques[i*10+j-1] + '? ' + ans[i*10+j-1])

    print 'Computing neighbors...'
    sample_size = 96280
    indices = np.random.choice(feat.shape[0], sample_size, replace=False)
    neighbors = [[] for _ in range(feat.shape[0])]
    #  for i in tqdm(range(feat.shape[0])):
    for i in tqdm(range(100)):
        #  if i % 5 == 0:
            #  indices = np.random.choice(feat.shape[0], sample_size, replace=False)
        #  while i in indices:
            #  indices = np.random.choice(feat.shape[0], sample_size, replace=False)
        feat_sub = feat[indices]
        nearest_neighbors = get_nearest_neighbors_query(feat_sub, feat[i], 500)
        neighbors[i] = indices[nearest_neighbors]

    for i in range(100):
        if i%10 == 0:
            print ''
        print '# Round %d' % (i%10+1)
        print '## History: %s' % cleanNonAscii(feat_str[i])
        print '## GT next question: %s' % cleanNonAscii(ques[i])
        print '## Nearest neighbors'
        id_to_print = 1
        count = 0
        while count < 5:
            if i // 10 == neighbors[i][id_to_print] // 10:
                id_to_print += 1
            else:
                print '### ' + cleanNonAscii(feat_str[neighbors[i][id_to_print]])
                print '### Next question: %s' % cleanNonAscii(ques[neighbors[i][id_to_print]])
                id_to_print += 1
                count += 1

    #  print 'Compiling results...'
    #  result = {}
    #  for i in tqdm(range(len(imgs))):
    #  #  for i in tqdm(range(10)):
        #  result[imgs[i]] = [{'gt_rank': -1, 'q_options': []} for _ in range(10)]
        #  for j in range(10):
            #  for k in range(200):
                #  if ques[neighbors[i*10+j][k]] not in result[imgs[i]][j]['q_options']:
                    #  result[imgs[i]][j]['q_options'].append(ques[neighbors[i*10+j][k]])
            #  result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]
            #  if ques[i*10+j] in result[imgs[i]][j]['q_options']:
                #  result[imgs[i]][j]['gt_rank'] = result[imgs[i]][j]['q_options'].index(ques[i*10+j])
            #  else:
                #  result[imgs[i]][j]['q_options'] = [ques[i*10+j]] + result[imgs[i]][j]['q_options']
                #  result[imgs[i]][j]['gt_rank'] = 0
                #  result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]


    #  print 'Saving to %s' % args.json_output
    #  json.dump(result, open(args.json_output, 'w'))



