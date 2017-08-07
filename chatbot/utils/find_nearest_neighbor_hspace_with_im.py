# python find_nearest_neighbor_hspace.py -json_input ../data/visdial_0.5/visdial_0.5_test.json -json_output working/qoptions_cap_qtm1_atm1_sample25k_visdial_0.5_test.json

import sys
import h5py
import spacy
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

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
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading json...'
    data = json.load(open(args.json_input, 'r'))

    print 'Loading spacy...'
    nlp = spacy.en.English()

    print 'Loading image features...'
    hf = h5py.File('../data/visdial_0.5/data_img.h5')
    fc7_feat = hf.get('images_%s' % args.split)
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]
    inv_fc7_map = {i: fc7_map.index(i) for i in fc7_map}

    print 'Normalizing gt fc7...'
    fc7_feat_norm = normalize(fc7_feat, axis=1, norm='l2')

    print 'Building set of histories and follow-up questions...'
    imgs, imgs_ind, caps, ques, ans = [], [], [], [], []
    #  max_len = 5000
    #  for i in tqdm(range(max_len)):
    for i in tqdm(range(len(data))):
        imgs.append(data[i]['image_id'])
        imgs_ind.append(inv_fc7_map[str(data[i]['image_id'])])
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
    #  feat = np.zeros((len(ques), cap_vectors.shape[1] + ques_vectors.shape[1]))
    feat = np.zeros((len(ques), cap_vectors.shape[1] + ques_vectors.shape[1] + fc7_feat_norm.shape[1]))
    print 'Feat shape'
    print feat.shape
    feat_str = []
    for i in tqdm(range(len(cap_vectors))):
        for j in range(10):
            if j == 0:
                feat[i*10][:300] = cap_vectors[i]
                feat[i*10][600:] = fc7_feat_norm[imgs_ind[i]]
                feat_str.append(caps[i])
            else:
                # caption + q_t-1 + a_t-1
                #  feat[i*10+j] = np.concatenate((cap_vectors[i],
                                               #  ques_vectors[i*10+j-1],
                                               #  ans_vectors[i*10+j-1]))
                # caption + qa_t-1_mean
                feat[i*10+j] = np.concatenate((cap_vectors[i],
                                            np.mean((ques_vectors[i*10+j-1], ans_vectors[i*10+j-1]), axis=0), fc7_feat_norm[imgs_ind[i]]))
                # caption + qa_mean
                #  ques_mean = np.mean((ques_vectors[i*10:i*10+j]), axis=0)
                #  ans_mean = np.mean((ans_vectors[i*10:i*10+j]), axis=0)
                #  feat[i*10+j] = np.concatenate((cap_vectors[i],
                                               #  np.mean((ques_mean, ans_mean), axis=0)))
                # caption + q_mean + a_mean
                #  feat[i*10+j] = np.concatenate((cap_vectors[i], ques_mean, ans_mean))
                feat_str.append(caps[i] + ' ' + ques[i*10+j-1] + '? ' + ans[i*10+j-1])

    print 'Computing neighbors...'
    sample_size = 10000
    #  indices = np.random.choice(feat.shape[0], sample_size, replace=False)
    neighbors = [[] for _ in range(feat.shape[0])]
    for i in tqdm(range(feat.shape[0])):
    #  for i in tqdm(range(10)):
        #  if i % 5 == 0:
            #  indices = np.random.choice(feat.shape[0], sample_size, replace=False)
        #  while i in indices:
        indices = np.random.choice(feat.shape[0], sample_size, replace=False)
        feat_sub = feat[indices]
        nearest_neighbors = get_nearest_neighbors_query(feat_sub, feat[i], 500)
        neighbors[i] = indices[nearest_neighbors]
        # Save every 100 conversations
        if i%10 == 0 and (i//10) % 100 == 0:
            print 'Compiling results...'
            result = {}
            for i in tqdm(range(i // 10)):
                result[imgs[i]] = [{'gt_rank': -1, 'q_options': []} for _ in range(10)]
                for j in range(10):
                    for k in range(200):
                        if ques[neighbors[i*10+j][k]] not in result[imgs[i]][j]['q_options']:
                            result[imgs[i]][j]['q_options'].append(ques[neighbors[i*10+j][k]])
                    result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]
                    if ques[i*10+j] in result[imgs[i]][j]['q_options']:
                        result[imgs[i]][j]['gt_rank'] = result[imgs[i]][j]['q_options'].index(ques[i*10+j])
                    else:
                        result[imgs[i]][j]['q_options'] = [ques[i*10+j]] + result[imgs[i]][j]['q_options']
                        result[imgs[i]][j]['gt_rank'] = 0
                        result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]
            print 'Saving to %s' % args.json_output
            json.dump(result, open(args.json_output, 'w'))

    #  for i in range(100):
        #  if i%10 == 0:
            #  print '<br><img src="https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg" style="max-width:200px;"><br>' % imgs[i//10]
        #  print '<b>Round %d </b><br>' % (i%10+1)
        #  print 'History: %s <br>' % cleanNonAscii(feat_str[i])
        #  print 'GT next question: %s <br>' % cleanNonAscii(ques[i])
        #  print '<b>Nearest neighbors </b><br>'
        #  id_to_print = 1
        #  count = 0
        #  while count < 5:
            #  if i // 10 == neighbors[i][id_to_print] // 10:
                #  id_to_print += 1
            #  else:
                #  print '<img src="https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg" style="max-width:100px;">' % int(fc7_map[imgs_ind[neighbors[i][id_to_print] // 10]])
                #  print cleanNonAscii(feat_str[neighbors[i][id_to_print]]) + '<br>'
                #  print 'Next question: %s <br>' % cleanNonAscii(ques[neighbors[i][id_to_print]])
                #  id_to_print += 1
                #  count += 1

    print 'Compiling results...'
    result = {}
    for i in tqdm(range(len(imgs))):
    #  for i in tqdm(range(1)):
        result[imgs[i]] = [{'gt_rank': -1, 'q_options': []} for _ in range(10)]
        for j in range(10):
            for k in range(200):
                if ques[neighbors[i*10+j][k]] not in result[imgs[i]][j]['q_options']:
                    result[imgs[i]][j]['q_options'].append(ques[neighbors[i*10+j][k]])
            result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]
            if ques[i*10+j] in result[imgs[i]][j]['q_options']:
                result[imgs[i]][j]['gt_rank'] = result[imgs[i]][j]['q_options'].index(ques[i*10+j])
            else:
                result[imgs[i]][j]['q_options'] = [ques[i*10+j]] + result[imgs[i]][j]['q_options']
                result[imgs[i]][j]['gt_rank'] = 0
                result[imgs[i]][j]['q_options'] = result[imgs[i]][j]['q_options'][:100]


    print 'Saving to %s' % args.json_output
    json.dump(result, open(args.json_output, 'w'))



