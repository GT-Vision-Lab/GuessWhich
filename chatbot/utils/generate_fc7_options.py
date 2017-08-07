import json
import h5py
import argparse
from tqdm import tqdm
from random import randint
from sklearn.preprocessing import normalize

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_output', type=str)
    parser.add_argument('-num_nearest', type=int, default=100)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading ground truth hdf5...'
    hf = h5py.File('../data/visdial_0.5/data_img.h5')
    fc7_feat = hf.get('images_%s' % args.split)
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]

    print 'Normalizing gt fc7...'
    fc7_feat_norm = normalize(fc7_feat, axis=1, norm='l2')

    print 'Computing neighbors...'
    neighbors = [[] for _ in range(fc7_feat.shape[0])]
    #  for i in tqdm(range(100)):
    for i in tqdm(range(fc7_feat.shape[0])):
        nearest_neighbors = get_nearest_neighbors_query(fc7_feat_norm, fc7_feat_norm[i], args.num_nearest)
        for j in range(args.num_nearest):
            if j < args.num_nearest//2:
                neighbors[i].append(fc7_map[nearest_neighbors[j]])
            else:
                neighbors[i].append(fc7_map[randint(0, fc7_feat.shape[0]-1)])

    print 'Writing to %s...' % args.json_output
    json.dump(neighbors, open(args.json_output, 'w'))

