import json
import h5py
import argparse
from tqdm import tqdm
from sklearn.preprocessing import normalize

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-hdf5_input', type=str)
    parser.add_argument('-json_output', type=str)
    parser.add_argument('-num_nearest', type=int, default=10)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading ground truth hdf5...'
    hf = h5py.File('../data/visdial_0.5/data_img.h5')
    fc7_feat = hf.get('images_%s' % args.split)
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]

    print 'Loading predictions hdf5...'
    hf = h5py.File(args.hdf5_input)
    data = hf.get('fc7_%s' % args.split)

    print 'Normalizing gt fc7...'
    fc7_feat_norm = normalize(fc7_feat, axis=1, norm='l2')
    #  fc7_feat_norm = fc7_feat

    print 'Computing neighbors...'
    neighbors = [[] for _ in range(data.shape[0])]
    gt_ranks, dists = [], []
    #  for i in tqdm(range(data.shape[0])):
    for i in tqdm(range(200)):
    #  for i in tqdm(range(1000)):
        #  (nearest_neighbors, rank, dist) = get_nearest_neighbors_query_with_rank(fc7_feat_norm, data[i], args.num_nearest, i // 10)
        (nearest_neighbors, rank, dist) = get_nearest_neighbors_query_with_rank(fc7_feat_norm, data[i], args.num_nearest, i)
        for j in range(args.num_nearest):
            neighbors[i].append(fc7_map[nearest_neighbors[j]])
        gt_ranks.append(rank)
        dists.append(float(dist))
        #  if i % 5000 == 0:
            #  print 'Compiling results...'
            #  res = {'neighbors': neighbors, 'gt_ranks': gt_ranks, 'dists': dists, 'split': args.split}
            #  print 'Writing to %s...' % args.json_output
            #  json.dump(res, open(args.json_output, 'w'))

    print 'Compiling results...'
    res = {'neighbors': neighbors, 'gt_ranks': gt_ranks, 'dists': dists, 'split': args.split}

    #  print 'Writing to %s...' % args.json_output
    #  json.dump(res, open(args.json_output, 'w'))

    print 'Evaluating...'
    (R1, R5, R10, ave, mrr) = printEval(np.array(gt_ranks)+1)
    print ('len: %d mrr: %f R1: %f R5 %f R10 %f Mean %f' %(len(gt_ranks), mrr, R1, R5, R10, ave))

    #  print 'Per round...'
    #  for i in range(10):
        #  idx = [_ for _ in range(i, len(gt_ranks), 10)]
        #  rs = np.array(gt_ranks)[idx]
        #  (R1, R5, R10, ave, mrr) = printEval(rs+1)
        #  print ('round: %d len: %d mrr: %f R1: %f R5 %f R10 %f Mean %f' % ((i+1), len (rs), mrr, R1, R5, R10, ave))
