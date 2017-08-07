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
    parser.add_argument('-num_nearest', type=int, default=100)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading ground truth hdf5...'
    hf = h5py.File('../data/visdial_0.5/data_img.h5')
    fc7_feat = hf.get('images_%s' % args.split)
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]
    inv_fc7_map = {i: fc7_map.index(i) for i in fc7_map}

    print 'Loading im options...'
    fc7_ops = json.load(open('working/imoptions_%s_wrand.json' % args.split, 'r'))
    fc7_ops_ind = np.zeros((len(fc7_ops), 100), dtype=np.uint32)
    for i in tqdm(range(len(fc7_ops))):
        assert fc7_ops[i][0] == fc7_map[i]
        for j in range(100):
            fc7_ops_ind[i][j] = inv_fc7_map[fc7_ops[i][j]]

    print 'Loading predictions hdf5...'
    hf = h5py.File(args.hdf5_input)
    data = hf.get('fc7_%s' % args.split)

    print 'Normalizing gt fc7...'
    fc7_feat_norm = normalize(fc7_feat, axis=1, norm='l2')

    print 'Computing neighbors...'
    neighbors = [[] for _ in range(data.shape[0])]
    gt_ranks, dists = [], []
    for i in tqdm(range(data.shape[0])):
    #  for i in tqdm(range(100)):
        print fc7_ops_ind[i//10]
        feat_sub = fc7_feat_norm[fc7_ops_ind[i//10]]
        (nearest_neighbors, rank, dist) = get_nearest_neighbors_query_with_rank(feat_sub, data[i], args.num_nearest, 0)
        for j in range(args.num_nearest):
            neighbors[i].append(fc7_map[nearest_neighbors[j]])
        gt_ranks.append(rank)
        dists.append(float(dist))
        if i == 9:
            break
        #  if i % 20000 == 0:
            #  print 'Compiling results...'
            #  res = {'neighbors': neighbors, 'gt_ranks': gt_ranks, 'dists': dists, 'split': args.split}
            #  print 'Writing to %s...' % args.json_output
            #  json.dump(res, open(args.json_output, 'w'))

    print 'Compiling results...'
    res = {'neighbors': neighbors, 'gt_ranks': gt_ranks, 'dists': dists, 'split': args.split}

    #  print 'Writing to %s...' % args.json_output
    #  json.dump(res, open(args.json_output, 'w'))

    print 'Evaluating...'
    #  R1 = np.sum(np.array(gt_ranks)+1==1) / float(len(gt_ranks))
    #  R5 =  np.sum(np.array(gt_ranks)+1<=5) / float(len(gt_ranks))
    #  R10 = np.sum(np.array(gt_ranks)+1<=10) / float(len(gt_ranks))
    #  ave = np.sum(np.array(gt_ranks)+1) / float(len(gt_ranks))
    #  mrr = np.sum(1/(np.array(gt_ranks, dtype='float')+1)) / float(len(gt_ranks))
    (R1, R5, R10, ave, mrr) = printEval(np.array(gt_ranks)+1)
    print ('len: %d mrr: %f R1: %f R5 %f R10 %f Mean %f' %(len(gt_ranks), mrr, R1, R5, R10, ave))

    print 'Per round...'
    for i in range(10):
        idx = [_ for _ in range(i, len(gt_ranks), 10)]
        rs = np.array(gt_ranks)[idx]
        (R1, R5, R10, ave, mrr) = printEval(rs+1)
        print ('len: %d mrr: %f R1: %f R5 %f R10 %f Mean %f' %(len (rs), mrr, R1, R5, R10, ave))
