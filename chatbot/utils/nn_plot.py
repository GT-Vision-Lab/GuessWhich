import json
import h5py
import random
import numpy as np
from sklearn.preprocessing import normalize

from utils import *

random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    hf = h5py.File('../data/visdial_0.5/data_img.h5')
    fc7_feat = hf.get('images_test')
    fc7_feat_norm = normalize(fc7_feat, axis=1, norm='l2')
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_test']

    #  img_id = 323399
    img_id = 165795
    #  preds_img = h5py.File('../results/im-hist-enc-dec-questioner-interact-nocur-noclamp-anneal-batched-iter35k-fc7.h5')['fc7_test']
    preds_img = json.load(open('working/im-hist-enc-dec-questioner-interact-nocur-noclamp-anneal-batched-iter35k-fc7_imresults.json', 'r'))
    preds_ques = json.load(open('../results/im-hist-enc-dec-questioner-interact-nocur-noclamp-anneal-batched-iter35k-results.json'))

    index = fc7_map.index(str(img_id))

    dialog = preds_ques[index]
    #  dists = preds_img['dists'][index*10:index*10+10]
    #  ranks = np.array(preds_img['gt_ranks'][index*10:index*10+10]) + 1
    neighbors = preds_img['neighbors'][index*10:index*10+10]

    #  indices = [i for i in range(9628) if i != index]
    size = 2000
    indices = np.random.choice(9628, size, replace=False)
    while index in indices:
        indices = np.random.choice(9628, size, replace=False)

    dist = (fc7_feat_norm[indices] - fc7_feat_norm[index]) ** 2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)

    idx = np.argsort(dist)

    before_idx = []
    before_idx_dist = []
    after_idx = []
    after_idx_dist = []
    pred_dists = []
    for r in range(10):
        cum_dist = 0
        #  for k in range(3):
        pred_index = fc7_map.index(neighbors[r][0])
        pred_dist = (fc7_feat_norm[pred_index] - fc7_feat_norm[index]) ** 2
        pred_dist = np.sum(pred_dist)
        pred_dist = np.sqrt(pred_dist)
            #  cum_dist += pred_dist
        #  cum_dist = cum_dist / 3.0
        pred_dists.append(pred_dist)
        for i in range(size-1):
            if dist[idx[i]] < pred_dist and dist[idx[i+1]] >= pred_dist:
                before_idx.append([i-2, i-1,i])
                before_idx_dist.append([dist[idx[i-2]], dist[idx[i-1]], dist[idx[i]]])
                after_idx.append([i+1,i+2,i+3])
                after_idx_dist.append([dist[idx[i+1]], dist[idx[i+2]], dist[idx[i+3]]])

    for i in range(len(before_idx)):
        print i
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[before_idx[i][0]]]]), before_idx_dist[i][0])
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[before_idx[i][1]]]]), before_idx_dist[i][1])
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[before_idx[i][2]]]]), before_idx_dist[i][2])
        print '1-NN https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(neighbors[i][0]), pred_dists[i])
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[after_idx[i][0]]]]), after_idx_dist[i][0])
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[after_idx[i][1]]]]), after_idx_dist[i][1])
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg %.05f' % (int(fc7_map[indices[idx[after_idx[i][2]]]]), after_idx_dist[i][2])
