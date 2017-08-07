import json
import h5py
import random
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-h5_output', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading json...'
    data = json.load(open(args.json_input, 'r'))
    vocab = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))

    print 'Building question dictionary...'
    ques_map = {}
    ques = []
    opt = np.zeros((len(data.keys()), 10, 100), dtype=np.uint32)
    gt = np.zeros((len(data.keys()), 10), dtype=np.uint32)
    for i in tqdm(range(len(vocab['unique_img_%s' % args.split]))):
        img_id = vocab['unique_img_%s' % args.split][i]
        for j in range(10):
            for k in range(100):
                if data[img_id][j]['q_options'][k] not in ques_map:
                    ques.append(data[img_id][j]['q_options'][k])
                    ques_map[data[img_id][j]['q_options'][k]] = len(ques) # adding 1 because Lua
                opt[i][j][k] = ques_map[data[img_id][j]['q_options'][k]]
            gt[i][j] = data[img_id][j]['gt_rank']+1

    print 'Tokenizing...'
    opt_list = np.zeros((len(ques), 25), dtype=np.uint32) # max 25 q length
    opt_length = np.zeros((len(ques)), dtype=np.uint32) 
    for i in tqdm(range(len(ques))):
        tokenized = word_tokenize(ques[i]) + ['?']
        tokenized = tokenized[:25]
        opt_length[i] = len(tokenized)
        for idx in range(min(len(tokenized), 25)):
            if tokenized[idx] in vocab['word2ind']:
                opt_list[i][idx] = vocab['word2ind'][tokenized[idx]]
            else:
                opt_list[i][idx] = vocab['word2ind']['UNK']

    print 'Writing to hdf5...'
    f = h5py.File(args.h5_output)
    f.create_dataset('ques_index_%s' % args.split, dtype='uint32', data=gt)
    f.create_dataset('ques_opt_%s' % args.split, dtype='uint32', data=opt)
    f.create_dataset('ques_opt_list_%s' % args.split, dtype='uint32', data=opt_list)
    f.create_dataset('ques_opt_length_%s' % args.split, dtype='uint32', data=opt_length)
    f.close()


