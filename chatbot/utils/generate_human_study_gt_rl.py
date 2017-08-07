import json
import h5py
import random
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize

random.seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading ground truth...'
    gt = json.load(open('../data/visdial_0.5/visdial_0.5_%s.json' % args.split, 'r'))
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]

    print 'Parsing ground truth...' 
    im2dialog, im2caption = {}, {}
    for i in range(len(gt)):
        dialog = []
        for j in range(10):
            question = gt[i]['dialog'][j]['question'].strip().lower() + '?'
            question = ' '.join(word_tokenize(question))
            answer = gt[i]['dialog'][j]['answer'].strip().lower()
            answer = ' '.join(word_tokenize(answer))
            dialog.append({'question': question, 'answer': answer})
        im2dialog[str(gt[i]['image_id'])] = dialog
        im2caption[str(gt[i]['image_id'])] = ' '.join(word_tokenize(gt[i]['caption'].strip().lower()))

    print 'Loading predictions...'
    preds = json.load(open(args.json_input, 'r'))

    print 'Loading fc7 neighbors...'
    neighbors = json.load(open('working/imoptions_%s.json' % args.split, 'r'))

    print 'Constructing human study set...'
    human_study = []
    for i in range(len(preds)):
        assert int(fc7_map[i]) == int(neighbors[i][0])
        el = {'image_id': int(fc7_map[i]),
              'predicted_dialog': preds[i],
              'gt_dialog': im2dialog[str(fc7_map[i])],
              'nn': neighbors[i][1:],
              'caption': im2caption[str(fc7_map[i])]}
        human_study.append(el)

    print 'Saving json...'
    json.dump(human_study, open('working/human_study_gt_rl.json', 'w'))
