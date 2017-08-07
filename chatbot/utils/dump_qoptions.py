import json
import random
import argparse
from tqdm import tqdm
from unidecode import unidecode

from utils import *

def cleanNonAscii(text):
    try:
        text = text.decode('ascii');

    except:
        try: text = unicode(text, encoding = 'utf-8');
        except:
            pass
        text = unidecode(text);
    return text;

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading json %s...' % args.json_input
    data = json.load(open(args.json_input, 'r'))
    gt = json.load(open('../data/visdial_0.5/visdial_0.5_%s.json' % args.split, 'r'))

    #  for i in tqdm(range(len(gt))):
    for i in tqdm(range(100)):
        img_id = gt[i]['image_id']
        print '# IMAGE'
        print 'https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg' % img_id
        print '# CAPTION'
        print gt[i]['caption']
        for j in range(10):
            print '# ROUND %d' % (j+1)
            print '## QUESTION OPTIONS'
            for k in range(10):
                assert data[str(img_id)][j]['q_options'][data[str(img_id)][j]['gt_rank']] == gt[i]['dialog'][j]['question']
                if k == data[str(img_id)][j]['gt_rank']:
                    continue
                else:
                    print cleanNonAscii(data[str(img_id)][j]['q_options'][k])
            qa = cleanNonAscii(gt[i]['dialog'][j]['question'] + '? ' + gt[i]['dialog'][j]['answer'])
            print '## GT'
            print qa
            print ''


