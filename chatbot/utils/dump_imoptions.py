import json
import random
import argparse
from tqdm import tqdm

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading json %s...' % args.json_input
    data = json.load(open(args.json_input, 'r'))
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]

    #  for i in range(len(fc7_map)):
    for i in range(100):
        img_id = fc7_map[i]
        print '<br><img style="max-width:300px" src="https://vqa_mscoco_images.s3.amazonaws.com/val2014/COCO_val2014_%012d.jpg">' % int(img_id)
        assert img_id == data[i][0]
        for j in xrange(1, 11):
            print '<img style="max-width:100px" src="https://vqa_mscoco_images.s3.amazonaws.com/val2014/COCO_val2014_%012d.jpg">' % int(data[i][j])
