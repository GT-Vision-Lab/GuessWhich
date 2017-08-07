import os
import sys
import json
import random
import datetime
import argparse
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input', type=str)
    parser.add_argument('-split', type=str, default='test')
    args = parser.parse_args()

    print 'Loading data...'
    hs = json.load(open(args.json_input, 'r'))
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]


    print 'Writing html...'
    html = '<!DOCTYPE html><html><head><title>VisDial RL Human Study</title><link rel="stylesheet" type="text/css" href="/~abhshkdz/static/css/bootstrap.min.css"></head><body><div class="container">'
    html += '<style>.container {width: 100%;}</style>'
    #  html += '<style>.gt_dialog {background-color: #3498db; color: white}</style>'
    #  html += '<div class="row"><div class="col-lg-12"><h1 style="font-size:2.5em;">Human study - GT vs. RL</h1></div></div>'
    html += '<div class="row"><div class="col-lg-12"><h1 style="font-size:2.5em;">Human study - SL vs. SL+RL</h1></div></div>'
    html += '<hr>'
    html += '<div class="row"><div class="col-lg-12"><table class="table table">'
    html += '<thead><tr><th>Id</th><th>Dialog 1</th><th>Dialog 2</th><th>Images</th></tr></thead>'
    html += '<tbody>'
    max_convs = 100
    #  for m in range(max_convs):
    for m in xrange(100, 100 + max_convs):
        #  qs = [i['question'] for i in hs[m]['predicted_dialog']]
        #  if len(list(set(qs))) != 10:
            #  continue
        html += "<tr>"
        dialog_order = ['predicted_dialog', 'gt_dialog']
        random.shuffle(dialog_order)
        html += "<td>" + str(m+1) + "</td>"
        html += "<td style='width:20%;' class='" + dialog_order[0] + "'><ol>"
        for j in range(10):
            html += '<li>%s %s</li>' % (hs[m][dialog_order[0]][j]['question'], hs[m][dialog_order[0]][j]['answer'])
        html += "</ol></td>"
        html += "<td style='width:20%;' class='" + dialog_order[1] + "'><ol>"
        for j in range(10):
            html += '<li>%s %s</li>' % (hs[m][dialog_order[1]][j]['question'], hs[m][dialog_order[1]][j]['answer'])
        html += "</ol></td>"
        html += "<td style='width:50%;'>"
        # creating image set
        img_indices = np.random.choice(9628, 10, replace=False)
        img_nn_indices = np.random.choice(99, 10, replace=False)
        img_ids = [int(hs[m]['nn'][i]) for i in img_nn_indices] + [int(fc7_map[i]) for i in img_indices]
        img_ids = list(set(img_ids))
        random.shuffle(img_ids)
        img_ids = [int(hs[m]['image_id'])] + img_ids[:15]
        random.shuffle(img_ids)
        html += '<div class="row">'
        for i in range(len(img_ids)):
            #  if i % 3 == 0:
                #  html += '</div><div class="row">'
            if img_ids[i] == int(hs[m]['image_id']):
                html += "<div class='gt_dialog col-lg-3'>"
            else:
                html += "<div class='col-lg-4'>"
            html += "<span>" + str(i+1) + "</span><img class='img-responsive' style='height:300px; width:300px;' src='https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg'></div>" % int(img_ids[i])
        html += "</td>"
        html += "</tr>"

    html += "</tbody></table></div></div></div></body></html>"

    with open('misc_output/human_study_sl_rl.html', 'w') as tf:
        tf.write(html.encode('utf-8'))
