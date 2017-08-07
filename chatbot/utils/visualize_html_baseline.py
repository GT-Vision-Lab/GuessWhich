import os
import sys
import json
import datetime
import argparse
import numpy as np
from tqdm import tqdm

np.random.seed(32)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_input_img', type=str)
    #  parser.add_argument('-json_input_ques', type=str)
    parser.add_argument('-split', type=str, default='test')
    parser.add_argument('-identifier', type=str, default='eval')
    args = parser.parse_args()

    print 'Loading data...'
    res_img = json.load(open(args.json_input_img, 'r'))
    #  res_ques = json.load(open(args.json_input_ques, 'r'))
    df = json.load(open('../data/visdial_0.5/chat_processed_params.json', 'r'))
    fc7_map = df['unique_img_%s' % args.split]
    #  res_hist = json.load(open('../results/im-hist-enc-dec-questioner-only-results.json', 'r'))

    print 'Compiling results...'
    res = []
    #  rounds_gt_rank = [[] for _ in range(10)]
    #  rounds_gt_dist = [[] for _ in range(10)]
    #  for i in tqdm(range(len(res_img['gt_ranks']))):
    for i in tqdm(range(200)):
        el = {'gt_img': fc7_map[i],
              'gt_img_rank': res_img['gt_ranks'][i] + 1,
              'gt_img_dist': res_img['dists'][i],
              'pred_imgs': res_img['neighbors'][i]}
        res.append(el)


    print 'Writing html...'
    html = '<!DOCTYPE html><html><head><title>VisDial RL Eval</title><link rel="stylesheet" type="text/css" href="/~abhshkdz/static/css/bootstrap.min.css"></head><body><div class="container">'
    html += '<div class="row"><div class="col-lg-12"><h1 style="font-size:2.5em;">%s</h1></div></div>' % args.identifier
    html += '<hr>'
    #  html += 'Total dialogs: %d <br>' % (len(res) // 10)
    html += '<hr>'
    html += '<div class="row"><div class="col-lg-12"><table class="table table">'
    html += '<thead><tr><th style="max-width:200px;">Image</th><th>NN images to predicted fc7</th></tr></thead>'
    html += '<tbody>'
    max_convs = 200
    #  indices = np.random.choice(100 // 10, max_convs, replace=False)
    #  indices = np.random.choice(len(res) // 10, max_convs, replace=False)
    indices = [i for i in range(max_convs)]
    for m in range(max_convs):
        i = indices[m]
        html += "<tr>"
        html += "<td style='max-width:200px;'><img class='img-responsive' style='max-width:200px;' src='https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg'></td>" % int(res[i]['gt_img'])
        html += "<td><ol>"
        #  for j in range(10):
            #  html += '<li>%s<br>' % res[i*10+j]['history']
        html += '<div class="row">'
        for k in range(6):
            html += "<div class='col-lg-2'><img class='img-responsive' src='https://vision.ece.vt.edu/mscoco/images/val2014/COCO_val2014_%012d.jpg'></div>" % int(res[i]['pred_imgs'][k])
        html += "</div>GT Rank %d MSE %.02f" % (res[i]['gt_img_rank'], res[i]['gt_img_dist'])
        #  html += "<br>%s<br></li>" % (res[i*10+j]['pred_ques'])
        html += "</ol></td>"
        html += "</tr>"

    html += "</tbody></table></div></div></div></body></html>"

    with open('eval_output/%s.html' % args.identifier, 'w') as tf:
        tf.write(html.encode('utf-8'))
