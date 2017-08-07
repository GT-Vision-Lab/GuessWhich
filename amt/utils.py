from django.conf import settings
from channels import Group

import h5py
import time
import cPickle
import pdb

from .constants import (
    POOL_IMAGES_URL,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    QUALIFICATION_TYPE_ID,
    AMT_HOSTNAME,
)

import json
import os
import random
import traceback
import numpy as np
import boto.mturk.connection


def log_to_terminal(socketid, message):
    Group(socketid).send({"text": json.dumps(message)})


def get_pool_images(pool_id=1):
    with open('data/pools.json', 'r') as f:
        pool_data = json.load(f)
    return pool_data[pool_id]


def get_url_of_image(image_id):
    return POOL_IMAGES_URL + str(image_id) + ".jpg"


def fc7_sort(imfeats, prev_sort, chosen_imID):
    target_f = imfeats[chosen_imID]
    dist_vec = np.zeros(len(prev_sort), dtype='float32')

    for i in range(len(prev_sort)):
        dist_vec[i] = np.linalg.norm(imfeats[prev_sort[i]] - target_f)

    sort_ind = np.argsort(dist_vec).tolist()
    new_sort = []
    for i in range(len(sort_ind)):
        new_sort.append(prev_sort[sort_ind[i]])
    return new_sort


def create_qualifications():
    mtc = boto.mturk.connection.MTurkConnection(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        host=AMT_HOSTNAME,
        debug=2
    )

    qualification = mtc.create_qualification_type(
        name='Some Qualification Name',
        description='Qualification to avoid bias in responses by preventing workers who have already completed a HIT from doing subsequent HITs.',
        status='Active',
        auto_granted=True,
        auto_granted_value=0
    )


def set_qualification_to_worker(worker_id=None, qualification_value=0):
    mtc = boto.mturk.connection.MTurkConnection(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        host=AMT_HOSTNAME,
        debug=2
    )

    mtc.assign_qualification(QUALIFICATION_TYPE_ID, worker_id,
                             value=qualification_value,
                             send_notification=False)


def updated_qualification_to_worker(worker_id=None, qualification_value=1):
    mtc = boto.mturk.connection.MTurkConnection(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        host=AMT_HOSTNAME,
        debug=2
    )

    mtc.update_qualification_score(
        QUALIFICATION_TYPE_ID, worker_id, qualification_value)
