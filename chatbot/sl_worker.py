from __future__ import absolute_import

import os
import sys
sys.path.append('..')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'demo.settings')

import django
django.setup()

from django.conf import settings

from amt.utils import log_to_terminal

import amt.constants as constants
import PyTorch
import PyTorchHelpers
import pika
import time
import yaml
import json
import traceback
import signal
import requests
import atexit

VisDialModel = PyTorchHelpers.load_lua_class(
    constants.SL_VISDIAL_LUA_PATH, 'SLConversationModel')

VisDialATorchModel = VisDialModel(
    constants.SL_VISDIAL_CONFIG['inputJson'],
    constants.SL_VISDIAL_CONFIG['qBotpath'],
    constants.SL_VISDIAL_CONFIG['aBotpath'],
    constants.SL_VISDIAL_CONFIG['gpuid'],
    constants.SL_VISDIAL_CONFIG['backend'],
    constants.SL_VISDIAL_CONFIG['imfeatpath'],
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='sl_chatbot_queue', durable=True)


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        body['history'] = body['history'].split("||||")

        # get the imageid here so that use the extracted features in lua script
        image_id = body['image_path'].split("/")[-1].replace(".jpg", "")

        print image_id
        print type(image_id)

        result = VisDialATorchModel.abot(
            image_id, body['history'], body['input_question'])
        result['input_image'] = body['image_path']
        result['question'] = str(result['question'])
        result['answer'] = str(result['answer'])
        result['history'] = result['history'].replace("<START>", "")
        result['history'] = result['history'].replace("<END>", "")
        # Store the result['predicted_fc7'] in the database after each round
        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception, err:
        print str(traceback.print_exc())

channel.basic_consume(callback,
                      queue='sl_chatbot_queue')

channel.start_consuming()
