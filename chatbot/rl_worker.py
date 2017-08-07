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

RLVisDialModel = PyTorchHelpers.load_lua_class(
    constants.RL_VISDIAL_LUA_PATH, 'RLConversationModel')

RLVisDialATorchModel = RLVisDialModel(
    constants.RL_VISDIAL_CONFIG['inputJson'],
    constants.RL_VISDIAL_CONFIG['qBotpath'],
    constants.RL_VISDIAL_CONFIG['aBotpath'],
    constants.RL_VISDIAL_CONFIG['gpuid'],
    constants.RL_VISDIAL_CONFIG['backend'],
    constants.RL_VISDIAL_CONFIG['imfeatpath'],
)

connection = pika.BlockingConnection(pika.ConnectionParameters(
    host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='rl_chatbot_queue', durable=True)


def callback(ch, method, properties, body):
    try:
        body = yaml.safe_load(body)
        body['history'] = body['history'].split("||||")

        # Get the imageid here so that use the extracted features in lua script
        image_id = body['image_path'].split("/")[-1].replace(".jpg", "")

        result = RLVisDialATorchModel.abot(
            image_id,
            body['history'],
            body['input_question'])

        result['question'] = str(result['question'])
        result['answer'] = str(result['answer'])
        result['history'] = result['history']
        result['history'] = result['history'].replace("<START>", "")
        result['history'] = result['history'].replace("<END>", "")

        log_to_terminal(body['socketid'], {"result": json.dumps(result)})
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception, err:
        print str(traceback.print_exc())

channel.basic_consume(callback,
                      queue='rl_chatbot_queue')

channel.start_consuming()
