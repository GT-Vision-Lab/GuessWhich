from django.conf import settings
from .utils import log_to_terminal

import os
import pika
import sys
import json


def chatbot(input_question, history, image_path, socketid, bot):
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost'))
    channel = connection.channel()

    queue_name = 'sl_chatbot_queue'
    if bot == "sl":
        queue_name = "sl_chatbot_queue"
    elif bot == "rl":
        queue_name = "rl_chatbot_queue"

    channel.queue_declare(queue=queue_name, durable=True)
    message = {
        'image_path': image_path,
        'input_question': input_question,
        'history': history,
        'socketid': socketid,
        'bot': bot,
    }

    log_to_terminal(
        socketid, {"terminal": "Publishing job to %s" % (queue_name.upper())})
    channel.basic_publish(exchange='',
                          routing_key=queue_name,
                          body=json.dumps(message),
                          properties=pika.BasicProperties(
                              delivery_mode=2,  # make message persistent
                          ))

    print(" [x] Sent %r" % message)
    log_to_terminal(socketid, {"terminal": "Job published successfully"})
    connection.close()
