from django.utils import timezone
from django.conf import settings

from .utils import log_to_terminal, fc7_sort
from .sender import chatbot
import constants as constants
from .models import GameRound, ImageRanking

from channels import Group

import json
import redis
import datetime
import os
import shutil
import pdb


r = redis.StrictRedis(host='localhost', port=6379, db=0)


def ws_connect(message):
    "Method called when a user is connected through SocketIO"
    pass


def ws_message(message):
    "Method called when there is message from the SocketIO client"

    body = json.loads(message.content['text'])

    if body["event"] == "ConnectionEstablished":
        # Event when the user is connected to the socketio client
        Group(body["socketid"]).add(message.reply_channel)
        log_to_terminal(body["socketid"], {
                        "info": "User added to the Channel Group"})

    elif body["event"] == "start":
        # Event when the user starts to play the game
        current_datetime = timezone.now()
        r.set("start_time_{}".format(
            body["socketid"]),
            current_datetime.strftime("%I:%M%p on %B %d, %Y"))

    elif body["event"] == "questionSubmitted":
        # Event when the user submits a question to the backend
        body['question'] = body['question'].lower()
        bot = body['bot']
        chatbot(body['question'],
            body['prev_history'],
            os.path.join(settings.BASE_DIR, body['target_image'][1:]),
            body["socketid"],
            bot)

    elif body['event'] == "imageSubmitted":
        # Event when the user selects an image after each round of a game
        GameRound.objects.create(
            socket_id=body['socketid'],
            user_picked_image=body['user_picked_image'],
            worker_id=body['worker_id'],
            assignment_id=body['assignment_id'],
            level=body['level'],
            hit_id=body['hit_id'],
            game_id=body['game_id'],
            round_id=body['round_id'],
            question=body['question'],
            answer=body['answer'].replace("<START>", "").replace("<END>", ""),
            history=body['history'],
            target_image=body['target_image'],
            bot=body['bot'],
            task=body['task'],
        )
        log_to_terminal(body["socketid"], {"image_selection_result": True})

    elif body['event'] == 'finalImagesSelected':
        # Event when the user submit the ranking of after completing all rounds
        ImageRanking.objects.create(
            socket_id=body['socketid'],
            final_image_list=body['final_image_list'],
            worker_id=body['worker_id'],
            assignment_id=body['assignment_id'],
            level=body['level'],
            hit_id=body['hit_id'],
            game_id=body['game_id'],
            bot=body['bot'],
            target_image=body['target_image'],
            score=body['bonus'],
            task=body['task'],
        )


def ws_disconnect(message):
    "Method invoked when the client disconnects the socket connection"
    pass
