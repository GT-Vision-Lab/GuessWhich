from django.conf import settings

import os
import sys
sys.path.append(os.path.join(settings.BASE_DIR, 'chatbot'))

POOL_IMAGES_URL = os.path.join(settings.MEDIA_URL, 'val2014/')

BOT_INTORDUCTION_MESSAGE = [
    "Hi, my name is Abot. I am an Artificial Intelligence." \
    "I have been assigned one of these images as the target image." \
    "I am not allowed to show you the image, but as a start," \
    "I will describe the image to you in a sentence." \
    "You can then ask me follow up questions about it. " \
    "When ready, submit one of the images on the left as your best guess. " \
    "I will try to describe the image and answer your questions, but I am not perfect." \
    "I make quite a few mistakes. I hope we can work together to find the image! " \
    "Let's do this! Note: My knowledge of English is limited." \
    "Sometimes if I don't know the right word, I say UNK. " \
    "You will win points based on how accurately you are able to guess.",
]

SL_VISDIAL_CONFIG = {
    'inputJson': os.path.join(settings.BASE_DIR, 'chatbot/data/chat_processed_params.json'),
    'qBotpath': os.path.join(settings.BASE_DIR, 'chatbot/data/qbot_hre_qih_sl.t7'),
    'aBotpath': os.path.join(settings.BASE_DIR, 'chatbot/data/abot_hre_qih_sl.t7'),
    'gpuid': 0,
    'backend': 'cudnn',
    'imfeatpath': os.path.join(settings.BASE_DIR, 'chatbot/data/all_pools_vgg16_features.t7'),
}

SL_VISDIAL_LUA_PATH = "sl_evaluate.lua"


RL_VISDIAL_CONFIG = {
    'inputJson': os.path.join(settings.BASE_DIR, 'chatbot/data/chat_processed_params.json'),
    'qBotpath': os.path.join(settings.BASE_DIR, 'chatbot/data/qbot_rl.t7'),
    'aBotpath': os.path.join(settings.BASE_DIR, 'chatbot/data/abot_rl.t7'),
    'gpuid': 0,
    'backend': 'cudnn',
    'imfeatpath': os.path.join(settings.BASE_DIR, 'chatbot/data/all_pools_vgg16_features.t7'),
}

RL_VISDIAL_LUA_PATH = "rl_evaluate.lua"

NUMBER_OF_ROUNDS_IN_A_GAME = 9

NUMBER_OF_GAMES_IN_A_HIT = 1

AWS_ACCESS_KEY_ID = "<Access Key ID>"

AWS_SECRET_ACCESS_KEY = "<Secret Access Key>"

QUALIFICATION_TYPE_ID = "<Qualification ID>"

AMT_HOSTNAME = 'mechanicalturk.amazonaws.com'

MAX_BONUS_IN_A_GAME = 200

BONUS_DEDUCTION_FOR_EACH_CLICK = 10

BONUS_FOR_CORRECT_IMAGE_AFTER_EACH_ROUND = 10
