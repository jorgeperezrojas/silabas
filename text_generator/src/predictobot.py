import time
import json 
import requests
import urllib
import numpy as np
import random
from sys import stdout
from keras.models import load_model
import utils


TOKEN = "391514614:AAHiDm2z5kY-7U4PTxz5wn4R6vE9KJGI2yY"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)


def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def get_json_from_url(url):
    content = get_url(url)
    js = json.loads(content)
    return js


def get_updates(offset=None):
    url = URL + "getUpdates?timeout=100"
    if offset:
        url += "&offset={}".format(offset)
    js = get_json_from_url(url)
    return js

def get_last_update_id(updates):
    update_ids = []
    for update in updates["result"]:
        update_ids.append(int(update["update_id"]))
    return max(update_ids)


def get_last_chat_id_and_text(updates):
    num_updates = len(updates["result"])
    last_update = num_updates - 1
    text = updates["result"][last_update]["message"]["text"]
    chat_id = updates["result"][last_update]["message"]["chat"]["id"]
    return (text, chat_id)


def send_message(text, chat_id):
    text = urllib.parse.quote_plus(text)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    get_url(url)

def handle_updates(updates,predictor,voc):
    first = True
    for update in updates["result"]:
        try:
            text = update["message"]["text"]
            chat = update["message"]["chat"]["id"]
            s = ''
            if text == "/horoscopo":
                if first:
                    first = False
                    s = predictor.generate_text()
                message = utils.token_sequence_to_text([voc[i] for i in s])
                send_message(message, chat)
        except KeyError:
            pass
    

def main():

    voc_file = '../data/horoscopo_5000_0300_voc.txt'
    voc = open(voc_file).read().split()
    voc_ind = dict((s,i) for i,s in enumerate(voc))

    model_file = '../models/lstm_model_170821.0948.h5'
    model = load_model(model_file)
    predictor = utils.PredictorParByParReal(model,voc,voc_ind,voc_ind['<nl>'])

    print("listoco!")

    last_update_id = None
    while True:
        updates = get_updates(last_update_id)
        if len(updates["result"]) > 0:
            last_update_id = get_last_update_id(updates) + 1
            handle_updates(updates, predictor, voc)
        time.sleep(0.5)


if __name__ == '__main__':
    main()