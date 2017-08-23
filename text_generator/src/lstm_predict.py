# coding: utf-8
import numpy as np
import random
from sys import stdout

np.random.seed(59)

from keras.models import load_model
import utils
# from utils import generator, next_token, next_token_generator

voc_file = '../data/horoscopo_5000_0300_voc.txt'
voc = open(voc_file).read().split()
voc_ind = dict((s,i) for i,s in enumerate(voc))

model_file = '../models/lstm_model_170819.0143.h5'
model = load_model(model_file)
predictor = utils.PredictorParByParReal(model,voc,voc_ind,voc_ind['<nl>'])
s = predictor.generate_text(mode='interactive')