# coding: utf-8
import numpy as np
np.random.seed(59)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, GRU
from keras.layers.wrappers import Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from utils import generator

#################
# change this if retraining
#################

retraining = False
previous_model_file = ''

#################

text_data_file = '../data/ReinaValera1960-sil-tags-frecuentes.txt'
voc_file = '../data/biblia_sil_voc_f100.voc'

tokens = open(text_data_file).read().split()
voc = open(voc_file).read().split()
voc_ind = dict((s,i) for i,s in enumerate(voc))

# generate a list of indices from corpus considering only tokens in the vocabulary
ind_tokens = [voc_ind[token] for token in tokens if token in voc]

### for testing pourposes you can use a small subset of the data
max_data = len(ind_tokens)
#max_data = 4000
ind_tokens = ind_tokens[:max_data]

print('vocabulary size:',len(voc))
print('data size:',len(ind_tokens))

##### set parameters of the model
## WATCHOUT: cannot be changed when retraining
max_len = 30
lstm_units = 512
optimizer = 'adam' 
#####

##### set parameters of the training process
batch_size = 128
epochs = 10
#####

if retraining == True:
    print('retraining model, loading from ' + previous_model_file)
    lstm_model = load_model(previous_model_file)

else:
    lstm_model = Sequential()
    lstm_model.add(LSTM(lstm_units, input_shape=(max_len, len(voc))))
    lstm_model.add(Dense(len(voc), activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer)


outfile = '../models/lstm_model_' + str(max_len) + '_' + str(lstm_units) + '_{loss:.4f}_' + optimizer + '.h5'

checkpoint = ModelCheckpoint(
    outfile, 
    monitor='loss', 
    verbose=1, 
    save_best_only=False ## save after every epoch!
)

lstm_model.fit_generator(generator(batch_size, ind_tokens, voc, max_len), 
    (len(ind_tokens) - max_len)/batch_size + 1, epochs=epochs, callbacks=[checkpoint])
