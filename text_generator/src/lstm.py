# coding: utf-8
import numpy as np
import random
np.random.seed(59)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, GRU
from keras import metrics
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from utils import generator, next_token, next_token_generator

#################
# change this if retraining
#################

retraining = False
previous_model_file = ''

#################

train_text_file = '../data/biblia_ntv_train.txt'
val_text_file = '../data/biblia_ntv_val.txt'
voc_file = '../data/biblia_ntv_voc.txt'

train_tokens = open(train_text_file).read().split()
val_tokens = open(val_text_file).read().split()
voc = open(voc_file).read().split()
voc_ind = dict((s,i) for i,s in enumerate(voc))

# generate a list of indices from corpus considering only tokens in the vocabulary
ind_train_tokens = [voc_ind[token] for token in train_tokens if token in voc]
ind_val_tokens = [voc_ind[token] for token in val_tokens if token in voc]

### for testing pourposes you can use a small subset of the train or val data
max_train_data = len(ind_train_tokens)
max_val_data = len(ind_val_tokens)
ind_train_tokens = ind_train_tokens[:max_train_data]
ind_val_tokens = ind_val_tokens[:max_val_data]

print('vocabulary size:',len(voc))
print('train data size:',len(ind_train_tokens))
print('validation data size:',len(ind_val_tokens))

##### set parameters of the model
## WATCHOUT: cannot be changed when retraining
max_len = 80
lstm_units = 256
dropout = 0.3
optimizer = 'adam'
impl = 2
#####

##### set parameters of the training process
batch_size = 64
epochs = 5
#####

if retraining == True:
    print('retraining model, loading from ' + previous_model_file)
    lstm_model = load_model(previous_model_file)

else:
    lstm_model = Sequential()
    lstm_model.add(LSTM(lstm_units, input_shape=(max_len, len(voc)), implementation=impl, return_sequences=True))
    lstm_model.add(Dropout(dropout))
    lstm_model.add(LSTM(lstm_units, implementation=impl))
    lstm_model.add(Dropout(dropout))
    lstm_model.add(Dense(len(voc), activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
        metrics=['top_k_categorical_accuracy'])

lstm_model.summary()

outfile = '../models/lstm_model_' + str(max_len) + '_' + str(lstm_units) + '_' + str(dropout) + '_{val_loss:.4f}_' + optimizer + '.h5'

checkpoint = ModelCheckpoint(
    outfile, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True ## save best
)

lstm_model.fit_generator(
    generator(batch_size, ind_train_tokens, voc, max_len), 
    (len(ind_train_tokens) - max_len)/batch_size + 1,
    validation_data=generator(batch_size, ind_val_tokens, voc, max_len),
    validation_steps=(len(ind_val_tokens) - max_len)/batch_size + 1,
    epochs=epochs, 
    callbacks=[checkpoint]
)









