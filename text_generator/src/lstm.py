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
import time

#################
# change this if retraining
#################

retraining = True
previous_model_file = '../models/lstm_model_170814.0259_100_512_0.50_adam_034_3.26_2.92_0.62.h5'

#################
## output parameters

out_directory_model = '../models/'
out_model_pref = 'lstm_model_'
out_directory_train_history = '../train_history/'
time_pref = time.strftime('%y%m%d.%H%M') + '_'

#################
## input files

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
max_len = 100
lstm_units = 512
dropout = 0.5
optimizer = 'adam'
impl = 2
#####

##### set parameters of the training process
batch_size = 64
epochs = 15
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


outfile = out_directory_model + out_model_pref + time_pref + \
    '{0:03d}_{1:03d}_{2:.2f}_{3}_'.format(max_len,lstm_units,dropout,optimizer) + \
    '{epoch:03d}_{loss:.2f}_{val_loss:.2f}_{val_top_k_categorical_accuracy:.2f}.h5'

checkpoint = ModelCheckpoint(
    outfile, 
    # monitor='val_loss', 
    monitor='val_top_k_categorical_accuracy',
    verbose=1, 
    save_best_only=True ## save best
)

model_output = lstm_model.fit_generator(
    generator(batch_size, ind_train_tokens, voc, max_len), 
    (len(ind_train_tokens) - max_len)/batch_size + 1,
    validation_data=generator(batch_size, ind_val_tokens, voc, max_len),
    validation_steps=(len(ind_val_tokens) - max_len)/batch_size + 1,
    epochs=epochs, 
    callbacks=[checkpoint]
)

# save also the last state (to continue training if needed)
lstm_modle.save('final_' + outfile)

# save history
outfile_history = out_directory_train_history + out_model_pref + time_pref + \
    '{0:03d}_{1:03d}_{2:.2f}_{3:03d}_{4}_'.format(max_len,lstm_units,dropout,epochs,optimizer) + \
    '.txt'

with open(outfile_history,'w') as out: 
    out.write(str(model_output.history))
    out.write('\n')

