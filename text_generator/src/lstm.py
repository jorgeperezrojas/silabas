# coding: utf-8
import numpy as np
import random
np.random.seed(59)

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM, Dropout, GRU, Masking
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l1, l2
from keras.models import load_model
from utils import ParByParGenerator, ContinuousGenerator, next_token, next_token_generator
import time

#################
# change this if retraining
#################

retraining = False
previous_model_file = '../models/final_lstm_model_170818.2150_.h5'

#################
## output parameters

out_directory_model = '../models/'
out_model_pref = 'lstm_model_'
out_directory_train_history = '../train_history/'
time_pref = time.strftime('%y%m%d.%H%M')


#################
## input files
train_text_file = '../data/horoscopo_hoy_8000_2000_train.txt' #'../data/horoscopo_0600_0300_train.txt'
val_text_file = '../data/horoscopo_hoy_8000_2000_val.txt' #'../data/horoscopo_0600_0300_val.txt'
voc_file = '../data/horoscopo_hoy_8000_2000_voc.txt' #'../data/horoscopo_0600_0300_voc.txt'

##### set parameters of the training process
batch_size = 128
epochs = 10
iterations = 4
ptj = 2
patience = 100
#####

##### MASK VALUE EQUALS 0
##### CONSIDER HAVING THE EMPTY STRING AS THE FIRST SYMBOL IN YOUR VOCABULARY TO MAKE THIS WORK, 
##### THE SYMBOL AT INDEX 0 WILL ALWAYS BE SKIPPED!
mask_value = 0
nl_symbol = '<nl>'

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
## all these are stored in the model file .yaml
max_len = 80
lstm_units = 512
dropout = 0.3
rec_dropout = 0.3
optimizer = 'adam'
impl = 2 # BE CAREFULL!!! must be 2 for GPU
#####

if retraining == True:
    print('retraining model, loading from ' + previous_model_file)
    lstm_model = load_model(previous_model_file)

else:
    lstm_model = Sequential()
    lstm_model.add(Masking(mask_value=mask_value, input_shape=(max_len, len(voc))))
    lstm_model.add(LSTM(lstm_units, recurrent_dropout=rec_dropout, 
        implementation=impl, return_sequences=True))
    lstm_model.add(Dropout(dropout))
    lstm_model.add(LSTM(lstm_units, recurrent_dropout=rec_dropout, 
        implementation=impl)) # , return_sequences=True))
    # lstm_model.add(Dropout(dropout/4))
    # lstm_model.add(LSTM(lstm_units, recurrent_dropout=rec_dropout/4, implementation=impl))
    lstm_model.add(Dense(len(voc), activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
        metrics=['top_k_categorical_accuracy'])
    print('compiling...')

lstm_model.summary()

### create the generator objects
train_gen = ParByParGenerator(batch_size, ind_train_tokens, voc, max_len, voc_ind[nl_symbol], paragraphs_to_join = ptj, mask_value = mask_value)
val_gen = ParByParGenerator(batch_size, ind_val_tokens, voc, max_len, voc_ind[nl_symbol], paragraphs_to_join = ptj, mask_value = mask_value)
#train_gen = ContinuousGenerator(batch_size, ind_train_tokens, voc, max_len)
#val_gen = ContinuousGenerator(batch_size, ind_val_tokens, voc, max_len)

print('steps per epoch training:', train_gen.steps_per_epoch)
print('steps per epoch validation:', val_gen.steps_per_epoch)


for i in range(0,iterations):
    print('beginning iteration ' + str(i))

    outfile = out_model_pref + time_pref + '.h5'

    checkpoint = ModelCheckpoint(
        out_directory_model + outfile, 
        # monitor='val_loss', 
        monitor='val_top_k_categorical_accuracy',
        verbose=1, 
        save_best_only=True ## save best
    )

    early_stopping = EarlyStopping(
        monitor='val_top_k_categorical_accuracy', min_delta=0, patience=patience, verbose=0, mode='auto')


    model_output = lstm_model.fit_generator(
        train_gen.generator(), 
        train_gen.steps_per_epoch,
        validation_data=val_gen.generator(),
        validation_steps=val_gen.steps_per_epoch,
        epochs=epochs, 
        callbacks=[checkpoint, early_stopping]
    )

    time_pref = time_pref[:-1] + str(i)

    # save also the last state (to continue training if needed)
    final_model_file = out_directory_model + 'final_' + out_model_pref + time_pref + '.h5'
    print('saving last model:', final_model_file)
    lstm_model.save(final_model_file)

    # save history
    outfile_history = out_directory_train_history + out_model_pref + time_pref + '_hist.txt'
    outfile_parameters = out_directory_train_history + out_model_pref + time_pref + '_pars.txt'

    print('saving history:', outfile_history)
    print('saving parameters:', outfile_parameters)
    with open(outfile_history,'w') as out: 
        out.write(str(model_output.history))
        out.write('\n')
    

