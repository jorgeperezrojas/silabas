# coding: utf-8
import numpy as np
import random

def generator(batch_size, ind_tokens, voc, max_len):
    n_features = len(voc)
    while 1:
        for i in range(0, len(ind_tokens) - max_len, batch_size):

            # consider the case of a possible small final batch_size
            actual_batch_size = min(batch_size, len(ind_tokens) - max_len - i)

            X_batch = np.zeros((actual_batch_size, max_len, n_features), dtype = np.bool)
            Y_batch = np.zeros((actual_batch_size, n_features), dtype = np.bool)

            for j in range(0, actual_batch_size): # iteration for every batch member
                for k, ind_token in enumerate(ind_tokens[i+j: i+j+max_len]): # iteration for every token
                    X_batch[j, k, ind_token] = 1
                Y_batch[j,ind_tokens[i+j+max_len]] = 1
            yield X_batch, Y_batch





def sample_token(pred, diversity):
    n = int(len(pred[0])*diversity) + 1
    return random.choice(np.argsort(pred[0])[-n:])

def next_token(model, ind_tokens, voc, max_len, diversity=0, pad_value=0):
    # use the last max_len tokens or pad if less
    if len(ind_tokens) >= max_len:
        input_seq = ind_tokens[-max_len:]
    else:
        input_seq = [pad_value for i in range(0,max_len-len(ind_tokens))]
        input_seq.extend(ind_tokens)

    n_features = len(voc)
    X = np.zeros((1, max_len, n_features), dtype = np.bool)

    for k, i in enumerate(input_seq):
        X[0, k, i] = 1

    pred = model.predict(X, verbose=0)
    return sample_token(pred, diversity)

def next_token_generator(model, ind_tokens, voc, max_len, min_diversity=0, max_diversity=0.05, pad_value=0, stop_tokens=[0]):
    current_seq = ind_tokens
    while 1:
        diversity = min_diversity
        if current_seq[len(current_seq)-1] in stop_tokens:
            diversity = max_diversity

        n_t = next_token(model, current_seq, voc, max_len, diversity, pad_value=pad_value)
        yield n_t

        curren_seq = current_seq[:-2].append(n_t)



# lstm_model.load_weights('../models/lstm_model_30_512_0.5631_adam.h5')

# from sys import stdout

# init = random.randint(0,len(ind_tokens)-max_len)

# print('init: ',init)

# initial_seq = ind_tokens[init:init+max_len]
# for token in initial_seq:
#     if token == 0:
#         stdout.write(' ')
#     else:
#         stdout.write(voc[token])

# stdout.write('///')

# max = 200
# for i,token in enumerate(next_token_generator(lstm_model, initial_seq, voc, max_len, min_diversity=0, max_diversity=0.05)):
#     if token == 0:
#         stdout.write(' ')
#     else:
#         stdout.write(voc[token])
#     if i == max:
#         break

# max = 200
# for i,token in enumerate(next_token_generator(lstm_model, i_s, voc, max_len, min_diversity=0, max_diversity=0.05)):
#     if token == 0:
#         stdout.write(' ')
#     else:
#         stdout.write(voc[token])
#     if i == max:
#         break

    



