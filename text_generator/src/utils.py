# coding: utf-8
import numpy as np

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