# coding: utf-8
import numpy as np
import random
import re
from separador_silabas import silabas
import sys
import time
np.random.seed(59)
random.seed(59)

class ContinuousGenerator:
    """Class to wrap a generator to train lstms with text as a continous stream"""

    def __init__(self, batch_size, ind_tokens, voc, max_len):
        self.ind_tokens = ind_tokens
        self.voc = voc
        self.max_len = max_len
        self.batch_size = batch_size
        self.steps_per_epoch = int(len(ind_tokens) / batch_size) + 1
        
    def generator(self):
        voc = self.voc
        ind_tokens = self.ind_tokens
        batch_size = self.batch_size
        max_len = self.max_len

        n_features = len(voc)

        X_batch = np.zeros((batch_size, max_len, n_features), dtype = np.bool)
        Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)
        current_batch_index = 0

        while 1:
            i = random.randint(0, len(ind_tokens) - max_len - 1)
            for k, ind_token in enumerate(ind_tokens[i: i+max_len]): # iteration for every token
                X_batch[current_batch_index, k, ind_token] = 1
            Y_batch[current_batch_index,ind_tokens[i+max_len]] = 1

            current_batch_index += 1

            if current_batch_index == batch_size:
                yield X_batch, Y_batch
                current_batch_index = 0
                X_batch = np.zeros((batch_size, max_len, n_features), dtype = np.bool)
                Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)

class ParByParGenerator:
    """Class to wrap a generator to train lstms paragraph by paragraph"""

    def __init__(self, batch_size, ind_tokens, voc, max_len, split_symbol_index, paragraphs_to_join = 1, mask_value = 0, mode='normal'):
        self.ind_tokens = ind_tokens
        self.voc = voc
        self.max_len = max_len
        self.batch_size = batch_size
        self.mask_value = mask_value
        self.mode = mode

        # first split paragraphs (not adding the split symbol)
        self.paragraphs = []
        current_par = []
        counter = 1

        total_number_of_examples = 0

        for ind in ind_tokens:
            if ind != split_symbol_index:
                current_par.append(ind)
            elif counter < paragraphs_to_join:
                counter += 1
            else:
                current_par.append(split_symbol_index)
                self.paragraphs.append(current_par)
                total_number_of_examples += (len(current_par) - 1) if len(current_par) < max_len else (len(current_par) - max_len - 1)
                counter = 1
                current_par = []
        # add a possible final paragraph
        if current_par != []:
            self.paragraphs.append(current_par)
            total_number_of_examples += (len(current_par) - 1) if len(current_par) < max_len else (len(current_par) - max_len - 1)

        # update steps per epoch
        self.steps_per_epoch = int(total_number_of_examples / batch_size) + 1


    def generator(self):
        voc = self.voc
        ind_tokens = self.ind_tokens
        batch_size = self.batch_size
        max_len = self.max_len
        paragraphs = self.paragraphs
        mode = self.mode

        n_features = len(voc)
        
        if mode == 'normal':
            X_batch = np.zeros((batch_size, max_len, n_features), dtype = np.bool)
            Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)
            m_v = -1
        elif mode == 'sparse':
            X_batch = np.zeros((batch_size, max_len), dtype = np.int32)
            Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)
            #Y_batch = np.zeros((batch_size), dtype = np.int32)
            m_v = 0

        current_batch_index = 0

        while 1:
            par = random.choice(paragraphs) # pick a paragraph
            i = random.randint(0, len(par) - 2) # pick an index in the paragraph

            # create the next example
            right_limit = i+1
            left_limit = 0 if right_limit-max_len < 0 else right_limit-max_len
            pad_length = 0 if right_limit-max_len >= 0 else max_len-right_limit

            X_data = [m_v] * pad_length + par[left_limit:right_limit]
            Y_data = par[right_limit]

            if mode == 'normal':
                # add it to the batch
                for j,ind_token in enumerate(X_data):
                    if ind_token == m_v:
                        X_batch[current_batch_index, j, :] = self.mask_value # mask the value for no existing index
                    else:
                        X_batch[current_batch_index, j, ind_token] = 1
                Y_batch[current_batch_index, Y_data] = 1
            elif mode == 'sparse':
                # add it to the batch
                for j,ind_token in enumerate(X_data):
                    X_batch[current_batch_index, j] = ind_token
                Y_batch[current_batch_index, Y_data] = 1
                #Y_batch[current_batch_index] = Y_data

            current_batch_index += 1
            if current_batch_index == batch_size:
                current_batch_index = 0

                if mode == 'normal':
                    yield X_batch, Y_batch
                    X_batch = np.zeros((batch_size, max_len, n_features), dtype = np.bool)
                    Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)
                elif mode == 'sparse':
                    yield X_batch, Y_batch # !!!!
                    X_batch = np.zeros((batch_size, max_len), dtype = np.int32)
                    Y_batch = np.zeros((batch_size, n_features), dtype = np.bool)
                    #Y_batch = np.zeros((batch_size), dtype = np.int32)


########
class PredictorParByPar:
    def __init__(self, model,voc,voc_ind,split_symbol_index,seed_text='en el amor',temperature=0.6,prob_tresh=0.5,mask_value = 0):
        self.model = model
        self.voc = voc
        self.voc_ind = voc_ind
        self.seed_text = seed_text
        self.temperature = temperature
        self.prob_tresh = prob_tresh
        self.mask_value = mask_value
        self.split_symbol_index = split_symbol_index

    def generate_text(self,length=100, mode='batch'):

        seed_text = self.seed_text

        max_len = self.model.layers[0].input_shape[1]
        _, initial_seq = text_to_sequence_tokens(seed_text, self.voc, self.voc_ind)

        if len(initial_seq) >= max_len:
            initial_seq = initial_seq[-max_len:]
        else:
            initial_seq = [-1] * (len(initial_seq) - max_len) + initial_seq

        text_tokens = [self.voc[token] for token in initial_seq]
        input_tokens = initial_seq
        output_tokens = initial_seq

        for i in range(0,length):
            # first generate input tensor
            n_features = len(self.voc)
            X = np.zeros((1, max_len, n_features), dtype = np.bool)
            for k, j in enumerate(input_tokens):
                X[0, k, j] = 1

            # predict next token
            pred_token = sample_token(self.model.predict(X, verbose=0), self.temperature, self.prob_tresh)
            output_tokens.append(pred_token)

            if pred_token == self.split_symbol_index:
                print('fin')
                break

            if mode == 'interactive':
                sys.stdout.write(token_sequence_to_text([self.voc[pred_token]]))
                sys.stdout.flush()
                time.sleep(0.001)

            input_tokens = input_tokens[1:] + [pred_token]

        return output_tokens # , token_sequence_to_text(output_tokens)

   
        
 
def next_token(model, ind_tokens, voc, max_len, prob_tresh = 0.5, temperature=0, circular=True, pad_value=1):
    # use the last max_len tokens or pad if less
    if len(ind_tokens) >= max_len:
        input_seq = ind_tokens[-max_len:]
    elif circular == False:
        input_seq = [pad_value for i in range(0,max_len-len(ind_tokens))]
        input_seq.extend(ind_tokens)
    else:
        input_seq = ind_tokens[-(max_len % len(ind_tokens)):]
        for _ in range(0,int(max_len/len(ind_tokens))):
            input_seq.extend(ind_tokens)

    n_features = len(voc)
    X = np.zeros((1, max_len, n_features), dtype = np.bool)

    for k, i in enumerate(input_seq):
        X[0, k, i] = 1

    pred = model.predict(X, verbose=0)
    return sample_token(pred, temperature, prob_tresh=prob_tresh)




def sample_token(pred, temperature=0, prob_tresh = 0.5):
    pred = pred[0]

    if random.random() > prob_tresh:
        pred = np.asarray(pred).astype('float64')
        pred = np.log(pred) / temperature
        exp_pred = np.exp(pred)
        pred = exp_pred / np.sum(exp_pred)
        #
        probas = np.random.multinomial(1, pred, 1)
        return np.argmax(probas)
    else:
        return np.argmax(pred)
        #return np.random.choice(range(0,len(pred)), p=pred)

    # if random.randint(0,100) < prob_tresh:
    #     return np.random.choice(range(0,len(pred[0])), p=pred[0])
    # else:
    #     return np.argmax(pred[0])


def next_token_generator(model, ind_tokens, voc, max_len, prob_tresh=0.5, temperature=0):
    current_seq = ind_tokens
    while 1:
        #to_print=[voc[ind] for ind in current_seq]
        #print('CURRENT SEQ:',to_print)
        n_t = next_token(model, current_seq, voc, max_len, prob_tresh=prob_tresh, temperature=temperature, circular=True)
        yield n_t

        current_seq.append(n_t)
        current_seq = current_seq[1:]


def token_sequence_to_text(input_seq):
    out = ''

    for token in input_seq:
        if token == '<pt>':
            out = out[:-1] + '. '
        elif token == '<cm>':
            out = out[:-1] + ', '
        elif token == '<ai>':
            out += '¿'
        elif token == '<ci>':
            out = out[:-1] + '? '
        elif token == '<nl>':
            out += '\n'
        elif token[-1] == ':':
            out += token[:-1] + ' '
        elif token[-1] == '+':
            out += token[:-1]
        else:
            out += token

    return(out)

def text_to_sequence_tokens(text, voc, voc_ind):

    punctuation = '¿?.\n'
    map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>'}

    letras = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz')
    acc_chars = set(punctuation).union(letras)

    # minúsculas
    text = text.lower()
    # elimina puntuación y pon espacios donde se necesite
    char_tokens = []
    for c in text:
        to_append = ''
        if c in letras or c == ' ':
            to_append = c
        elif c in punctuation:
            to_append = ' ' + map_punctuation[c] + ' '
        char_tokens.append(to_append)
    text = re.sub(' +',' ',''.join(char_tokens))

    word_tokens = text.split(' ') 

    final_tokens = []
    # partelas en tokens

    for word in word_tokens:
        if word + ':' in voc:
            final_tokens.append(word+':')
        elif word in map_punctuation.values():
            final_tokens.append(word)
        else: #intenta silabar
            try:
                to_extend = []
                sils = silabas(word).split('-')            
                __temp = '+ '.join(sils)
                __temp += ':'
                for sil in __temp.split(' '):
                    if sil in voc:
                        to_extend.append(sil)
                    else:
                        __temp_2 = '+ '.join(list(sil[:-1]))
                        __temp_2 += sil[-1]
                        to_extend.extend(__temp_2.split(' '))
                final_tokens.extend(to_extend)
            except TypeError:
                __temp = '+ '.join(list(word))
                __temp += ':'
                final_tokens.extend(__temp.split(' '))

    return final_tokens, [voc_ind[token] for token in final_tokens]


def generate_text(model,seed_text,length,voc,voc_ind,temperature=0,prob_tresh=0.5):

    max_len = model.layers[0].input_shape[1]
    multiplier = 10

    if len(seed_text) > multiplier * max_len: # to lengthy text is splitted randomly
        n = random.randint(0,len(seed_text) - multiplier * max_len)
        seed_text = seed_text[n:n+multiplier*max_len]

    _, initial_seq = text_to_sequence_tokens(seed_text, voc, voc_ind)


    tokens = []

    for token in initial_seq[-max_len:]:
        tokens.append(voc[token])
   
    for i,token in enumerate(next_token_generator(model, initial_seq, voc, max_len, 
        temperature=temperature, prob_tresh=prob_tresh)):

        tokens.append(voc[token])
        sys.stdout.write(token_sequence_to_text([voc[token]]))
        sys.stdout.flush()
        time.sleep(0.1)
        if i == length:
            break

    #print(token_sequence_to_text(tokens))
