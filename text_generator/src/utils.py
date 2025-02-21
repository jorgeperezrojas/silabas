# coding: utf-8
import numpy as np
import random
import re
from separador_silabas import silabas
import sys
import time
from scipy import spatial
from nltk.metrics import distance


np.random.seed(59)
random.seed(59)

from datetime import datetime

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
        counter = 0

        total_number_of_examples = 0

        for ind in ind_tokens:
            current_par.append(ind)
            if ind == split_symbol_index:
                counter += 1
            if counter == paragraphs_to_join:
                self.paragraphs.append(current_par)
                counter = 0
                current_par = []
        # add a possible final paragraph
        if current_par != []:
            self.paragraphs.append(current_par)
            
        par_sizes = [len(par) for par in self.paragraphs]

        avg_length = np.average(par_sizes)
        # update steps per epoch
        self.steps_per_epoch = (avg_length - max_len)*len(self.paragraphs) / batch_size


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
            # limits
            if len(par) < int(max_len/2) + 2: # ignore paragraphs that are too short
                continue

            i = random.randint(int(max_len/2), len(par) - 2) # pick an index in the paragraph between max_len/2 and len(par) - 2

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
    def __init__(self, model,voc,voc_ind,split_symbol_index,
        seed_text='es muy bueno que de vez en cuando nos preocupemos de vernos bien para mantener la atracción de nuestra pareja . ',
        temperature=0.5,prob_tresh=0.2,mask_value = 0,input_mode='normal'):

        self.model = model
        self.voc = voc
        self.voc_ind = voc_ind
        self.seed_text = seed_text
        self.temperature = temperature
        self.prob_tresh = prob_tresh
        self.mask_value = mask_value
        self.split_symbol_index = split_symbol_index
        self.input_mode = input_mode

    def generate_text(self,length=100, mode='batch'):

        seed_text = self.seed_text

        max_len = self.model.layers[0].input_shape[1]
        _, initial_seq = text_to_sequence_tokens(seed_text, self.voc, self.voc_ind)

        if len(initial_seq) >= max_len:
            initial_seq = initial_seq[-max_len:]
        else:
            initial_seq = [0] * (len(initial_seq) - max_len) + initial_seq

        text_tokens = [self.voc[token] for token in initial_seq]
        input_tokens = initial_seq
        output_tokens = initial_seq

        for i in range(0,length):
            # first generate input tensor
            n_features = len(self.voc)

            if self.input_mode == 'normal':
                X = np.zeros((1, max_len, n_features), dtype = np.bool)
                for k, j in enumerate(input_tokens):
                    X[0, k, j] = 1
            elif self.input_mode == 'sparse':
                X = np.zeros((1, max_len), dtype = np.int32)
                for k,j in enumerate(input_tokens):
                    X[0, k] = j

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


################################################################
################################################################
################################################################
################################################################
################################################################

class PredictorParByParReal:
    def __init__(self, model,voc,voc_ind,split_symbol_index,
        use_random_seed = True,
        number_of_random_sentences = 2,
        input_raw_text = '../data/raw/horoscopo_raw.txt',
        seed_text='su estúpido orgullo hará que usted se quede absolutamente solo . si no cambia , difícilmente logrará una mejora en su calidad de vida . ',
        max_temperature=1,
        min_temperature=0.3,
        min_prob_tresh=0.2,
        max_prob_tresh=0.6,
        max_sentences=4,
        multiplier = 0.6,
        mask_value = -1,
        exploration_breath = 10,
        exploration_depth = 1,
        input_mode='normal',
        word_vectors_model=None):

        np.random.seed()
        random.seed(datetime.now())

        self.model = model
        self.voc = voc
        self.voc_ind = voc_ind
        self.seed_text = seed_text
        self.max_temperature = max_temperature
        self.min_temperature = min_temperature
        self.min_prob_tresh = min_prob_tresh
        self.max_prob_tresh = max_prob_tresh
        self.mask_value = mask_value
        self.split_symbol_index = split_symbol_index
        self.input_mode = input_mode
        self.max_sentences = max_sentences
        self.last_used_seed = seed_text
        self.use_random_seed = use_random_seed
        self.multiplier = multiplier
        self.number_of_random_sentences = number_of_random_sentences
        self.exploration_breath = exploration_breath
        self.exploration_depth = exploration_depth
        self.wvm = word_vectors_model

        if use_random_seed == True:
            self.text_lines = open(input_raw_text).read().split('\n')

        self.max_len = self.model.layers[0].input_shape[1]



    def generate_text(self, mode='batch', 
        new_multiplier = False, multiplier = 0.6, 
        new_diversity = False, diversity = 1, similar_to = 'amor'):

        voc_ind = self.voc_ind
        max_len = self.max_len
        voc = self.voc

        if self.use_random_seed == True:
            ### elige self.number_of_random_sentences frases random
            text = ''
            for _ in range(0,self.number_of_random_sentences):
                line = self.text_lines[random.randint(0,len(self.text_lines))]
                sentences = line.strip().split('.')[:-1]
                if len(sentences) < 1:
                    continue
                i = random.randint(0,len(sentences)-1)
                sentence = sentences[i]
                if sentence.strip() == '.':
                    continue
                text += sentence + '.'

            seed_text = text
        else:
            seed_text = self.seed_text

        seed_tokens, initial_seq = text_to_sequence_tokens(seed_text, self.voc, self.voc_ind)

        # esto es para borrar el ultimo caracter que siempre era ":"
        initial_seq = initial_seq[:-1]

        if len(initial_seq) >= max_len:
            initial_seq = initial_seq[0:max_len]
        else:
            initial_seq = [self.mask_value] * (max_len - len(initial_seq)) + initial_seq


        text_tokens = [self.voc[token] for token in initial_seq if token != self.mask_value]
        input_tokens = initial_seq

        if mode == 'interactive':
            print(initial_seq)
            sys.stdout.write('seed /// ')
            sys.stdout.write(''.join(text_tokens))
            sys.stdout.write('\n')
            sys.stdout.flush()
            sys.stdout.write('to drop ///')

        temper = self.max_temperature
        prob_tresh = self.min_prob_tresh

        n_features = len(self.voc)
        pred_token = -100


        ### generate until the first point
        while pred_token != voc_ind['<pt>']:
            X = np.zeros((1, max_len, n_features), dtype = np.bool)
            for k, j in enumerate(input_tokens):
                if j == self.mask_value:
                    continue
                else:
                    X[0, k, j] = 1
            pred_token = sample_token(self.model.predict(X, verbose=0), temper, prob_tresh)
            input_tokens = input_tokens[1:] + [pred_token]

            if mode == 'interactive':
                sys.stdout.write(self.voc[pred_token])
                sys.stdout.flush()

        #print('//////')
        if mode == 'interactive':
            sys.stdout.write('\n')
            sys.stdout.flush()


        if new_diversity == True:
            temper = diversity
        else:
            temper = self.max_temperature

        max_temper = temper

        if new_multiplier == True:
            mult = multiplier
        else:
            mult = self.multiplier

        # we can generate text now:
        count_sentences = 0
        output_tokens = []
        # put the prefix text generated so far
        pref = ''
        while count_sentences < self.max_sentences:

            X = np.zeros((1, max_len, n_features), dtype = np.bool)
            for k, j in enumerate(input_tokens):
                if j == self.mask_value:
                    continue
                else:
                    X[0, k, j] = 1

            #pred_token = sample_token(self.model.predict(X, verbose=0), temper, prob_tresh)
            pred_token = new_sample_token(self.model.predict(X, verbose=0), temper, prob_tresh)
            #pred_token = self.real_sample_token(self.model.predict(X, verbose=0), 
            #    self.exploration_breath, temperature=temper, prob_tresh=prob_tresh, mode=mode, 
            #    similar_to = similar_to, pref = pref)

            if pred_token == self.split_symbol_index and count_sentences <= self.max_sentences / 2:
                # count_sentences += 1
                temper = max_temper 
                input_tokens = input_tokens # does not change the set of input tokens

            elif pred_token == self.split_symbol_index:
                print('FIN')
                break

            else:
                if pred_token == voc_ind['<pt>']:
                    count_sentences += 1
                    temper = max_temper
                    pref = ''

                elif pred_token == voc_ind['<cm>']:
                    temper = self.min_temperature

                else:
                    temper = max(temper * mult, self.min_temperature)
                
                output_tokens.append(pred_token)
                input_tokens = input_tokens[1:] + [pred_token]

                if mode == 'interactive':
                    sys.stdout.write(self.voc[pred_token])
                    sys.stdout.flush()
            
            word = voc[pred_token]
            if word[-1] != '>':
                if len(pref) == 0 or pref[-1] == '+':
                    pref += word[:-1]
                else:
                    pref += ' ' + word[:-1]
                    



        return output_tokens # , token_sequence_to_text(output_tokens)

    def real_sample_token(self, pred, number_of_examples, temperature=0.001, 
        prob_tresh = 0.5, mode = 'non_interactive', similar_to = 'amor', pref = ''):
        pred = pred[0]

        # if random.random() > prob_tresh:
        #     pred = np.asarray(pred).astype('float64')
        #     pred = np.log(pred) / temperature
        #     exp_pred = np.exp(pred)
        #     pred = exp_pred / np.sum(exp_pred)
        
        #     out = np.random.choice(len(pred), number_of_examples,  replace=True, p=pred)
        #     if mode == 'interactive':
        #         print(str([str(i) + ': ' + self.voc[x] for i,x in enumerate(out)]))
        #         i = int(input())
        #         return out[i]
        #     return out[0]
        # else:
        #     return np.argmax(pred)

        out = np.random.choice(len(pred), number_of_examples,  replace=False, p=pred)
        if mode == 'interactive':
            print('possibilities: ' + str([str(i) + ': ' + self.voc[x] for i,x in enumerate(out)]))
            print('prefix = ' + pref)

        # if current sentence is too long and a <pt> is adviced, just return <pt>        
        if len(pref) > 40 and (self.voc[out[0]] == '<pt>' or self.voc[out[0]] == '<nl>'):
            return out[0]

        # if current sentence is too long and a <cm> is adviced, just return <cm>
        if len(pref) > 20 and self.voc[out[0]] == '<cm>':
            return out[0]

        ### pick the most similar
        max_value = 0.5
        selected_index = 0
        selected_value = out[selected_index]

        

        for i,k in enumerate(out):
            # compute similarity
            sim = similarity(self.wvm, similar_to, pref + ' ' + self.voc[k][:-1])
            
            if mode == 'interactive':
                print('comparing with: ' + self.voc[k][:-1] + ', value: ' + str(sim))
            if sim > max_value + 0.05:
                selected_index = i
                selected_value = k 
                max_value = sim

        if mode == 'interactive':
            print('selected: ' + self.voc[selected_value] + ' ' + str(out[selected_index]))
            input()

        return out[selected_index]

################################################################
################################################################
################################################################
################################################################
################################################################



def similarity(model,string1,string2):
    vec1 = avg_feature_vector(string1.split(), model)
    vec2 = avg_feature_vector(string2.split(), model)
    return 1 - spatial.distance.cosine(vec1,vec2)


def avg_feature_vector(words, model):
        #function to average all words vectors in a given paragraph
        featureVec = np.zeros((model.vector_size,), dtype="float32")
        nwords = 0

        for word in words:
            try: # try to compute the word vectors
                vec = model[word]
            except KeyError:
                continue
            nwords = nwords+1
            featureVec = np.add(featureVec, vec)

        if(nwords>0):
            featureVec = np.divide(featureVec, nwords)
        return featureVec
   
        
 
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


def new_sample_token(pred, temperature=0, prob_tresh = 0.5):
    pred = pred[0]

    pred = np.asarray(pred).astype('float64')
    pred = np.log(pred) / temperature
    exp_pred = np.exp(pred)
    pred = exp_pred / np.sum(exp_pred)
    
    out = np.random.choice(len(pred), 3, replace=False, p=pred)

    if random.random() < prob_tresh:
        # if mode == 'interactive':
        #    # print('\nnot selecting ' + some.voc[out[0]])
        #    # print('chosing between ' + some.voc[out[1]] + ' ' + some.voc[out[2]])
        #    pass
        if random.random() < 0.7:
            return out[1]
        else:
            return out[2]
    else:
        return out[0]


def sample_token(pred, temperature=0, prob_tresh = 0.5):
    pred = pred[0]

    if random.random() > prob_tresh:
        pred = np.asarray(pred).astype('float64')
        pred = np.log(pred) / temperature
        exp_pred = np.exp(pred)
        pred = exp_pred / np.sum(exp_pred)
        #
        #probas = np.random.multinomial(1, pred, 1)
        #return np.argmax(probas)
        out = np.random.choice(len(pred), p=pred)
        return out
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
    upper = True

    for token in input_seq:
        if token == '<pt>':
            upper = True
            out = out[:-1] + '. '
        elif token == '<cm>':
            out = out[:-1] + ', '
        elif token == '<ai>':
            out += '¿'
        elif token == '<ci>':
            out = out[:-1] + '? '
            upper = True
        elif token == '<nl>':
            out += '\n'
            upper = True
        else:
            to_add = ''
            if upper == True:
                token = token[0].upper() + token[1:]
                upper = False

            if token[-1] == ':':
                to_add = token[:-1] + ' '
            elif token[-1] == '+':
                to_add = token[:-1] 
            else:
                to_add = token
            
            out += to_add

    return(out)

def text_to_sequence_tokens(text, voc, voc_ind):

    punctuation = '¿?.,\n'
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

