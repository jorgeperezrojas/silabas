import re
from separador_silabas import silabas
from collections import Counter
import random

random.seed(57)

#word_freq_percent = 0.02
#sil_freq_percent = 0.2
word_voc_size = 500
sil_voc_size = 300
val_percentage = 10

# raw data file
filename = '../data/raw/biblia_ntv.txt'
out_filename_pref = '../data/biblia_ntv_'

punctuation = '¿?.\n'
map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>'}

letras = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz')
acc_chars = set(punctuation).union(letras)

text = open(filename).read()

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
# computa las palabras más comunes
word_cnt = Counter()
for word in word_tokens:
    word_cnt[word] += 1

#word_most_freq = [word for word,_ in word_cnt.most_common()[:int(len(word_cnt)*word_freq_percent)]]
word_most_freq = set([word for word,_ in word_cnt.most_common()[:word_voc_size]])
#word_most_freq = set(word_cnt.most_common()[:word_voc_size])

sil_cnt = Counter()
for word in word_tokens:
    if word not in word_most_freq and word not in map_punctuation.values():
        try:
            sils = silabas(word).split('-')
            __temp = '+ '.join(sils)
            __temp += ':'
            for sil in __temp.split(' '):
                sil_cnt[sil] += 1
        except TypeError:
            pass

sil_most_freq = [sil for sil,_ in sil_cnt.most_common()[:sil_voc_size] if sil not in word_most_freq]
#sil_most_freq = set(sil_cnt.most_common()[:sil_voc_size])

final_tokens = []
for word in word_tokens:
    if word in word_most_freq or word in map_punctuation.values():
        word_to_append = word
        if word not in map_punctuation.values():
            word_to_append += ':'
        final_tokens.append(word_to_append)
    else:
        # trata de silabar
        try:
            to_extend = []
            sils = silabas(word).split('-')            
            __temp = '+ '.join(sils)
            __temp += ':'
            for sil in __temp.split(' '):
                if sil in sil_most_freq:
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

# computa el vocabulario
voc_freq = Counter(final_tokens).most_common()

_length = 1500
_inicio = random.randint(0,len(final_tokens)-_length)
print('SAMPLE:')
print(' '.join(final_tokens[_inicio:_inicio+_length]))
print()
print('FINAL CORPUS SIZE:', len(final_tokens))
print('FINAL VOC SIZE:', len(voc_freq))

lines = ' '.join(final_tokens).split('<nl>')

with open(out_filename_pref + 'voc.txt', 'w') as outfile:
    for token,_ in voc_freq:
        outfile.write(token)
        outfile.write('\n')

with open(out_filename_pref + 'train.txt', 'w') as outfile_train, open(out_filename_pref + 'val.txt','w') as outfile_val:
    for line in lines:
        p = random.choice(range(0,100))
        if p <= val_percentage:
            outfile_val.write(line.strip())
            outfile_val.write(' <nl>\n')
        else:
            outfile_train.write(line.strip())
            outfile_train.write(' <nl>\n')
