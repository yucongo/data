'''
Deep Learning with Python
    Chollet

ch6: Deep learning for text and sequences
    raw imdb http://mng.bz/0tIo
'''

# 6.1.3
imdb_dir = '/Users/fchollet/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

# os.path.join(train_dir, label_type[0])
# '/Users/fchollet/Downloads/aclImdb\\train\\neg'

label_type = ['neg', 'pos']
# >>> text = [t1, t2, t3, t4, ]
# >>> labels = [0, 0, 1, 1, 0, ]

# Listing 6.9
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

df = pd.read_csv(filepath)
texts = df.text.values

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

'''
A Hands-on Tutorial on Neural Language Models
tflearn, spacy, tenflow
git clone https://github.com/dashayushman/neural-language-model.git
# https://dashayushman.github.io/tutorials/2017/08/19/neural-language-model.html

# Thanks to http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
# This method generated n-grams
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

from tflearn.data_utils import to_categorical, pad_sequences

vocab.append(('UNKNOWN', 1))
Idx = range(1, len(vocab)+1)
vocab = [t[0] for t in vocab]

Word2Idx = dict(zip(vocab, Idx))
Idx2Word = dict(zip(Idx, vocab))

'''
index2word = dict([[val, key] for key, val in word_index.items()])