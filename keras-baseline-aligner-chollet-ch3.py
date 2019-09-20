'''
keras-baseline-aligner-chollet-ch3.ipynb

https://colab.research.google.com/drive/1ldJCI5oYSUGJEvyXXT5cw-sJjnGSFeyH#scrollTo=CzGHCAVszem1&uniqifier=2

%load file.py

%run file.py   # file.py must comply py syntax,  %cd data not allowed
'''

import os
from pathlib import Path
import platform  # portable
import re
print("platform.node():", platform.node())

COLAB = not not re.findall(r'[a-z\d]{12}', platform.node())

# assert COLAB

DIRNAME = '/content' if COLAB else '.'

if COLAB:
    %cd /content
    if not Path('data').exists():
        !git clone https://github.com/yucongo/data.git
    else:
        os.chdir('data')
        !git pull
    os.chdir(DIRNAME)
else:
    %cd data

from google_tr import google_tr

# {keras-baseline-chollet-ch3.py
# end}

# @title !!! Aligner Setting up
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

'''
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000'''

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 500  # ?
VALIDATION_SPLIT = 0.1

# @title Ready texts
import pandas as pd
import numpy as np

from txtfile_to_paras import txtfile_to_paras

from clean_puncts import clean_puncts
from tqdm import tqdm

ENFILE = 'wu_ch1_en.txt'
ZHFILE = 'wu_ch1_zh.txt'

ENFILE = 'lover-ch10_sents45-40_en.txt'
ZHFILE = 'lover-ch10_sents45-40_zh.txt'

ENFILE = 'lover-ch10_en.txt'
ZHFILE = 'lover-ch10_zh.txt'

# INFILE = 'wu_ch1_en_noised.txt'  # !ls wu*ch1*.txt
INFILE = '%s_noised.txt' % Path(ENFILE).stem

# gen_train_dev(ENFILE, 299)

assert Path(ENFILE).exists(), '<%s> does not exist' % ENFILE
assert Path(ZHFILE).exists(), '<%s> does not exist' % ZHFILE
assert Path(INFILE).exists(), '<%s> does not exist' % INFILE

# INFILE =
# texts =
df_en = pd.read_csv(INFILE)
NB_OUTPUT = df_en.label.value_counts().shape[0]

en_paras = txtfile_to_paras(ENFILE)
en_texts = [clean_puncts(elm).lower() for elm in en_paras]

zh_paras = txtfile_to_paras(ZHFILE)
# len(zh_paras)  # 33

mt_texts = []
for elm in tqdm(zh_paras, leave=1, desc=' gen aux data (mt)'):
    mt_texts += [google_tr(elm, 'zh', 'en')]
mt_texts = [clean_puncts(elm).lower() for elm in mt_texts]

texts = [elm.lower() for elm in df_en.text.values]
texts = [clean_puncts(elm) for elm in texts]

labels = df_en.label.values
labels0 = labels[:]
df_en.head(), texts[:2], mt_texts[:2]

# @title Preparing data
# @markdown INFILE = 'wu_ch1_en_noised.txt'

# tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts + mt_texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

print('Found %s unique tokens.' % len(word_index))
# ch1 1626 unique tokens
# ch1 1938 (wit mt_texts) unique tokens
# ch1 2067 (199, wit mt_texts unique tokens

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

assert data.shape[0] == labels.shape[0], ' data.shape[0] and labels.shape[0] mismatch'

# ch1 gen_noised_doc(numb=199)
# Shape of data tensor: (4417, 500)
# Shape of label tensor: (4417, 30)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]

labels = labels[indices]
labels0 = labels0[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

len_lst = [len(elm) for elm in texts]
print(max(len_lst), np.mean(len_lst), len_lst[:10])  # 1530 474.8
print('\n*** NB_OUTPUT: %s ***\n' % NB_OUTPUT)

# @title Define mode and Train
# @markdown INFILE = 'wu_ch1_en_noised.txt'
# @markdown refer to Chollet Listing 3.3 3.8
from keras import models
from keras import layers
from keras.layers import Flatten, Dense

'''
# https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
'''
from keras.layers.embeddings import Embedding

EMBEDDING_DIM = 3 * NB_OUTPUT  # 2-5 times NB_OUTPUT
EMBEDDING_DIM = 150  # 2-5 times NB_OUTPUT
EMBEDDING_DIM = 100  # 2-5 times NB_OUTPUT

print(" EMBEDDING_DIM: ", EMBEDDING_DIM)

model = models.Sequential()

model.add(
    # layers.Dense(4, activation='relu', input_shape=(MAX_SEQUENCE_LENGTH,)),
    Embedding(
        vocab_size,
        EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=1,
    ),
)
model.add(Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(NB_OUTPUT, activation='softmax'))

# markdown epochs

model.compile(
    # optimizer='rmsprop',
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'],
)
print(model.summary())

history = model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=256,
    validation_data=(x_val, y_val),
)

# @title Plot Losses
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
val_acc = history_dict['val_acc']
epochs = range(1, len(val_acc) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# @title Plot Acc
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc_values, 'r', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# @title Prediction
sequences_mt = tokenizer.texts_to_sequences(mt_texts)
data_mt = pad_sequences(sequences_mt, maxlen=MAX_SEQUENCE_LENGTH)

# @title Aligning
pred_proba = model.predict_proba(data_mt)
print(pred_proba[:2])
max_ = np.max(pred_proba, axis=1)
argmax = np.argmax(pred_proba, axis=1)
print(np.max(pred_proba, axis=1))
print(np.argmax(pred_proba, axis=1))
print('mean: ', np.mean(pred_proba, axis=1))

# @title Evaluation
argmax, max_, str([[idx, argmax[idx], max_[idx]] if elm > 0.1 else '' for idx, elm in enumerate(max_)])
align_triples = sorted([[idx, argmax[idx], max_[idx]] if elm >= 0.0 else '' for idx, elm in enumerate(max_)], key=lambda val: val[2], reverse=1)
idx = -1
argmax, max_, align_triples

idx += 1; idx, align_triples[idx], mt_texts[align_triples[idx][0]], zh_paras[align_triples[idx][0]], en_texts[align_triples[idx][1]]

# @title Deliver
len_en = len(en_texts)
len_mt = len(mt_texts)
scale = len_en/len_mt
delta = 3
from pprint import pprint
for idx, elm in enumerate(align_triples):
    if abs(scale * elm[0] - (elm[1] - 1)) <= delta:
        print('\t\t=== %s ===' % idx)
        # pprint([elm, mt_texts[elm[0]], zh_paras[elm[0]], en_texts[elm[1] - 1]])
        pprint([elm, mt_texts[elm[0]], zh_paras[elm[0]], en_texts[elm[1]]])

# @title Plot Alignment Matrix Heattmap
if 'hamming' in platform.node():
    try:
        %matplotlib qt4
    except Exception as exc:
        print(exc)
import matplotlib.pyplot as plt
# plt.plot(pred_proba)

import seaborn as sns
# df_iris = sns.load_dataset('iris')
# sns.kdeplot(df_iris.sepal_width, df_iris.sepal_length, cmap="Reds", shade=True, bw=.15)

plt.contourf(pred_proba, levels=20, cmap="gist_heat_r")
plt.colorbar()
