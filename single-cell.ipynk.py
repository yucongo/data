# ##### setup
%pwd
# !ls data/flairdata
import pandas as pd
INFILE = 'wu_ch3_en.txt'
STEM = Path(INFILE).stem
DIRNAME = 'data/flairdata'

for elm in ['train', 'test', 'dev']:
    globals()[f'{elm.upper()}_FILE'] = f'{elm}_{STEM}'
    print(f'{elm.upper()}_FILE', globals()[f'{elm.upper()}_FILE'])

# ##### install and import packages
try: import flair
except ModuleNotFoundError:
    !pip install flair
    # !pip install --upgrade git+https://github.com/zalandoresearch/flair.git
finally: import flair

from flair.data_fetcher import NLPTaskDataFetcher

from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings, DocumentRNNEmbeddings, DocumentPoolEmbeddings, BertEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

# #### ready data
from pathlib import Path
import os, sys
from os import chdir
if not Path('data').exists():
    !git clone https://github.com/yucongo/data.git
else:
    chdir('data')
    !git pull
    chdir('..')
!ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# ##### define model
# corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')
corpus = NLPTaskDataFetcher.load_classification_corpus(Path(DIRNAME), test_file=TEST_FILE, dev_file=DEV_FILE, train_file=TRAIN_FILE)

# word_embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
# word_embeddings = [FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]
# word_embeddings = [WordEmbeddings('glove'), ]

# document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)
# document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=50, reproject_words=True, reproject_words_dimension=256)

# https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_5_DOCUMENT_EMBEDDINGS.md
# initialize the word embeddings
glove_embedding = WordEmbeddings('glove')
bert_embedding = BertEmbeddings()
# flair_embedding_forward = FlairEmbeddings('news-forward')
# flair_embedding_backward = FlairEmbeddings('news-backward')

'''
document_embeddings = DocumentRNNEmbeddings(
    [glove_embedding, ],
    hidden_size=100,
    reproject_words=True,
    reproject_words_dimension=256,
    # rnn_type='LSTM',
)
# '''

# '''
document_embeddings = DocumentPoolEmbeddings([
        glove_embedding,
        # bert_embedding,
        # flair_embedding_backward,
        # flair_embedding_forward,
    ],
    # fine_tune_mode='none',
)
# '''

# CUDA_LAUNCH_BLOCKING=1
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
trainer = ModelTrainer(classifier, corpus)
trainer.train('./',
    max_epochs=10,
    embeddings_storage_mode='gpu',
)