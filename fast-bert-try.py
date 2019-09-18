'''

https://github.com/kaushaltrivedi/fast-bert
pip install fast-bert
    # needs torch 1.2.0
    pip install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

    pip install bottleneck  # Failed to build bottleneck

    To install bottleneck on Windows, first install MinGW and add it to your system path. Then install Bottleneck with the commands:

        python setup.py install --compiler=mingw32

    or https://www.lfd.uci.edu/~gohlke/pythonlibs/#bottleneck
    pip install C:\Users\mike\Downloads\Bottleneck-1.2.1-cp36-cp36m-win_amd64.whl
    # OK

    pip install fast-bert

!wget -c https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
# 388 MB (407,727,028 bytes)

https://www.kaggle.com/lapolonio/bert-squad-forked-from-sergeykalutsky
repo = 'model_repo'
with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
    zip_ref.extractall(repo)

torch.cuda.device_count()  # 0 => cpu

'''
from pathlib import Path

import torch
# import apex

from pytorch_pretrained_bert.tokenization import BertTokenizer

from fast_bert.data import BertDataBunch
from fast_bert.learner import BertLearner
# from apex import amp  # need apex

from fast_bert.metrics import accuracy

DATA_PATH = Path('../data/')     # path for data files (train and val)
LABEL_PATH = Path('../labels/')  # path for labels file
MODEL_PATH=Path('../models/')    # path for model artifacts to be stored
LOG_PATH=Path('../logs/')       # path for log files to be stored

# location for the pretrained BERT models
BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
BERT_PRETRAINED_PATH = Path('bert_models/uncased_L-12_H-768_A-12/')

args = {
    "max_seq_length": 512,
    "do_lower_case": True,
    "train_batch_size": 32,
    "learning_rate": 6e-5,
    "num_train_epochs": 12.0,
    "warmup_proportion": 0.002,
    "local_rank": -1,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "loss_scale": 128
}

tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH, do_lower_case=args['do_lower_case'])

databunch = BertDataBunch(DATA_PATH, LABEL_PATH, tokenizer, 
  train_file='train.csv', val_file='val.csv', label_file='labels.csv',
  bs=args['train_batch_size'], maxlen=args['max_seq_length'], 
  # multi_gpu=multi_gpu, 
  multi_label=False)