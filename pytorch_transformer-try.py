'''

https://zhuanlan.zhihu.com/p/75606225

 pytorch-pretrained-bert to pytorch-transformers

from pytorch_transformers import BertTokenizer, BertForSequenceClassification


BERT Fine-Tuning Tutorial with PyTorch
https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO

https://www.kaggle.com/c/cola-in-domain-open-evaluation/download/cola_in_domain_test.tsv
    cola_in_domain_test.tsv
https://raw.githubusercontent.com/nyu-mll/CoLA-baselines/master/acceptability_corpus/raw/in_domain_train.tsv

df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

df.sample(10)

# Create sentence and label lists
sentences = df.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()
    file_utils.py[line:238] 2019-09-15 09:38:23,634 : INFO : https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin not found in cache or force_download set to True, downloading to C:\Users\Public\Documents\Wondershare\CreatorTemp\tmpkcgrbq1y

'''