'''

from keras-bert calssification demo

https://colab.research.google.com/drive/1Lbgn7n7M0UZ4SkI7E2ESHBxkOKuEOeT9


'''

# @title Preparation
!pip install -q keras-bert keras-rectified-adam
!wget -q https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
!unzip -o uncased_L-12_H-768_A-12.zip