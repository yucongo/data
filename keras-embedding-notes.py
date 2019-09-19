'''
keras embedding notes

https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456
'''

vocab_size = len(word_index) + 1

# pad sequence

EMBEDDING_DIM = 100
Embedding(vocab_size, EMBEDDING_DIM, input_length=PAD_LENGTH)
