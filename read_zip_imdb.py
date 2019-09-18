'''
'''
import zipfile as zf

filepath = r'C:\Users\mike\Downloads\Compressed\aclImdb.zip'

zfile = zf.ZipFile(filepath)
file_list = [elm.filename for elm in zfile.filelist if elm.filename.endswith('.txt') and ('neg' in elm.filename or 'pos' in elm.filename) and elm.filename.startswith('aclImdb')]
# len(file_list) == 5004

# zfile.open(file_list[0]).read().decode()

# train and test
labels = []
texts = []
for elm in file_list:
    text = zfile.open(elm).read().decode()
    texts += [text]
    if 'neg' in elm:
        labels += [0]
    else:
        labels += [1]
# sum(labels) == 25002
# sum([1 for elm in labels if elm == 0]) == 25002

# train only
tr_file_list = [elm.filename for elm in zfile.filelist if elm.filename.endswith('.txt') and ('neg' in elm.filename or 'pos' in elm.filename) and elm.filename.startswith('aclImdb') and 'train' in elm.filename]
labels = []
texts = []
for elm in tr_file_list:
    text = zfile.open(elm).read().decode()
    texts += [text]
    if 'neg' in elm:
        labels += [0]
    else:
        labels += [1]
# sum(labels) == 12501
# sum([1 for elm in labels if elm == 1]) == 12501
# sum([1 for elm in labels if elm == 0]) == 12501

zfile.close()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# 92036

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# Shape of data tensor: (25002, 100)
# Shape of label tensor: (25002,)

index_word = dict([[val, key] for key, val in word_index.items()])
# In [545]: ' '.join([*map(lambda elm: index_word[elm], sequences[0])])
# => texts[0]: lower(), puctuation removed
# texts[0][:50]
# 'Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra a'
# Out[545]: "story of a man who has unnatural feelings for a pig starts out with a opening scene that is a terrific example of absurd comedy a orchestra audience is turned into an insane violent mob by the crazy of it's singers unfortunately it stays absurd the whole time with no general narrative eventually making it just too off putting even those from the era should be turned off the dialogue would make shakespeare seem easy to a third on a technical level it's better than you might think with some good cinematography by future great future stars sally and forrest can be seen briefly"


# In [564]: ' '.join([*map(lambda elm: index_word[elm] if elm > 0 else 'x', data[0])])
# Out[564]: 'x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x this movie had it all action comedy and best of all some of the finest actors gunga din will remain a classic to be enjoyed by all who like good movies excellent picture i have it in my collection'