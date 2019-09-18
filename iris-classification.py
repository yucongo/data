'''
https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/

http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
=> iris.csv
'''

from pathlib import Path

filename = 'iris.csv'
if not Path(filename).exists():
    !wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    !mv iris.data iris.csv
assert Path(filename).exists(), f'<{filenae}> does not exist'

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=4, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# prediction
# model.fit(X, dummy_y, epochs=833)
# model.predict()
_ = '''
# adopt to wu3?
# playground\keras-playground\flair-encoding-finetune.py
# sentence = Sentence('The grass is green .')
# document_embeddings.embed(sentence)
# now check out the embedded tokens.
# sentence embeddings
# sentence.embedding # 512
# classfy to 46: 46 output neurons
# train and evaluation?
# '''


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# ccuracy: 97.33% (4.42%)
# colab Baseline: 96.00% (3.27%)
# win10 Baseline: 96.67% (4.47%)
