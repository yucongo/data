'''

keras baseline

chollet ch3.4.5

see also url realpython keras classification

https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
'''
from pathlib import Path
import pandas as pd
imort numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



