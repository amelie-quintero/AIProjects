import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.layers import TextVectorization
from keras.utils import pad_sequences

training_data = pd.read_csv('data/training.csv')

training_data = training_data.drop(['Unnamed: 0'], axis=1)

le = preprocessing.LabelEncoder()
le.fit(training_data['label'])
training_data['label'] = le.transform(training_data['label'])