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

embedding_dim = 50
max_length = 54
padding_type = 'post'
trunc_type = 'post'
oov_token = "<OOV>"
training_size = 3000
test_portion = 0.1

title = []
text = []
labels = []
for x in range(training_size):
    title.append(training_data['title'][x])
    text.append(training_data['text'][x])
    labels.append(training_data['label'][x])

vectorizer = TextVectorization(output_mode='int')

vectorizer.adapt(title)

vocab = vectorizer.get_vocabulary()
word_index = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

sequences = vectorizer(tf.constant(title))