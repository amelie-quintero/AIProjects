import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.layers import TextVectorization
from keras.utils import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
training_data = pd.read_csv(BASE_DIR / 'data' / 'training.csv')

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

padded_sequences = pad_sequences(sequences, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)
training_sequences = padded_sequences[split:training_size]
testing_sequences = padded_sequences[0:split]
training_labels = labels[split:training_size]
testing_labels = labels[0:split]

training_sequences = np.array(training_sequences)
testing_sequences = np.array(testing_sequences)

embedding_index = {}
with open(BASE_DIR / 'data' / 'glove.6B.50d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = keras.Sequential([
    keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False),
    keras.layers.Dropout(0.2),
    keras.layers.Conv1D(64, 5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=4),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()