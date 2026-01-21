This project is based on the project in https://www.geeksforgeeks.org/nlp/fake-news-detection-model-using-tensorflow-in-python/

The goal of this project is to create a fake news detection algorithm using TensorFlow in Python.

Embeddings were obtained as in the GFG article from https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip

Note 1: Due to deprecation of tf.keras.preprocessing, I switched to using keras.layers.TextVectorization and keras.utils.pad_sequences.

Note 2: I also added the pathlib.Path import to harden the logic around importing the training and embedding data.