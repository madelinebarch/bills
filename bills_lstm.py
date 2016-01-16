'''Train a Bidirectional LSTM on the IMDB sentiment classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_bidirectional_lstm.py

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils.np_utils import accuracy, to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

import load_bills

max_features = 10000
num_cats = 226 
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = load_bills.load_data('data/bills.pkl', nb_words=max_features,
                                                      test_split=0.1)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
y_train = to_categorical(y_train, nb_classes=num_cats)
y_test = to_categorical(y_test, nb_classes=num_cats)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(num_cats))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy', optimizer='adam')

print('Train...')
model.fit(X_train, y_train,
          batch_size=batch_size,
	  show_accuracy=True,
          validation_data=(X_test, y_test),
          nb_epoch=20)


