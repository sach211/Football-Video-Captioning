#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:33:25 2017

@author: sachi
"""
import pandas as pd

sentence_start = 'SS'
sentence_end = '.'

cnn = pd.read_csv('/home/sachi/Documents/football/labels.csv')
cnn = cnn[['Image No.', 'Goal(post)', 'Ball', 'Player', 'Action', 'Goalkeeper']]
captions = pd.read_csv('/home/sachi/Documents/football/captions.csv')

bool_idx = cnn['Action'] != 0
cnn = cnn[bool_idx]
captions = captions[bool_idx]
cnn = cnn.reset_index()
captions = captions.reset_index()


keys = []
values = []
count = 1
sentence_ids = []

for sentence in captions['caption']:
    sentence = sentence[:-1]
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        if word not in keys:
            keys.append(word)
            values.append(count)
            count = count + 1
keys = keys + [sentence_start, sentence_end]
values = values + [0, count]
vocab_size = count + 1

word_dict = dict(zip(keys, values))
index_dict = dict(zip(values, keys))

inpt = []
oupt = []
for sentence in captions['caption']:
    row = []
    inpt_row = []
    oupt_row = []
    sentence = sentence[:-1]
    sentence = sentence.lower()
    words = sentence.split(' ')
    for word in words:
        row.append(word_dict[word])
    sentence_ids.append(row)
    inpt_row.append(0)
    inpt_row.extend(row[:])
    oupt_row = row[:]
    oupt_row.append(vocab_size - 1)
    inpt.append(inpt_row)
    oupt.append(oupt_row)

cnn_orig = cnn
cnn = cnn[['Goal(post)', 'Ball', 'Player', 'Action', 'Goalkeeper']]    
  
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

seed = 7
np.random.seed(seed)

label = [] 
for row in oupt:
    for col in row:
        l_row = [0] * (vocab_size)
        l_row[col] = 1
        label.append(l_row)
        
features = []
for index, row in enumerate(inpt):
    for col in row:
        f_row = [0] * (vocab_size)
        f_row[col] = 1
        for val in cnn.loc[index]:
            f_row.append(int(val))
        features.append(f_row)
     
label = np.array(label)
features = np.array(features)
#features = np.reshape(features, (features.shape[0], features.shape[1], 1))

model = Sequential()
model.add(Dense(40, input_dim = features.shape[1], init = 'uniform', activation = 'relu'))
model.add(Dense(38, init = 'uniform', activation = 'relu'))
model.add(Dense(46, init = 'uniform', activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(features, label, epochs=100, batch_size=10)


"""Test

x = [0] * (vocab_size)
x[0] = 1
for val in cnn.loc[273]:
            x.append(int(val))
x = np.reshape(x, (1,vocab_size + 5))
pred = []
i = 0
while i != word_dict['.']:
    y = model.predict(x, verbose=0)
    j = np.argmax(y)
    pred.append(j)
    x[0][i] = 0
    x[0][j] = 1
    i = j
for i in pred:
    print index_dict[i]
    
    
    
for l in range(1, len(cnn), 60):
    x = [0] * (vocab_size)
    x[0] = 1
    for val in cnn.loc[l]:
            x.append(int(val))
    x = np.reshape(x, (1,vocab_size + 5))
    pred = []
    i = 0
    while i != word_dict['.']:
        y = model.predict(x, verbose=0)
        j = np.argmax(y)
        pred.append(j)
        x[0][i] = 0
        x[0][j] = 1
        i = j
    sent = ""
    for i in pred:
        if index_dict[i] != "." and sent != "":
            sent = sent + ' '
        sent = sent + index_dict[i]
    print sent
"""


        