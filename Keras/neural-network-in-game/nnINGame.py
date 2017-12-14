# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:03:00 2017

@author: abdullahhepasar
"""

import keras
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Convolution2D, BatchNormalization, Flatten
import numpy as np
from keras import optimizers
from keras.models import load_model


giris = np.load('slopedatabig.npy')
print(giris.shape," ", giris.size," ")
cikis = np.load("outdatabig.npy")

giris = np.array(giris).reshape(-1,80,60,3)
cikis = np.array(cikis)
a = np.array([])
for i in range(1000):
    a = np.append(a,cikis[i][0])

cikis = a.reshape(1000,5)

print(cikis[3])

model = Sequential()
model.add(Convolution2D(64, 3, input_shape(80,60,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Convolution2D(64, 5))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5,5)))
model.add(Convolution2D(64, 3))

model.add(Flatten())
model.add(Dense(4000))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(4000))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(5))
model.add(BatchNormalization())
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.summary())
model.fit(giris,cikis, epochs=1, validation_split=0.1)

model.save("saveData")

model.fit(giris, cikis, epochs=10, validation_split=0.10)

model = load_model("saveData")
#keras.models.load_model("saveData")










