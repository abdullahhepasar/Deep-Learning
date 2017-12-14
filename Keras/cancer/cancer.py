# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 23:03:00 2017

@author: abdullahhepasar
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
import keras.optimizers

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?', -99999, inplace='true')
#veri.drop(['id'], axis=1)
veriyeni = veri.drop(['1000025'],axis=1)

imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
veriyeni = imp.fit_transform(veriyeni)


giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]

#relu    --> loss: 1.4417 - acc: 0.7414 - val_loss: 0.4226 - val_acc: 0.7143
#sigmoid --> loss: 1.3440 - acc: 0.5161 - val_loss: 0.5376 - val_acc: 0.9786
#tanh    --> loss: 0.8408 - acc: 0.8333 - val_loss: 0.1406 - val_acc: 0.9357

#sigmoid
#tanh
#Relu

model = Sequential()
model.add(Dense(10,input_dim=8))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#lr = learning Rate ile epochs değerleri ters orantılıdır.
optimizer = keras.optimizers.sgd(lr=0.01)
#optimizer
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(giris,cikis,epochs=100,batch_size=32,validation_split=0.20)


tahmin = np.array([5,10,6,1,10,4,4,10]).reshape(1,8)
print("=================================================")
print("Kanser Olasılığı: ")
print(model.predict_classes(tahmin))
print("=================================================")
