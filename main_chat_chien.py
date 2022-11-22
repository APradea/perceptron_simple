#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 00:01:12 2022

@author: anthelmepradeau
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data()



y_trad = list()

for i in range(len(y_train)):
    if y_train[i] == 0:
        y_trad.append('Cat')
    else : 
        y_trad.append('Dog')
        
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

plt.figure(figsize=(16, 8))
for i in range(1, 11):
    plt.subplot(4,4,i)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(y_trad[i])
    plt.tight_layout()
plt.show()

X_train=X_train.reshape((X_train.shape[0]),-1)
X_train = scale(X_train)
X_test=X_test.reshape((X_test.shape[0]),-1)
X_test = scale(X_test)


p= Perceptron(X_train)
p.fit(y_train)
pred = p.predict(X_test)

print(accuracy_score(y_test, pred))
