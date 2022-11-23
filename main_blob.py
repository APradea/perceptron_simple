#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:28:03 2022

@author: anthelmepradeau
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale


#dataset initialization 
X, y = make_blobs(n_samples = 100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0],1))

#dataset visualisation 
plt.scatter(X[:,0],X[:,1], c = y, cmap = 'summer')
plt.ylabel('largeur de la tige')
plt.xlabel('largeur des petales')
plt.show()

#Standardize data or not
# X_=scale(X)

#Train X and y
X_train = X[:75]
y_train = y[:75]

#Test X and y
X_test = X[75:]
y_test = y[75:]


#Training 
p = Perceptron(X_train, X_test, y_train, y_test)
hist=p.fit(y_train)
pred = p.predict(X_test)
w,b=p.w,p.b

print(w,b)
print(accuracy_score(y_test, pred))

fleur = [-1,2]
print(p.predict(fleur))

x0 = np.linspace(min(X[:,0]), max(X[:,0]), 100)
plt.scatter(X[:,0],X[:,1], c = y, cmap = 'summer')
plt.scatter(fleur[0],fleur[1],color='r')
droite = -(w[0]*x0 + b)/w[1]
plt.plot(x0, droite)
plt.show()

lim = 10
h = 100
W1 = np.linspace(-lim, lim, h)
W2 = np.linspace(-lim, lim, h)

W11,W22 = np.meshgrid(W1,W2)

W_final = np.c_[W11.ravel(), W22.ravel()].T

b= 0 
Z = X.dot(W_final) + b
A = 1/(1+np.exp(-Z))

epsilon = 1e-15
L = (-1/len(y))* np.sum(y*np.log(A + epsilon)+(1-y)*np.log(1-A + epsilon), axis = 0).reshape(W11.shape)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.contourf(W11, W22, L)
plt.colorbar()

plt.subplot(1,2,2)
plt.contourf(W11, W22, L)
plt.scatter(hist[:,0],hist[:,1], c = hist[:,2], marker = 'x', cmap = "magma")
plt.colorbar()
plt.show()






