#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:30:50 2022

@author: anthelmepradeau
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, X, n_iter = 200, alpha = 0.1):
        self.alpha = alpha
        self.n_iter = n_iter
        self.X_train = X

        
        self.w = np.random.randn(X.shape[1],1)
        self.b = 0
        
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    
    def forward_propagation(self, X):
        Z = np.dot(X, self.w) + self.b
        A = self.sigmoid(Z)
        
        return A 
        
    def loss(self, A,y):
        return (-1/len(y))* np.sum(y*np.log(A)+(1-y)*np.log(1-A))
    
    def gradients(self, A, y):
        dw = (1/len(y)) * np.dot(self.X_train.T, A-y)
        db = (1/len(y)) * np.sum(A-y)
        
        return dw, db
        
    def backpropagation(self, A, y):
        dw, db = self.gradients(A, y)
        self.w = self.w - self.alpha*dw
        self.b = self.b - self.alpha * db
        
    def fit(self, y):
        loss = []
        for i in range(self.n_iter):
            A = self.forward_propagation(self.X_train)
            loss.append(self.loss(A, y))
            self.backpropagation(A,y)
        plt.plot(loss, 'ro')
        plt.show()

    def predict(self, X):
        return self.forward_propagation(X) >= 0.5