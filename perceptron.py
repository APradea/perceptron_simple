#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 22:30:50 2022

@author: anthelmepradeau
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class Perceptron():
    def __init__(self, X_train, X_test, y_train, y_test, n_iter = 10000, alpha = 0.01):
        self.alpha = alpha
        self.n_iter = n_iter
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        

        
        self.w = np.random.randn(X_train.shape[1],1)
        self.w[1], self.w[0] = -7.5,-7.5
        self.b = 0
        
        
    def sigmoid(self,X):
        return 1/(1+np.exp(-X))
    
    def forward_propagation(self, X):
        Z = np.dot(X, self.w) + self.b
        A = self.sigmoid(Z)
        
        return A 
        
    def loss(self, A,y):
        #adding epsilon to avoid log(0)
        epsilon = 1e-15
        return (-1/len(y))* np.sum(y*np.log(A + epsilon)+(1-y)*np.log(1-A + epsilon))
    
    def gradients(self, A, y):
        dw = (1/len(y)) * np.dot(self.X_train.T, A-y)
        db = (1/len(y)) * np.sum(A-y)
        
        return dw, db
        
    def backpropagation(self, A):
        dw, db = self.gradients(A, self.y_train)
        self.w = self.w - self.alpha*dw
        self.b = self.b - self.alpha * db
        
    def fit(self):
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        nb = 10
        history = np.zeros((self.n_iter//nb,3))
        j=0
       
        for i in tqdm(range(self.n_iter)):
            
            A = self.forward_propagation(self.X_train)
            
            if i % nb == 0:
                history[j,0]=self.w[0]
                history[j,1]=self.w[1]
                history[j,2]=self.b
                j += 1
                
                train_loss.append(self.loss(A, self.y_train))
                
                A_test = self.forward_propagation(self.X_test)
                test_loss.append(self.loss(A_test, self.y_test))
                
                y_pred_train = self.predict(self.X_train)
                train_accuracy.append(accuracy_score(y_pred_train, self.y_train))
                
                y_pred_test = self.predict(self.X_test)
                test_accuracy.append(accuracy_score(y_pred_test, self.y_test))
            
           
            
            self.backpropagation(A)
            
        plt.figure(figsize=(12,4))    
        plt.subplot(1,2,1)
        plt.plot(train_loss, color = 'r')
        plt.plot(test_loss, color = 'b')
        
        plt.subplot(1,2,2)
        plt.plot(test_accuracy)
        plt.plot(train_accuracy)
        plt.show()
        return history
    
    def predict(self, X):
        return self.forward_propagation(X) >= 0.5