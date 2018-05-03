# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:23:47 2018

@author: Arthur
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X,y = make_blobs()

colors = {0:'C0', 1:'C1', 2:'C2', 3:'C3', 4:'C4', 5:'C5', 6:'C6', 7:'C7', 8:'C8', 9:'C9'}

plt.figure()
for x1, x2, k in zip(X[:,0], X[:,1], y):
    color = colors[k]
    plt.plot(x1, x2, marker='o', c=color)
    
hidden_units = [10, 5]
hidden_func='tanh'
output_func='softmax'

NN_clf = ANN_clf(hidden_units, hidden_func, output_func, epoch=5000, learning_rate=0.1)

NN_clf.fit(X, y)

y_pred = NN_clf.predict(X)

test_accuracy = accuracy_score(y, y_pred)

print('Train accuracy :', train_accuracy)