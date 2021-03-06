# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:49:42 2018

@author: Arthur
"""

import numpy as np
import matplotlib.pyplot as plt
from AdamANN import AdamANN_clf
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X,y = make_moons(500, noise=0.1)

normalizer = StandardScaler()
X = normalizer.fit_transform(X)

# set hyperparameters
hidden_units = [50,50,50]
hidden_func = 'tanh'
alpha = 0.1
p_dropout = 0
epoch = 20
learning_rate = 0.01
learn_decay = 1
batch_size = 128

NN_clf = AdamANN_clf(hidden_units, hidden_func, alpha, p_dropout, epoch, learning_rate, 
                     learn_decay, batch_size)

NN_clf.fit(X, y)

y_pred = NN_clf.predict(X)

train_accuracy = accuracy_score(y, y_pred)

print('Train accuracy :', train_accuracy)


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    # Predict the function value for the whole grid
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]
    Z = model.predict(X_grid)
    Z = Z.reshape(xx1.shape)
    # Plot the contour and training examples
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Spectral)
    plt.show()

plt.figure()
plot_decision_boundary(NN_clf, X, y)