# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:23:47 2018

@author: Arthur
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score

X,y = make_blobs(500,2,5)
#X,y = make_moons(500, noise=0.3)

alpha = 0.1
hidden_units = [10,5]
hidden_func='tanh'
output_func='softmax'

NN_clf = ANN_clf(alpha=alpha, hidden_units=hidden_units, hidden_func=hidden_func, 
                 output_func=output_func, epoch=5000, learning_rate=0.1, grad_check=True)

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