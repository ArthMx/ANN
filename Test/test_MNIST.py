# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:29:24 2018

@author: Arthur
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AdamANN import AdamANN_clf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load MNIST
path = '../Data/MNIST/'

mnist_train = pd.read_csv(path+'mnist_train.csv', header=None)
mnist_test = pd.read_csv(path+'mnist_test.csv', header=None)

X_train = mnist_train.drop(0, axis=1)
y_train = mnist_train[0]

X_test = mnist_test.drop(0, axis=1)
y_test = mnist_test[0]

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('Number of training samples :', len(X_train))
print('Number of cross-validation samples :', len(X_cv))
print('Number of test samples :', len(X_test))

# normalize the data
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_cv = normalizer.transform(X_cv)
X_test = normalizer.transform(X_test)


# set hyperparameters
hidden_units = [1000,1000]
hidden_func = 'relu'
alpha = 0.1
p_dropout = 0.2
epoch = 20
learning_rate = 0.001
learn_decay = 20
batch_size = 512

NN_clf = AdamANN_clf(hidden_units, hidden_func, alpha, p_dropout, epoch, learning_rate, 
                     learn_decay, batch_size)

NN_clf.fit(X_train, y_train)

y_train_pred = NN_clf.predict(X_train)
y_cv_pred = NN_clf.predict(X_cv)

train_acc = accuracy_score(y_train_pred, y_train)
cv_acc = accuracy_score(y_cv_pred, y_cv)

print('train accuracy : ', train_acc)
print('CV accuracy : ', cv_acc)

print(classification_report(y_cv, y_cv_pred))