# -*- coding: utf-8 -*-
"""
Created on Sun May  6 11:44:16 2018

@author: Arthur
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from AdamANN import AdamANN_reg

boston = load_boston()

X = boston['data']
y = boston['target']

print('X shape : ', X.shape)
print('y shape : ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# set hyperparameters
hidden_units = [100,100]
hidden_func = 'relu'
alpha = 0.1
p_dropout = 0
epoch = 1000
learning_rate = 0.01
learn_decay = 5
batch_size = 100

ANN_reg = AdamANN_reg(hidden_units, hidden_func, alpha, p_dropout, epoch, 
                      learning_rate, learn_decay, batch_size)

ANN_reg.fit(X_train, y_train)

y_train_pred = ANN_reg.predict(X_train)
y_test_pred = ANN_reg.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print('Train RMSE : ', np.sqrt(mse_train))
print('Test RMSE : ', np.sqrt(mse_test))


# linear model
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)

y_train_pred_lin = lin_reg.predict(X_train)
y_test_pred_lin = lin_reg.predict(X_test)

mse_train_lin = mean_squared_error(y_train, y_train_pred_lin)
mse_test_lin = mean_squared_error(y_test, y_test_pred_lin)

print('Train RMSE linear regression : ', np.sqrt(mse_train_lin))
print('Test RMSE linear regression : ', np.sqrt(mse_test_lin))

# Random Forest
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)

forest_reg.fit(X_train, y_train)

y_train_pred_forest = forest_reg.predict(X_train)
y_test_pred_forest = forest_reg.predict(X_test)

mse_train_forest = mean_squared_error(y_train, y_train_pred_forest)
mse_test_forest = mean_squared_error(y_test, y_test_pred_forest)

print('Train RMSE Random Forest : ', np.sqrt(mse_train_forest))
print('Test RMSE Random Forest : ', np.sqrt(mse_test_forest))
