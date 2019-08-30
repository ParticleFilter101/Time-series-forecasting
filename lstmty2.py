#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:18:21 2019

@author: shivambhardwaj
"""

import sys 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split, KFold 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics 
from sklearn.metrics import mean_squared_error,r2_score
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout


data = pd.read_csv('occupancy_t4013.csv')
plt.figure()
data.plot(kind = 'line', x = 'timestamp', y = 'value', color = 'red')
plt.show()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	dff = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


values = data.iloc[:,1].values.astype('float32') 
scaler = MinMaxScaler(feature_range=(0, 1))
values = values.reshape(-1,1)
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)


train_X, train_y = reframed.iloc[0:2000, :-1].values.astype('float32'), reframed.iloc[0:2000, -1].values.astype('float32')

test_X, test_y = reframed.iloc[2000:, :-1].values.astype('float32'), reframed.iloc[2000:, -1].values.astype('float32')

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

fit_history = model.fit(train_X, train_y, epochs=200, batch_size=70, 
                    validation_data=(test_X, test_y), verbose=2, shuffle=False)


plt.figure()
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], 1))
inv_yhat = scaler.inverse_transform(yhat)
y_org = test_y.reshape(-1,1)
inv_y = scaler.inverse_transform(y_org)

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


aa=[x for x in range(200)]
plt.figure()
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('Global_active_power', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show()
