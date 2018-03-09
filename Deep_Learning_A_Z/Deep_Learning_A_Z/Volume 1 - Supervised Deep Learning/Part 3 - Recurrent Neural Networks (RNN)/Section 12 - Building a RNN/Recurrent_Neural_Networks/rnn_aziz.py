# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 06:02:54 2018

@author: abdel.aziz.sereme

This code will create a deep learning network to predict the stock price google
"""

# Part 1 -- Data Preprocessing
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

# Feature scaling(normalization) 
"""
Here we will apply normalization since we have a 
sigmoid function in the output layer
"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # all new scaled stock prices will be between [0,1]
training_set_scaled = sc.fit_transform(training_set)

# Creating the data structure with the 60 timesteps and 1 output.
"""
At each time t, the rnn will look at the stock prices between t and t-60 stock prices
and based on that info, the rnn will predict the stock price a t+1. So look at the stock
prices 60 previous financial days and predict the stock price the next day, t + 1
"""
x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Part 2 -- Building the model
# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
regressor = Sequential() # Since we are predicting a contineous values

# Adding the FIRST LSTM layers and drop out regularization
"""
The number of units is number of LSTM cells or memory units. Since we want a high
dimensionality of a model, we need a large number of neuron so we choose 50(at first)
Return Sequence, since we are adding multiple layers we set it to true.
Input shape, shape of input corresponding to how the training set is created
Dropout Rate: rate of neuron we want to drop to avoid overfitting. During training
some percentage of neuron will be dropped out during each iteration
"""
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (x_train.shape[1], 1 )))
regressor.add(Dropout(0.2))

# Adding SECOND LSTM and dropout  regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding THIRD LSTM and dropout  regularization
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding FOURTH LSTM and dropout  regularization
regressor.add(LSTM(units = 50, return_sequences=False))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1)) # Since we are outputing 1 layer which is the stock price

# Compile the RNN with right optimizer and right loss function
"""
Different optimizer work for different Deep learning model. Check keras Lib to get an idea
"""
regressor.compile(optimizer = 'adam' , loss = 'mean_squared_error')

# Fitting the RNN to the training set
"""
input x_train--what is being sent into the network
y_train = the truth
epochs : the number of time the RNN is going to be trained. number of iteration
batch size : batches of stock prices in the RNN. number of stock prices being sent 
into the network. here every 32 batch size, the weight are updated and the next 32 are sent
into network.

"""
regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

# Part 3 -- Making the predictions and visualizing the results
# Getting the real stock price of 2017
dataset_test  = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
# horizontal axis = 1 vertical axis = 0
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) # Reshaping
predicted_stock_price = regressor.predict(x_test) # Predict stock prices
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
