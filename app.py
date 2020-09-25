from numpy import array
import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from numpy.random import seed
import time
import copy
import chainer
import chainer.functions as F
import chainer.links as L
from plotly import tools
from keras.models import load_model
# from plotly.graph_objs import *
# from plotly.offline import init_notebook_mode, iplot, iplot_mpl


image_path = ('image.jpg')
st.image(image_path, use_column_width=True)

st.title("Finock ~ Stock Price Predictor 📈")
st.header("Welcome to Finock!")
st.markdown(
    "In this Deep Learning application, we have used the historical stock price data for HDFC to forecast their price for the next 10 days.")


DATA_URL = ('./DATASETS/'+'HDFC'+'.csv')


def load_data():
    data = pd.read_csv(DATA_URL)
    return data


data = load_data()

new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

train = new_data[:4112]
test = new_data[4112:]

scaler = MinMaxScaler(feature_range=(0, 1))
values = new_data.values

values = np.array(values)
values = scaler.fit_transform(values)


x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(values[i-60:i, 0])
    y_train.append(values[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1, 1)


X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)


X_test = np.reshape(X_test, (1029, 60, 1))


x_train = np.reshape(x_train, (4052, 60, 1))
X_test = np.reshape(X_test, (1029, 60, 1))


model = load_model('HDFC_Model.h5')

x_input = X_test[340].reshape(1, -1)

x_input = np.asarray(x_input[0])

temp_input = x_input.tolist()


# lst_output = []
# n_steps = 60
# i = 0
# while(i < 11):
#     if(len(temp_input) > 60):
#         # print(temp_input)
#         x_input = np.array(temp_input[1:])
#         #print("{} day input {}".format(i,x_input))
#         x_input = x_input.reshape(1, -1)
#         x_input = x_input.reshape((1, n_steps, 1))
#         # print(x_input)
#         yhat = model.predict(x_input, verbose=0)
#         print
#         ("{} day output {}".format(i, scaler.inverse_transform(yhat)))
#         temp_input.extend(yhat[0].tolist())
#         temp_input = temp_input[1:]
#         # print(temp_input)
#         lst_output.extend(yhat.tolist())
#         i = i+1
#     else:
#         x_input = x_input.reshape((1, n_steps, 1))
#         yhat = model.predict(x_input, verbose=0)
#         # print(yhat[0])
#         temp_input.extend(yhat[0].tolist())
#         # print(len(temp_input))
#         lst_output.extend(yhat.tolist())
#         i = i+1


# # print(lst_output)


# day_new = np.arange(1, 61)
# day_pred = np.arange(61, 72)
# plt.plot(day_new, scaler.inverse_transform(new_data[5081:]))
# plt.plot(day_pred, scaler.inverse_transform(lst_output), 'r')
