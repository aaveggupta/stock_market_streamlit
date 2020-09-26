from numpy import array
import streamlit as st
import pandas as pd
import numpy as np
import missingno as msno
# import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM
# from numpy.random import seed
# import time
# import copy
# import chainer
# import chainer.functions as F
# import chainer.links as L
# from plotly import tools
# from keras.models import load_model
from plotly import graph_objs as go
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


# model = load_model('HDFC_Model.h5')

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

pred1 = [[1709.119], [1725.7922], [1746.3138], [1766.9338], [1786.2156],
         [1803.7695], [1819.6552], [1834.151], [1847.6532], [1860.6]]


close = [1772.1, 1809.3, 1787.45, 1792.2, 1751.65, 1821.9, 1794.65,
         1855.95, 1835.1, 1819.4, 1838.1, 1797.75, 1777.8, 1769.35,
         1741.15, 1754.65, 1835.15, 1889.45, 1885.2, 1868.7, 1886.75,
         1886.05, 1941.85, 1886., 1845.55, 1791.6, 1789.9, 1787.8,
         1805.85, 1806.05, 1883.65, 1881.05, 1875.8, 1841.3, 1850.75,
         1897.35, 1877.3, 1810.65, 1781.95, 1739.8, 1781.1, 1776.7,
         1783.75, 1777.7, 1798.95, 1826.75, 1814.2, 1804.85, 1791.7,
         1800.55, 1820.7, 1827.95, 1785.15, 1804.05, 1829.85, 1829.6,
         1816., 1864.1, 1883.25, 1832.6]

pred2 = [1709.119, 1725.7922, 1746.3138, 1766.9338, 1786.2156,
         1803.7695, 1819.6552, 1834.151, 1847.6532, 1860.6]



def plot_fig():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.Date, y=data['Open'], name="stock_open", line_color='deepskyblue'))
    fig.add_trace(go.Scatter(
        x=data.Date, y=data['Close'], name="stock_close", line_color='dimgray'))
    fig.layout.update(
        title_text='Opening and Closing Price of Stock', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig


plot_fig()

st.header('Candelstick Analyser')

fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])
st.plotly_chart(fig)

pred = [1709.119, 1725.7922, 1746.3138, 1766.9338, 1786.2156,
        1803.7695, 1819.6552, 1834.151, 1847.6532, 1860.6]

st.header('Stock Prices Predictor')

option = st.selectbox('Decide the Day of Prediction', [
    'Day-1', 'Day-2', 'Day-3', 'Day-4', 'Day-5', 'Day-6', 'Day-7', 'Day-8', 'Day-9', 'Day-10', ])

if (option == 'Day-1'):
    st.write('Closing price of HDFC will be ' + str(pred[0]))
elif (option == 'Day-2'):
    st.write('Closing price of HDFC will be ' + str(pred[1]))
elif (option == 'Day-3'):
    st.write('Closing price of HDFC will be ' + str(pred[2]))
elif (option == 'Day-4'):
    st.write('Closing price of HDFC will be ' + str(pred[3]))
elif (option == 'Day-5'):
    st.write('Closing price of HDFC will be ' + str(pred[4]))
elif (option == 'Day-6'):
    st.write('Closing price of HDFC will be ' + str(pred[5]))
elif (option == 'Day-7'):
    st.write('Closing price of HDFC will be ' + str(pred[6]))
elif (option == 'Day-8'):
    st.write('Closing price of HDFC will be ' + str(pred[7]))
elif (option == 'Day-9'):
    st.write('Closing price of HDFC will be ' + str(pred[8]))
elif (option == 'Day-10'):
    st.write('Closing price of HDFC will be ' + str(pred[9]))
    
st.header('Prediction Graph for the Next 10 Days')

day_new = np.arange(1, 61)
day_pred = np.arange(61, 71)

close = np.array(close)
pred2 = np.array(pred2)

plt.plot(day_new, close)
plt.plot(day_pred, pred2, 'r')
st.pyplot(plt)

