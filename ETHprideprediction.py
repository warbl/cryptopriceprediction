# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 22:12:34 2021

@author: alexl
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

hist = pd.read_csv('D:\CryptoMachineLearningProject\Data\coin_Bitcoin.csv')
hist = hist.set_index('Date')

target_col = 'Close'
"""
maybe test roi as target here
"""

hist = hist.drop(['SNo', 'Name', 'Symbol'], axis = 1)
hist['average'] = (hist['High'] + hist['Low'])/2
hist['momentum'] = (hist['Close'] - hist['Open'])/hist['Open']

print(hist.head(5))


def train_test_split(df, test_size = .2):
    split_row = len(df) - int (test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(hist, test_size = .2)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    
def normalise(df):
    # Assuming same lines from your example
    normalized_df = df
    cols_to_norm = ['High','Low','Open','Close','Volume','Marketcap']
    normalized_df[cols_to_norm] = normalized_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    return normalized_df
    
line_plot(train[target_col], test[target_col], 'training', 'test', title='')


def extract_window_data(df, window_len=5):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        tmp = normalise(df)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10, test_size=0.2):
    train_data, test_data = train_test_split(df, test_size=test_size)
    X_train = extract_window_data(train_data, window_len)
    X_test = extract_window_data(test_data, window_len)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    """
    Trying to fix an error here where can't convert NumPy array to Tensor'
    """
    X_train=np.asarray(X_train).astype(np.int)
    y_train=np.asarray(y_train).astype(np.int)
    X_test=np.asarray(X_test).astype(np.int)
    y_test=np.asarray(y_test).astype(np.int)

    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear', dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(36)
window_len = 5
test_size = 0.2
lstm_neurons = 100
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.3
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = prepare_data(
    hist, target_col, window_len=window_len, test_size=test_size)
model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
print(mean_absolute_error(preds, y_test))

preds = test[target_col].values[:-window_len] * (preds+1)
preds = pd.Series(index=targets.index, data=preds)
line_plot(targets, preds, 'actual', 'prediction', lw=3)