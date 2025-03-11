import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from dataset import kline_data, prepare_prediction_data
from models import rnn_gru


kline_dataset = kline_data(interval='1d')

n_last_samples = 240
close_prices = np.array(kline_dataset['close'])[-n_last_samples-1:-1]

m = np.min(close_prices)
M = np.max(close_prices)

close_prices_norm = (close_prices - m) / (M - m)

train = close_prices_norm

input_size = 5
x_train, y_train = prepare_prediction_data(train, input_size)
x_train = np.expand_dims(x_train, axis=-1)
gru = rnn_gru(input_size)
ea = EarlyStopping(patience=100)
cb = [ea]
history = gru.fit(x_train, y_train, batch_size=32, epochs=1000, callbacks=cb, validation_split=0.25)

print(f'Validation Loss: {history.history["val_loss"][-1]}')

x_tomorrow = np.reshape(train[-input_size:], (1, input_size, 1))
y_tomorrow_norm_pred = gru.predict(x_tomorrow)
y_tomorrow_pred = y_tomorrow_norm_pred * (M-m) + m

print(f'Predicted Tomorrow Close: {y_tomorrow_pred}')

K.clear_session()
