from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense


def rnn_gru(input_size):
    model = Sequential()
    model.add(GRU(64, input_shape=(input_size, 1), return_sequences=True))
    model.add(GRU(16))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss='mse')
    return model


def rnn_lstm(input_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_size, 1), return_sequences=True))
    model.add(LSTM(16))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="adam", loss='mse')
    return model
