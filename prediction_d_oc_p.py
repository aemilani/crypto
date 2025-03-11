import os
import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend
from sklearn.preprocessing import MinMaxScaler
from dataset import kline_data, prepare_prediction_data
from models import rnn_gru
from PyEMD import CEEMDAN
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def model_eval():
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    btc = kline_data()
    btc['d_oc_p'] = (btc['close'] - btc['open']) / btc['open']

    data = np.array(btc['d_oc_p'])
    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(data).T

    input_size = 20
    train_pred = []
    test_pred = []
    for i in range(imfs.shape[1]):
        print(f'*** IMF {i + 1} / {imfs.shape[1]} ***')
        minmax = MinMaxScaler()
        data_norm = np.squeeze(minmax.fit_transform(np.expand_dims(imfs[:, i], axis=-1)))
        x, y = prepare_prediction_data(data_norm, input_size=input_size)
        n_samples = x.shape[0]
        x_train = np.expand_dims(x[:int(n_samples * 0.85), :], axis=-1)
        x_test = np.expand_dims(x[int(n_samples * 0.85):, :], axis=-1)
        y_train = y[:int(n_samples * 0.85)]
        y_test = y[int(n_samples * 0.85):]

        model = rnn_gru(input_size=input_size)
        ea = EarlyStopping(patience=20)
        cp = ModelCheckpoint('saved_models/gru_d_oc_p_1d.h5')
        cb = [ea, cp]
        history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=cb, validation_split=0.3)

        print(f'Test set MSE: {model.evaluate(x_test, y_test, verbose=0)}')
        y_test_pred = np.squeeze(minmax.inverse_transform(model.predict(x_test)))
        test_pred.append(y_test_pred)

        y_train_pred = np.squeeze(minmax.inverse_transform(model.predict(x_train)))
        train_pred.append(y_train_pred)

        backend.clear_session()

    y_test_pred = np.sum(np.array(test_pred).T, axis=1)
    y_train_pred = np.sum(np.array(train_pred).T, axis=1)

    test_open_prices = np.array(btc['open'])[-y_test_pred.shape[0]:]
    test_close_prices = np.array(btc['close'])[-y_test_pred.shape[0]:]
    train_open_prices = np.array(btc['open'])[input_size:-y_test_pred.shape[0]]
    train_close_prices = np.array(btc['close'])[input_size:-y_test_pred.shape[0]]

    d_oc_p_test = y_test_pred
    d_oc_test = d_oc_p_test * test_open_prices
    test_close_prices_pred = test_open_prices + d_oc_test

    d_oc_p_train = y_train_pred
    d_oc_train = d_oc_p_train * train_open_prices
    train_close_prices_pred = train_open_prices + d_oc_train

    error_train = train_close_prices - train_close_prices_pred
    error_test = test_close_prices - test_close_prices_pred

    n_test_samples = error_test.shape[0]
    all_error = np.concatenate((error_train, error_test))

    ceemdan = CEEMDAN()
    error_imfs = ceemdan.ceemdan(all_error).T

    input_size = 20
    test_pred = []
    for i in range(error_imfs.shape[1]):
        print(f'*** IMF {i + 1} / {error_imfs.shape[1]} ***')
        minmax = MinMaxScaler()
        data_norm = np.squeeze(minmax.fit_transform(np.expand_dims(error_imfs[:, i], axis=-1)))
        x, y = prepare_prediction_data(data_norm, input_size=input_size)
        n_samples = x.shape[0]
        x_train = np.expand_dims(x[:-n_test_samples, :], axis=-1)
        x_test = np.expand_dims(x[-n_test_samples:, :], axis=-1)
        y_train = y[:-n_test_samples]
        y_test = y[-n_test_samples:]

        model = rnn_gru(input_size=input_size)
        ea = EarlyStopping(patience=20)
        cp = ModelCheckpoint('saved_models/gru_d_oc_p_1d.h5')
        cb = [ea, cp]
        history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=cb, validation_split=0.3)

        print(f'Test set MSE: {model.evaluate(x_test, y_test, verbose=0)}')
        y_test_pred = np.squeeze(minmax.inverse_transform(model.predict(x_test)))
        test_pred.append(y_test_pred)

        backend.clear_session()

    error_test_pred = np.sum(np.array(test_pred).T, axis=1)
    test_close_prices_final_pred = test_close_prices_pred + error_test_pred

    print(f'Test set MAE: {mean_absolute_error(test_close_prices, test_close_prices_final_pred)}')
    print(f'Test set MAPE: {mean_absolute_percentage_error(test_close_prices, test_close_prices_final_pred)}')


def predict_tomorrow():
    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    btc = kline_data()
    btc['d_oc_p'] = (btc['close'] - btc['open']) / btc['open']

    data = np.array(btc['d_oc_p'])
    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(data).T

    input_size = 20
    tomorrow = 0
    train_pred = []
    for i in range(imfs.shape[1]):
        print(f'*** IMF {i + 1} / {imfs.shape[1]} ***')
        minmax = MinMaxScaler()
        data_norm = np.squeeze(minmax.fit_transform(np.expand_dims(imfs[:, i], axis=-1)))
        x, y = prepare_prediction_data(data_norm, input_size=input_size)
        x = np.expand_dims(x, axis=-1)

        model = rnn_gru(input_size=input_size)
        ea = EarlyStopping(patience=20)
        cp = ModelCheckpoint('saved_models/gru_d_oc_p_1d.h5')
        cb = [ea, cp]
        model.fit(x, y, batch_size=32, epochs=100, callbacks=cb, validation_split=0.3)

        today = np.expand_dims(data_norm[-input_size:], axis=-1)
        today = np.expand_dims(today, axis=0)
        pred = np.squeeze(minmax.inverse_transform(model.predict(today)))
        tomorrow += pred

        y_train_pred = np.squeeze(minmax.inverse_transform(model.predict(x)))
        train_pred.append(y_train_pred)

        backend.clear_session()

    y_train_pred = np.sum(np.array(train_pred).T, axis=1)

    train_open_prices = np.array(btc['open'])[input_size:]
    train_close_prices = np.array(btc['close'])[input_size:]

    d_oc_p_train = y_train_pred
    d_oc_train = d_oc_p_train * train_open_prices
    train_close_prices_pred = train_open_prices + d_oc_train

    error_train = train_close_prices - train_close_prices_pred

    ceemdan = CEEMDAN()
    error_imfs = ceemdan.ceemdan(error_train).T

    input_size = 20
    tomorrow_error = 0
    for i in range(error_imfs.shape[1]):
        print(f'*** IMF {i + 1} / {error_imfs.shape[1]} ***')
        minmax = MinMaxScaler()
        data_norm = np.squeeze(minmax.fit_transform(np.expand_dims(error_imfs[:, i], axis=-1)))
        x, y = prepare_prediction_data(data_norm, input_size=input_size)
        x = np.expand_dims(x, axis=-1)

        model = rnn_gru(input_size=input_size)
        ea = EarlyStopping(patience=20)
        cp = ModelCheckpoint('saved_models/gru_d_oc_p_1d.h5')
        cb = [ea, cp]
        model.fit(x, y, batch_size=32, epochs=100, callbacks=cb, validation_split=0.3)

        today = np.expand_dims(data_norm[-input_size:], axis=-1)
        today = np.expand_dims(today, axis=0)
        pred = np.squeeze(minmax.inverse_transform(model.predict(today)))
        tomorrow_error += pred

        backend.clear_session()

    tomorrow_d_oc = tomorrow
    tomorrow_close = train_close_prices[-1] + tomorrow_d_oc

    tomorrow_d_oc = tomorrow + tomorrow_error
    tomorrow_close_corr = train_close_prices[-1] + tomorrow_d_oc

    return tomorrow_close, tomorrow_close_corr


if __name__ == '__main__':
    start = datetime.datetime.now()
    tomorrow_close_price, tomorrow_close_price_corrected = predict_tomorrow()
    print(f'Tomorrow Close Price: {tomorrow_close_price}')
    print(f'Tomorrow Close Price with error correction: {tomorrow_close_price_corrected}')
    print(f'Runtime: {datetime.datetime.now() - start}')
