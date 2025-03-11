import datetime
import numpy as np
from tensorflow.keras import Sequential, backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from PyEMD import CEEMDAN
from dataset import kline_data, prepare_prediction_data, get_daily_volume_profile
from models import rnn_gru


def predict_tomorrow():
    btc_1m = kline_data(interval='1m')
    btc_1d = kline_data(interval='1d')

    n_data = 1000
    data = btc_1d.iloc[btc_1d.index.values[-n_data:]]

    n_bins = 50
    profiles = []
    for idx, row in data.iterrows():
        date = row['open_time'].split()[0]
        profile = get_daily_volume_profile(btc_1m, date, n_bins=n_bins)
        profiles.append(profile)

    pocs = []
    for profile in profiles:
        poc = profile['price'].iloc[profile['total_volume'].argmax()]
        pocs.append(poc)

    d_poc_p = []
    for i in range(1, len(pocs)):
        d_poc_p.append((pocs[i] - pocs[i - 1]) / pocs[i - 1])
    d_poc_p = np.array(d_poc_p)

    ceemdan = CEEMDAN()
    imfs = ceemdan.ceemdan(d_poc_p).T

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
        cp = ModelCheckpoint('saved_models/gru_d_poc_p_1d.h5')
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

    train_open_prices = np.array(pocs)[input_size: - 1]
    train_close_prices = np.array(pocs)[input_size + 1:]

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

    tomorrow_d_oc = tomorrow + tomorrow_error
    tomorrow_poc = train_close_prices[-1] + tomorrow_d_oc

    return tomorrow_poc


if __name__ == '__main__':
    start = datetime.datetime.now()
    tomorrow_poc_price = predict_tomorrow()
    print(f'Tomorrow POC Price: {tomorrow_poc_price}')
    print(f'Runtime: {datetime.datetime.now() - start}')
