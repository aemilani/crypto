import numpy as np
import pandas as pd
from dataset import kline_data
from dateutil import parser


class Portfolio(object):
    def __init__(self, interval='1d', usdt=1000, btc=0, start_time=None):
        self.data = kline_data(interval=interval)
        if start_time:
            self.idx = self.data['open_time'][self.data['open_time'] == str(parser.parse(start_time))].index[0]
        else:
            self.idx = 0
        self.usdt = usdt
        self.btc = btc
        self.history = pd.DataFrame(columns=['open_time', 'balance_usdt', 'balance_btc'])

    def open_time(self):
        return self.data['open_time'].iloc[self.idx]

    def open(self):
        return self.data['open'].iloc[self.idx]

    def high(self):
        return self.data['high'].iloc[self.idx]

    def low(self):
        return self.data['low'].iloc[self.idx]

    def close(self):
        return self.data['close'].iloc[self.idx]

    def close_time(self):
        return self.data['close_time'].iloc[self.idx]

    def volume(self):
        return self.data['volume'].iloc[self.idx]

    def balance_usdt(self):
        return self.usdt + self.btc * self.close()

    def balance_btc(self):
        return self.usdt / self.close() + self.btc

    def buy(self, amount, unit):
        if unit == 'usdt':
            self.usdt = self.usdt - amount
            self.btc = self.btc + amount / self.close()
        elif unit == 'btc':
            self.usdt = self.usdt - amount * self.close()
            self.btc = self.btc + amount

    def sell(self, amount, unit):
        if unit == 'usdt':
            self.usdt = self.usdt + amount
            self.btc = self.btc - amount / self.close()
        elif unit == 'btc':
            self.usdt = self.usdt + amount * self.close()
            self.btc = self.btc - amount

    def next_candle(self):
        if self.open_time() == self.data['open_time'].iloc[-1]:
            print('Last candle reached')
            return
        self.history = self.history.append(pd.DataFrame({
            'open_time': [self.open_time()],
            'balance_usdt': [self.balance_usdt()],
            'balance_btc': [self.balance_btc()]}))
        self.idx += 1


def ma_cross_signal(fast, slow):
    """MAs should be at the shape of price and non-existant indices must be NaN-padded."""
    n = fast.shape[0]
    buys = np.empty((n,))
    buys[:] = np.nan
    sells = np.empty((n,))
    sells[:] = np.nan
    d_fs = fast - slow
    for i in range(1, len(d_fs)):
        if not (np.isnan(d_fs[i]) or np.isnan(d_fs[i-1])):
            if d_fs[i] > 0 and d_fs[i-1] < 0:
                buys[i] = 1
            elif d_fs[i] < 0 and d_fs[i-1] > 0:
                sells[i] = 1
    return buys, sells
