import os
import datetime
import requests
import numpy as np
import pandas as pd
from binance.client import Client
from dateutil import parser
from pycoingecko import CoinGeckoAPI


# Params
BINANCE_API_KEY = 'KihCFmQvn5XcFeaLI153hHwKNd2wHyKuCIBGsXNQXphDz8s771t4Dm6JHjXNstMa'
BINANCE_API_SECRET = 'HEuflUxpxTg5jmG7h79pgSMfNGiWC1NuYMGZQKJuC4t5DYqPTCmdMOwcAcEhICBb'
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

cg = CoinGeckoAPI()

interval_dict = {}
for i in [1, 3, 5, 15, 30]:
    interval_dict[f'{i}m'] = datetime.timedelta(minutes=i)
for i in [1, 2, 4, 6, 8, 12]:
    interval_dict[f'{i}h'] = datetime.timedelta(hours=i)
for i in [1, 3]:
    interval_dict[f'{i}d'] = datetime.timedelta(days=i)
interval_dict['1w'] = datetime.timedelta(weeks=1)
interval_dict['1M'] = datetime.timedelta(weeks=4)


def get_all_binance_coins():
    coins = client.get_all_coins_info()
    return [c['coin'] for c in coins]


def get_all_binance_symbols():
    tickers = client.get_all_tickers()
    return [ticker['symbol'] for ticker in tickers]


def get_asset_balance(asset):
    asset_balance = client.get_asset_balance(asset.upper())
    free = asset_balance['free']
    locked = asset_balance['locked']
    return free, locked


def get_trades(symbol='BTCUSDT'):
    trades = client.get_historical_trades(symbol=symbol, limit=1000)
    trades = pd.DataFrame(trades, columns=trades[0].keys())
    trades['time'] = pd.to_datetime(trades['time'], unit='ms')
    return trades


def get_aggr_trades(symbol='BTCUSDT'):
    trades = client.get_aggregate_trades(symbol=symbol, limit=1000)
    trades = pd.DataFrame(trades)
    trades.columns = ['aggrID', 'price', 'qty', 'firstID', 'lastID', 'time', 'isBuyerMaker', 'isBestMatch']
    trades['time'] = pd.to_datetime(trades['time'], unit='ms')
    return trades


def get_order_book(symbol='BTCUSDT'):
    orderbook = client.get_order_book(symbol=symbol)
    bids = orderbook['bids']
    asks = orderbook['asks']
    return bids, asks


def volume_binance(denom='USDT'):
    data = [p for p in client.get_products()['data'] if p['q'] == denom]
    vols = {}
    for dic in data:
        if dic['cs']:
            vols[dic['b']] = dic['v'] * dic['c']
    vols = {k: v for k, v in sorted(vols.items(), key=lambda item: item[1], reverse=True)}
    return vols


def total_volume():
    data = cg.get_coins_markets('usd')
    vols = {}
    for c in data:
        vols[c['symbol'].upper()] = c['total_volume']
    vols = {k: v for k, v in sorted(vols.items(), key=lambda item: item[1], reverse=True)}
    return vols


def kline_data(symbol='btcusdt', interval='1d'):
    """
    Candlestick data.
    :param symbol:
    :param interval:
    :return: Candlestick DataFrame

    Columns: Open time, Open, High, Low, Close, Volume, Close time, Quote asset volume, Number of trades,
    Taker buy base asset volume, Taker buy quote asset volume, Ignore.
    """
    symbol = symbol.upper()
    if not os.path.exists('data'):
        os.mkdir('data')
    filepath = f'data/{symbol}-{interval}.csv'
    if os.path.isfile(filepath):
        old_data = pd.read_csv(filepath)
        last = str(old_data['close_time'].iloc[-1])
        print(f'Last saved data candle close time: {last}')
    else:
        old_data = pd.DataFrame()
        last = '1 Jan 2017'
        print('No saved data exist.')
    if parser.parse(last) + interval_dict[interval] < datetime.datetime.utcnow():
        new_data = client.get_historical_klines(symbol, interval, last)
        new_data = pd.DataFrame(new_data)
        new_data.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades',
                            'tb_base_av', 'tb_quote_av', 'ignore']
        new_data['close_time'] = new_data['close_time'].apply(lambda x: np.ceil(x / 1000) * 1000)
        new_data['open_time'] = pd.to_datetime(new_data['open_time'], unit='ms')
        new_data['close_time'] = pd.to_datetime(new_data['close_time'], unit='ms')
        if new_data.close_time.iloc[-1] > datetime.datetime.utcnow():
            new_data = new_data.drop(index=new_data.index[-1])
        new_data['open_time'] = new_data['open_time'].apply(lambda x: str(x))
        new_data['close_time'] = new_data['close_time'].apply(lambda x: str(x))
        data = old_data.append(new_data)
        data.to_csv(filepath, index=False)
        data = pd.read_csv(filepath)
    else:
        data = old_data
    data.reset_index(drop=True, inplace=True)
    return data


def fear_and_greed(n_data=0):
    """
    Get Crypto Fear and Greed Index data.
    :param n_data: Number of returned data. 0 --> all data.
    :return: A dictionary with 'data', 'dates', and 'timestamps' of the data, and 'time_until_update'.
    """
    response = requests.get(f'https://api.alternative.me/fng/?limit={n_data}')

    data = [int(dic['value']) for dic in response.json()['data']]
    timestamps = [int(dic['timestamp']) for dic in response.json()['data']]
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    time_until_update = int(response.json()['data'][0]['time_until_update'])

    return {'data': data, 'dates': dates, 'timestamps': timestamps,
            'time_until_update': datetime.timedelta(seconds=time_until_update)}


def prepare_prediction_data(data, input_size):
    x, y = [], []
    n_samples = data.shape[0] - input_size
    for i in range(n_samples):
        x.append(data[i:i + input_size])
        y.append(data[i + input_size])
    x = np.array(x)
    y = np.array(y)
    return x, y


def volume_profile(df, n_bins=10):
    df = df[['open', 'high', 'low', 'close', 'volume']]
    high = np.max(df['high'])
    low = np.min(df['low'])
    d_hl = high - low
    d = d_hl / n_bins
    coords = np.arange(low, high + 0.1, d)
    profile = pd.DataFrame(columns=['price', 'buy_volume', 'sell_volume', 'total_volume'])
    profile['price'] = pd.Series(coords).rolling(2).mean().dropna()
    buy_candles = df.where(df['close'] - df['open'] >= 0).dropna()
    sell_candles = df.where(df['close'] - df['open'] < 0).dropna()
    profile['buy_volume'] = np.histogram(buy_candles['close'], bins=coords, weights=buy_candles['volume'])[0]
    profile['sell_volume'] = np.histogram(sell_candles['close'], bins=coords, weights=sell_candles['volume'])[0]
    profile['total_volume'] = profile['buy_volume'] + profile['sell_volume']
    return profile


def get_daily_volume_profile(data, date, n_bins=50):
    """
    Calculates the volume profile for the given day
        Parameters:
            data (DataFrame): BTC 1m data
            date (str): The day for which the volume profile is to be calculated
            n_bins (int):  Number of bins in the volume profile histogram
        Returns:
            profile (DataFrame): BTC volume profile for the given day
    """
    idx_start = data[data['open_time'] == str(parser.parse(date))].index.values[0]
    idx_end = data[data['open_time'] == str(parser.parse(date) + datetime.timedelta(days=1))].index.values[0]
    day_data = data.iloc[idx_start:idx_end]
    profile = volume_profile(day_data, n_bins=n_bins)
    return profile
