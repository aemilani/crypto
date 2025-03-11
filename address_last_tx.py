import requests
import time
import datetime


def get_address_last_tx(address='1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ', n_last_tx=10):
    response = requests.get(f'https://chain.api.btc.com/v3/address/{address}/tx')
    last_tx_time = [datetime.datetime.fromtimestamp(
        response.json()['data']['list'][i]['block_time']) for i in range(10)]
    last_tx_diff = [response.json()['data']['list'][i]['balance_diff'] / 10 ** 8 for i in range(n_last_tx)]

    return [(last_tx_time[i], last_tx_diff[i]) for i in range(n_last_tx)]


if __name__ == "__main__":
    print('Default address: 1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ')
    ad = input('Enter address (press Enter for default):')
    print('\nLast transactions:')
    old_res = 0
    while True:
        if len(ad) > 0:
            new_res = get_address_last_tx(address=ad)
        else:
            new_res = get_address_last_tx()
        if new_res[0][1] != old_res:
            for res in new_res:
                print(f'{res[0]}: {res[1]} BTC')
        time.sleep(10)
        old_res = new_res[0][1]
