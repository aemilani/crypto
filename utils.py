import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def json_print(obj):
    # create a formatted string of the Python JSON object
    text = json.dumps(obj, sort_keys=True, indent=4)
    print(text)


def plot_candlestick(kline_data, ticker):
    trace1 = {
        'x': kline_data.open_time,
        'open': kline_data.open,
        'close': kline_data.close,
        'high': kline_data.high,
        'low': kline_data.low,
        'type': 'candlestick',
        'name': ticker.upper(),
        'showlegend': True
    }
    data = [trace1]
    layout = go.Layout({
        'title': {
            'text': ticker.upper(),
            'font': {
                'size': 15
            }
        }
    })
    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_line(kline_data, ticker):
    data = [go.Scatter(x=kline_data['open_time'], y=kline_data['close'])]
    layout = go.Layout({
        'title': {
            'text': ticker.upper(),
            'font': {
                'size': 15
            }
        }
    })
    fig = go.Figure(data=data, layout=layout)
    return fig


def plot_volume_profile(profile):
    plt.figure()
    plt.bar(profile['price'], profile['buy_volume'],
            width=profile['price'].iloc[-1] - profile['price'].iloc[-2], color='green')
    plt.bar(profile['price'], profile['sell_volume'],
            width=profile['price'].iloc[-1] - profile['price'].iloc[-2], color='red', bottom=profile['buy_volume'])
    plt.show()
