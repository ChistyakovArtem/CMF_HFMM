import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime


def prepare_df(df, latency):
    df['receive_ts'] = pd.to_datetime(df['receive_ts'])
    df[' exchange_ts'] = pd.to_datetime(df[' exchange_ts'])

    df['time_after_500ms'] = df['receive_ts'] + datetime.timedelta(milliseconds=500)
    df['time_after_latency'] = df['receive_ts'] + datetime.timedelta(milliseconds=latency)

    df['index_after_500ms'] = df.receive_ts.searchsorted(df['time_after_500ms'])
    df['index_after_latency'] = df.receive_ts.searchsorted(df['time_after_latency'])

    df['midprice'] = (df['ethusdt:Binance:LinearPerpetual_ask_price_0'] +
                      df['ethusdt:Binance:LinearPerpetual_bid_price_0']) / 2

    df = df[df['index_after_500ms'] != df['index_after_500ms'].max()]
    df = df[df['index_after_latency'] != df['index_after_latency'].max()]

    df['price_after_500ms'] = df['index_after_500ms'].map(dict(zip(df.index, df.midprice)))
    df['price_after_latency'] = df['index_after_latency'].map(dict(zip(df.index, df.midprice)))

    df['target'] = np.log(df['price_after_500ms'] / df['price_after_latency'])
    df = df.dropna()

    df = df.drop(columns=['time_after_latency', 'time_after_500ms', 'index_after_latency',
                          'index_after_500ms', 'price_after_500ms', 'price_after_latency'])

    return df


def prepare_features(df):
    features = []

    df['ask_cumvol_0'] = df['ethusdt:Binance:LinearPerpetual_ask_vol_0']
    features.append('ask_cumvol_0')
    for i in range(1, 10):
        df['ask_cumvol_' + str(i)] = df['ask_cumvol_' + str(i - 1)] + df['ethusdt:Binance:LinearPerpetual_ask_vol_' + str(i - 1)]
        features.append('ask_cumvol_' + str(i))

    df['bid_cumvol_0'] = df['ethusdt:Binance:LinearPerpetual_bid_vol_0']
    features.append('bid_cumvol_0')
    for i in range(1, 10):
        df['bid_cumvol_' + str(i)] = df['bid_cumvol_' + str(i - 1)] + df['ethusdt:Binance:LinearPerpetual_bid_vol_' + str(i - 1)]
        features.append('bid_cumvol_' + str(i))

    return [df[features], 'target']


def predict(X, model):
    return model.predict(X)


def