import json
from os import path
from enum import Enum
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

# From connecting source file, `import engine` and run `engine.connect()`. Need each connection to be separate
# (see https://stackoverflow.com/questions/3724900/python-ssl-problem-with-multiprocessing)
config_json = json.load(open(os.path.dirname(__file__) + '/../config/btc.json'))
DB = config_json['DB_HISTORY'].split('/')[-1]
engine_runs = create_engine(config_json['DB_RUNS'])

# Decide which exchange you want to trade on (significant even in training). Pros & cons; Kraken's API provides more
# details than GDAX (bid/ask spread, VWAP, etc) which means predicting its next price-action is easier for RL. It
# also has a lower minimum trade (.002 BTC vs GDAX's .01 BTC), which gives it more wiggle room. However, its API is
# very unstable and slow, so when you actually go live you'r bot will be suffering. GDAX's API is rock-solid. Look
# into the API stability, it may change by the time you're using this. If Kraken is solid, use it instead.
class Exchange(Enum):
    GDAX = 'gdax'
    KRAKEN = 'kraken'
EXCHANGE = Exchange.KRAKEN

# see {last_good_commit} for imputes (ffill, bfill, zero),
# alex database


class Data(object):
    def __init__(self, window=300, indicators={}):
        self.window = window
        self.indicators = indicators

        # self.ep_stride = ep_len  # disjoint
        # self.ep_stride = 100  # overlap; shift each episode by x seconds.
        # TODO overlapping stride would cause test/train overlap. Tweak it so train can overlap data, but test gets silo'd

        col_renames = {
            'Timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume_(BTC)': 'volume_btc',
            'Volume_(Currency)': 'volume',
            'Weighted_Price': 'vwap'
        }

        filenames = {
            # 'bitstamp': 'bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv',
            'coinbase': 'coinbaseUSD_1-min_data_2014-12-01_to_2018-06-27.csv',
            # 'coincheck': 'coincheckJPY_1-min_data_2014-10-31_to_2018-06-27.csv'
        }
        primary_table = 'coinbase'
        self.target = f"{primary_table}_close"

        df = None
        for table, filename in filenames.items():
            df_ = pd.read_csv(path.join(path.dirname(__file__), 'populate', 'bitcoin-historical-data', filename))
            col_renames_ = {k: f"{table}_{v}" for k, v in col_renames.items()}
            df_ = df_.rename(columns=col_renames_)
            ts = f"{table}_timestamp"
            df_[ts] = pd.to_datetime(df_[ts], unit='s')
            df_ = df_.set_index(ts)
            df = df_ if df is None else df.join(df_)

            # [sfan] Select features
            features = config_json['DATA']['features']
            columns = [f"{table}_{feature}" for feature in features]
            df = df[columns]

        # too quiet before 2015, time waste. copy() to avoid pandas errors
        # [sfan] start year is read from the config file
        # df = df.loc['2015':].copy()
        start_year = config_json['DATA']['start_year']
        df = df.loc[start_year:].copy()

        # [sfan] fill nan
        df = df.replace([np.inf, -np.inf], np.nan).ffill()  # .bfill()?
        # [sfan] Use scale or not?
        """
        df = pd.DataFrame(
            robust_scale(df.values, quantile_range=(.1, 100-.1)),
            columns=df.columns, index=df.index
        )
        """

        """
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        """

        # TODO drop null rows? (inner join?)
        # TODO arbitrage
        # TODO indicators

        """
        diff_cols = [
            f"{table}_{k}" for k in
            'open high low close volume_btc volume vwap'.split(' ')
            for table in filenames.keys()
        ]
        df[diff_cols] = df[diff_cols].pct_change()\
            .replace([np.inf, -np.inf], np.nan)\
            .ffill()  # .bfill()?
        df = df.iloc[1:]
        target = df[self.target]  # don't scale price changes; we use that in raw form later
        df = pd.DataFrame(
            robust_scale(df.values, quantile_range=(.1, 100-.1)),
            columns=df.columns, index=df.index
        )
        df[self.target] = target

        # [sfan] 'cash' and 'value' features are filled in every timestep with default value 0
        df['cash'], df['value'] = 0., 0.
        """

        self.df = df

    def offset(self, ep_start, step):
        return ep_start + step

    def has_more(self, ep_start, step):
        return self.offset(ep_start, step) + self.window < self.df.shape[0]
        # return (ep + 1) * self.ep_stride + self.window < self.df.shape[0]

    def get_data(self, ep_start, step):
        offset = self.offset(ep_start, step)
        X = self.df.iloc[offset:offset+self.window]
        y = self.df.iloc[offset+self.window]
        # [sfan] normalized by close price of the last timestep in the window
        base = X.iloc[-1]['coinbase_close']
        X = X / base
        y = y / base
        return X, y

