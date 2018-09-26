from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', required=True, help="Row data csv file path")
    parser.add_argument('-o', '--output', required=True, help="Output data csv file path")
    parser.add_argument('-p', '--period', required=True, help="Time period(minutes) of the data:\n"
                                                              "1T: 1 minute\n"
                                                              "1H：1 hour\n;"
                                                              "1D: 1 day\n"
                                                              "1W: 1 week\n"
                                                              "1M: 1 month\n"
                                                              "1A: 1 year")

    args = parser.parse_args()

    source = args.source
    output = args.output
    period = args.period

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'populate', source))
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df.index = df['Timestamp'].tolist()

    df_out = pd.DataFrame(columns=df.columns)
    df_out['Open'] = df['Open'].resample(period).first()
    df_out['High'] = df['High'].resample(period).max()
    df_out['Low'] = df['Low'].resample(period).min()
    df_out['Close'] = df['Close'].resample(period).last()
    df_out['Volume_(BTC)'] = df['Volume_(BTC)'].resample(period).sum()
    df_out['Volume_(Currency)'] = df['Volume_(Currency)'].resample(period).sum()
    df_out['Weighted_Price'] = df['Weighted_Price'].resample(period).mean()
    df_out['Timestamp'] = df_out.index.view('int64') // 1e9

    df_out.to_csv(output, index=0)


if __name__ == '__main__':
    main()






