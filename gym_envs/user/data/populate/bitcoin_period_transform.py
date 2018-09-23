from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', help="Row data csv file path")
    parser.add_argument('-o', '--output', help="Output data csv file path")
    parser.add_argument('-p', '--period', help="Time period(minutes) of the data")

    args = parser.parse_args()

    source = args.source
    output = args.output
    period = args.period

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'populate', source))




if __name__ == '__main__':
    main()






