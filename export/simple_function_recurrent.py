
import os
import csv
import sys
import argparse

import stable_nalu

# Parse arguments
parser = argparse.ArgumentParser(description='Export results from simple function task')
parser.add_argument('--tensorboard-dir',
                    action='store',
                    type=str,
                    help='Specify the directory for which the data is stored')
parser.add_argument('--csv-out',
                    action='store',
                    type=str,
                    help='Specify the file for which the csv data is stored at')

args = parser.parse_args()


def matcher(tag):
    return (
        tag in ['metric/valid/interpolation', 'metric/test/extrapolation', 'metric/test/range_extrapolation']
    )

reader = stable_nalu.reader.TensorboardMetricReader(
    args.tensorboard_dir,
    metric_matcher=matcher,
    step_start=0,
    processes=None
)

with open(args.csv_out, 'w') as csv_fp:
    for index, df in enumerate(reader):
        df.to_csv(csv_fp, header=(index == 0), index=False)
        csv_fp.flush()
