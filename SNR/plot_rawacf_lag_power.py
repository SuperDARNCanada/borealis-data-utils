# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to plot all beams, lag zero, main_acfs, intf_acfs,
and xcfs power. 
"""

import argparse
import copy
import glob
import os
import sys

from hdf5_rtplot_utils import plot_rawacf_lag_pwr

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    
    Returns
    ------- 
    usage_message: str
        The usage message on how to use this 
    """

    usage_message = """ plot_rawacf_lag_power.py [-h] rawacf_file --beams --lags --datasets --max-power --min-power
    
    Pass in the name of the rawacf array-restructured file that you want 
    to plot range time data from. All ranges power will be plotted 
    on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("rawacf_file", help="Name of the file to plot.")
    parser.add_argument("--beams", help="Beams to plot. Format as --beams=0,1,2-5", type=str)
    parser.add_argument("--lags", help="Lags to plot. Format as --lags=0,1-4,11", type=str, default='0')
    parser.add_argument("--datasets", help="Datasets to plot. Format as --datasets=main_acfs,intf_acfs,xcfs", type=str)
    parser.add_argument("--max-power", help="Maximum Power of color scale (dB).", default=50.0, type=float)
    parser.add_argument("--min-power", help="Minimum Power of color scale (dB).", default=10.0, type=float)
    return parser


if __name__ == '__main__':
    parser = plot_parser()
    args = parser.parse_args()

    filename = args.rawacf_file

    beam_nums = []
    if args.beams is not None:
        beams = args.beams.split(',')
        for beam in beams:
            # If they specified a range, then include all numbers in that range (including endpoints)
            if '-' in beam:
                small_beam, big_beam = beam.split('-')
                beam_nums.extend(range(int(small_beam), int(big_beam) + 1))
            else:
                beam_nums.append(int(beam))

    print(beam_nums)

    lag_nums = []
    if args.lags is not None:
        lags = args.lags.split(',')
        for lag in lags:
            # If they specified a range, then include all numbers in that range (including endpoints)
            if '-' in lag:
                small_lag, big_lag = lag.split('-')
                lag_nums.extend(range(int(small_lag), int(big_lag)+1))
            else:
                lag_nums.append(int(lag))

    print(lag_nums)

    if args.datasets is not None:
        datasets = args.datasets.split(',')
    else:
        datasets = None

    print(datasets)

    plot_rawacf_lag_pwr(filename, beam_nums=beam_nums, lag_nums=lag_nums, datasets=datasets, num_processes=3,
                        vmax=args.max_power, vmin=args.min_power)


