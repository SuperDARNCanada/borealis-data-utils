# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified: Dec. 2, 2021 (Remington Rohel)
"""
This script is used to plot all antennas range-time data from a single
antennas_iq file.
"""

import argparse
from hdf5_rtplot_utils import plot_antennas_range_time


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    
    Returns
    ------- 
    usage_message: str
        The usage message on how to use this 
    """

    usage_message = """ plot_antennas_range_time.py [-h] antennas_iq_file --antenna-nums --max-power --min-power --start-sample --end-sample
    
    Pass in the name of the antennas_iq array-restructured file that you want 
    to plot range time data from. Antenna numbers from 0 to 20 will be plotted 
    on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("antennas_iq_file", help="Name of the file to plot.")
    parser.add_argument("--antenna-nums", help="Antenna indices to plot.")
    parser.add_argument("--max-power", help="Maximum Power of color scale (dB).", default=40.0)
    parser.add_argument("--min-power", help="Minimum Power of color scale (dB).", default=10.0)
    parser.add_argument("--start-sample", help="Sample Number to start at.", default=0)
    parser.add_argument("--end-sample", help="Sample Number to end at.", default=80)
    return parser


if __name__ == '__main__':
    args_parser = plot_parser()
    args = args_parser.parse_args()

    filename = args.antennas_iq_file
    
    plot_antennas_range_time(filename, antenna_nums=args.antenna_nums, num_processes=3, vmax=args.max_power,
                             vmin=args.min_power, start_sample=args.start_sample, end_sample=args.end_sample)

