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

    usage_message = """ plot_antennas_range_time.py [-h] antennas_iq_file
    
    Pass in the name of the antennas_iq file that you want to plot range time data from. 
    Antenna numbers will be plotted on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("antennas_iq_file", help="Name of the file to plot.")
    parser.add_argument("--antennas", help="Antenna indices to plot. Format as --antennas=0,2-4,8")
    parser.add_argument("--max-power", help="Maximum Power of color scale (dB).", default=40.0, type=float)
    parser.add_argument("--min-power", help="Minimum Power of color scale (dB).", default=10.0, type=float)
    parser.add_argument("--start-sample", help="Sample Number to start at.", default=0, type=int)
    parser.add_argument("--end-sample", help="Sample Number to end at.", default=70, type=int)
    parser.add_argument("--plot-directory", help="Directory to save plots.", default='', type=str)
    return parser


if __name__ == '__main__':
    antennas_iq_parser = plot_parser()
    args = antennas_iq_parser.parse_args()

    filename = args.antennas_iq_file

    antenna_nums = []
    if args.antennas is not None:
        antennas = args.antennas.split(',')
        for antenna in antennas:
            # If they specified a range, then include all numbers in that range (including endpoints)
            if '-' in antenna:
                small_antenna, big_antenna = antenna.split('-')
                antenna_nums.extend(range(int(small_antenna), int(big_antenna) + 1))
            else:
                antenna_nums.append(int(antenna))

    plot_antennas_range_time(filename, antenna_nums=antenna_nums, num_processes=3, vmax=args.max_power,
                             vmin=args.min_power, start_sample=args.start_sample, end_sample=args.end_sample,
                             plot_directory=args.plot_directory)
