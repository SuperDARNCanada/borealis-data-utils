# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified: Dec. 2, 2021 (Remington Rohel)

"""
This script is used to plot all beams range-time data from a single
bfiq file.
"""

import argparse
from hdf5_rtplot_utils import plot_arrays_range_time


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    
    Returns
    ------- 
    usage_message: str
        The usage message on how to use this 
    """

    usage_message = """ plot_bfiq_range_time.py [-h] bfiq_file
    
    Pass in the name of the bfiq array-restructured file that you want 
    to plot range time data from. All beams and arrays will be plotted 
    on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("bfiq_file", help="Name of the file to plot.")
    parser.add_argument("--beams", help="Beams to plot.")
    parser.add_argument("--max-power", help="Maximum Power of color scale (dB).", default=50.0)
    parser.add_argument("--min-power", help="Minimum Power of color scale (dB).", default=10.0)
    parser.add_argument("--start-sample", help="Sample Number to start at.", default=0)
    parser.add_argument("--end-sample", help="Sample Number to end at.", default=80)
    return parser


if __name__ == '__main__':
    args_parser = plot_parser()
    args = args_parser.parse_args()

    filename = args.bfiq_file
    
    plot_arrays_range_time(filename, beam_nums=args.beams, num_processes=3, vmax=args.max_power, vmin=args.min_power,
                           start_sample=args.start_sample, end_sample=args.end_sample)  # plot all beams
