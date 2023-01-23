# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified: Dec. 2, 2021 (Remington Rohel)
"""
This script is used to plot all antennas range-time data from a single
antennas_iq file.
"""

import argparse
from hdf5_rtplot_utils import plot_antennas_range_time
import utils


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
    parser.add_argument("--figsize", help="Figure dimensions in inches. Format as --figsize=10,6", type=str)
    parser.add_argument("--num-processes", help="Number of processes to use for plotting.", default=3, type=int)
    return parser


if __name__ == '__main__':
    antennas_iq_parser = plot_parser()
    args = antennas_iq_parser.parse_args()

    filename = args.antennas_iq_file

    antenna_nums = []
    if args.antennas is not None:
        antenna_nums = utils.build_list_from_input(args.antennas)

    sizes = (32, 16)    # Default figsize
    if args.figsize is not None:
        sizes = []
        for size in args.figsize.split(','):
            sizes.append(float(size))
        if len(sizes) == 1:
            sizes.append(sizes[0])  # If they only pass in one size, assume they want a square plot.
        else:
            if len(sizes) > 2:
                print(f'Warning: only keeping {sizes[:2]} from input figure size.')
            sizes = sizes[:2]

    plot_antennas_range_time(filename, antenna_nums=antenna_nums, num_processes=args.num_processes, vmax=args.max_power,
                             vmin=args.min_power, start_sample=args.start_sample, end_sample=args.end_sample,
                             plot_directory=args.plot_directory, figsize=sizes)
