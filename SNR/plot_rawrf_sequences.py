# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This script is used to plot rawrf data from a single rawrf file.
"""

import argparse
from hdf5_rtplot_utils import plot_rawrf_data
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

    usage_message = """ plot_rawrf_sequences.py [-h] rawrf_file

    Pass in the name of the rawrf file that you want to plot data from. 
    Antenna numbers will be plotted on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("rawrf_file", help="Name of the file to plot.")
    parser.add_argument("--antennas", help="Antenna indices to plot. Format as --antennas=0,2-4,8")
    parser.add_argument("--sequences", help="Sequence indices to plot. Format as --sequences=0,2-4,8")
    parser.add_argument("--plot-directory", help="Directory to save plots.", default='', type=str)
    parser.add_argument("--figsize", help="Figure dimensions in inches. Format as --figsize=10,6", type=str)
    parser.add_argument("--num-processes", help="Number of processes to use for plotting.", default=3, type=int)
    return parser


if __name__ == '__main__':
    rawrf_parser = plot_parser()
    args = rawrf_parser.parse_args()

    filename = args.rawrf_file

    antenna_nums = []
    if args.antennas is not None:
        antenna_nums = utils.build_list_from_input(args.antennas)

    sequence_nums = []
    if args.sequences is not None:
        sequence_nums = utils.build_list_from_input(args.sequences)

    sizes = (32, 16)    # Default figsize
    if args.figsize is not None:
        sizes = []
        for size in args.figsize.split(','):
            sizes.append(float(size))
        if len(sizes) == 1:
            sizes.append(sizes[0])  # If they only pass in one size, assume they want a square plot.
        else:
            sizes = sizes[:2]  # If they pass in more than 2 sizes, truncate to just the first two.
            print(f'Warning: only keeping {sizes} from input figure size.')

    plot_rawrf_data(filename, antenna_nums=antenna_nums, num_processes=args.num_processes, sequence_nums=sequence_nums,
                    plot_directory=args.plot_directory, figsize=sizes)
