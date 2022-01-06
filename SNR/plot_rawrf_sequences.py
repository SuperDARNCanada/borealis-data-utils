# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This script is used to plot rawrf data from a single rawrf file.
"""

import argparse
from hdf5_rtplot_utils import plot_rawrf_data


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
    parser.add_argument("--num-processes", help="Number of processes to use for plotting.", default=3, type=int)
    return parser


if __name__ == '__main__':
    rawrf_parser = plot_parser()
    args = rawrf_parser.parse_args()

    filename = args.rawrf_file

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

    sequence_nums = []
    if args.sequences is not None:
        sequences = args.sequences.split(',')
        for sequence in sequences:
            # If they specified a range, then include all numbers in that range (including endpoints)
            if '-' in sequence:
                small_sequence, big_sequence = sequence.split('-')
                sequence_nums.extend(range(int(small_sequence), int(big_sequence) + 1))
            else:
                sequence_nums.append(int(sequence))

    plot_rawrf_data(filename, antenna_nums=antenna_nums, num_processes=args.num_processes, sequence_nums=sequence_nums,
                    plot_directory=args.plot_directory)
