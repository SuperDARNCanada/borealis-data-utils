# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to plot all beams range-time data from a single
bfiq file.
"""

import argparse
import copy
import glob
import os
import sys

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
    return parser


if __name__ == '__main__':
    parser = plot_parser()
    args = parser.parse_args()

    filename = args.bfiq_file
    
    plot_arrays_range_time(filename) # plot all beams
