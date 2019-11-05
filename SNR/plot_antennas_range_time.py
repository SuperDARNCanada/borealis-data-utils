# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to plot all antennas range-time data from a single
antennas_iq file.
"""

import argparse
import copy
import glob
import os
import sys

from hdf5_rtplot_utils import plot_antennas_range_time
from pydarn import BorealisRead

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
    
    Pass in the name of the antennas_iq array-restructured file that you want 
    to plot range time data from. Antenna numbers from 0 to 20 will be plotted 
    on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("antennas_iq_file", help="Name of the file to plot.")
    return parser

if __name__ == '__main__':
    parser = plot_parser()
    args = parser.parse_args()

    filename = args.antennas_iq_file
    
    plot_antennas_range_time(filename) # plot all antennas

