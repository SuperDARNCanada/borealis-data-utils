# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to 
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

    usage_message = """ plot_arrays_range_time.py [-h] bfiq_file
    
    Pass in the name of the bfiq array-restructured file that you want 
    to plot range time data from. All beams and arrays will be plotted 
    on separate plots.
    """

    return usage_message


def plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("bfiq_file", help="Name of the file to plot.")
    return parser

#colour_ranges = { iq_file : {'vmax' : 110.0, 'vmin' : 30.0} for iq_file in iq_files} # 

if __name__ == '__main__':
    parser = plot_parser()
    args = parser.parse_args()

    filename = args.bfiq_file
    
    plot_arrays_range_time(filename) # plot all beams
