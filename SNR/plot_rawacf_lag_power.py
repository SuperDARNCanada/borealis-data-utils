# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified: Dec. 3, 2021 (Remington Rohel)

"""
This script is used to plot all beams, lag zero, main_acfs, intf_acfs,
and xcfs power. 
"""

import argparse
from hdf5_rtplot_utils import plot_rawacf_lag_pwr
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

    usage_message = """ plot_rawacf_lag_power.py [-h] rawacf_file
    
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
    parser.add_argument("--plot-directory", help="Directory to save plots.", default='', type=str)
    parser.add_argument("--figsize", help="Figure dimensions in inches. Format as --figsize=10,6", type=str)
    parser.add_argument("--num-processes", help="Number of processes to use for plotting.", default=3, type=int)
    return parser


if __name__ == '__main__':
    rawacf_parser = plot_parser()
    args = rawacf_parser.parse_args()

    filename = args.rawacf_file

    beam_nums = []
    if args.beams is not None:
        beam_nums = utils.build_list_from_input(args.beams)

    lag_nums = []
    if args.lags is not None:
        lag_nums = utils.build_list_from_input(args.lags)

    if args.datasets is not None:
        datasets = args.datasets.split(',')
    else:
        datasets = None

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

    plot_rawacf_lag_pwr(filename, beam_nums=beam_nums, lag_nums=lag_nums, datasets=datasets,
                        num_processes=args.num_processes, vmax=args.max_power, vmin=args.min_power,
                        plot_directory=args.plot_directory, figsize=sizes)
