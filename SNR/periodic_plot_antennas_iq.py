# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
Usage:

convert_and_plot_site_antennas_iq.py [-h] [--plot-only] [--convert-only] borealis_site_file

Pass in the directory with files you wish to convert (files should end
in '.hdf5.site' ('.bz2' optional)). The script will decompress if a
bzipped hdf5 site file with 'bz2' extension is in the directory.

The script will :
1. convert the records to an array style file, writing the file as
    the borealis_site_file with the last extension (should be '.site')
    removed.
2. convert the records to a dmap dictionary and then write to file as
    the given filename, with extensions '.[borealis_filetype].hdf5.site'
    replaced with [dmap_filetype].dmap. The script will also bzip the
    resulting dmap file.
3. plot the antennas_iq for each antenna in each file.
"""

import argparse
import os
import sys
import subprocess as sp
from datetime import datetime
import convert_and_plot_site_antennas_iq as capsaiq

def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns: the usage message
    """

    usage_message = """ periodic_plot_antennas_iq.py [-h] [--convert-only] [--convert-all-first] [--convert-then-plot] borealis_site_directory plot_directory

    Pass in the directory filled with files you wish to convert (should end in '.hdf5.site' ('.bz2' optional)),
    plus the directory you wish for plotted files to be in.
    The script will decompress if a bzipped hdf5 site file with 'bz2' extension is provided.

    The script will :
    1. Copy all antennas_iq files from borealis_site_directory to plot_directory/date
    2. Convert the records to an array style file, writing the file as the borealis_site_file
       with the last extension (should be '.site') removed. 
    3. Plot the antennas_iq for each antenna in a png file, in the same directory.
    4. Remove all hdf5 files from the plot_directory"""

    return usage_message

def execute_cmd(cmd):
    """
    Execute a shell command and return the output

    :param      cmd:  The command
    :type       cmd:  string

    :returns:   Decoded output of the command.
    :rtype:     string
    """
    output = sp.check_output(cmd, shell=True)
    return output.decode('utf-8')

def periodic_plot_parser():
    """
    Parses the command line arguments passed when this script was called.

    :returns:   Object with all the arguments and flags supported by this script.
    :rtype:     ArgumentParser object
    """
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("borealis_site_directory", help="Path to the directory with files that "
                                                    "you wish to convert. "
                                                    "(e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("plot_directory", help="Path to directory where you want files to be stored.")
    parser.add_argument("--plot-only", help="Flag to only plot from array files in the directory.",
                        action="store_true")
    parser.add_argument("--convert-only", help="Flag to only convert (not plot) the site files in the directory.",
                        action="store_true")
    parser.add_argument("--convert-all-first", help="Flag to convert all files first, then plot all files after.",
                        action="store_true")
    parser.add_argument("--convert-then-plot", help="Flag to convert then plot each file, one by one.",
                        action="store_true")

    return parser

def get_files(data_dir, plot_dir, date, hour):
    """
    Get all the antennas_iq.site files from data_dir and put them in plot_dir.
    """
    ls_cmd = "ls {data_dir}/{date}/{date}.{hour}*antennas*".format(data_dir=data_dir, date=date, hour=hour)
    ls_plot_cmd = "ls {}".format(plot_dir)
    print(ls_cmd)
    print(ls_plot_cmd)
    files = execute_cmd(ls_cmd)
    plot_dir_contents = execute_cmd(ls_plot_cmd)

    if date not in plot_dir_contents:
        mkdir_cmd = "mkdir {}/{}".format(plot_dir, date)
        print(mkdir_cmd)
        execute_cmd(mkdir_cmd)

    cp_cmd = "cp {data_dir}/{date}/{date}.{hour}*antennas* {plot_dir}".format(data_dir=data_dir,
                                        date=date, hour=hour, plot_dir=plot_dir)
    print(cp_cmd)
    execute_cmd(cp_cmd)

def remove_hdf5_files(plot_dir, date):
    """
    Removes all hdf5 files from plot_dir/date
    """
    rm_cmd = "rm {}/{}/*hdf5*".format(plot_dir, date)
    print(rm_cmd)
    execute_cmd(rm_cmd)

def main():
    """
    Takes a path to a directory, then converts all antennas_iq.hdf5.site(.bz2) files into
    array-formatted files (antennas_iq.hdf5). Then, plots the antennas_iq for each antenna
    in each file. Optional flags allow either conversion only or plotting only.
    
    Converted files are saved in the same directory, and are named identically sans the .site(.bz2) suffix.

    Plots are saved in the same directory, with the same naming convention.
    """
    parser = periodic_plot_parser()
    args = parser.parse_args()
    
    runtime = datetime.utcnow()
    date = runtime.strftime('%Y%m%d')
    hour = runtime.strftime('%H')

    # User error 
    if args.plot_only and args.convert_only:
        print(usage_msg())
        return(-1)

    # User error
    if args.convert_all_first and args.convert_then_plot:
        print(usage_msg())
        return(-2)

    get_files(args.borealis_site_directory, args.plot_directory, date, hour)
    
    if args.convert_only:
        capsaiq.convert_all_files("{}/{}".format(args.plot_directory, date))
    
    elif args.convert_all_first:
        capsaiq.convert_all_first("{}/{}".format(args.plot_directory, date))
    
    elif args.convert_then_plot:
        capsaiq.convert_then_plot_sequentially("{}/{}".format(args.plot_directory, date))
    
    else:
        capsaiq.convert_then_plot_sequentially("{}/{}".format(args.plot_directory, date))
    
    remove_hdf5_files(args.plot_directory, date)

if __name__ == "__main__":
    main()

