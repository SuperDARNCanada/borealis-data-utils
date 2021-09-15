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

def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns: the usage message
    """

    usage_message = """ convert_and_plot_site_antennas_iq.py [-h] [--plot-only] [--convert-only] [--convert-all-first] [--convert-then-plot] borealis_site_file_directory

    Pass in the directory filled with files you wish to convert (should end in '.hdf5.site' ('.bz2' optional)).
    The script will decompress if a bzipped hdf5 site file with 'bz2' extension is provided.

    The script will :
    1. convert the records to an array style file, writing the file as the borealis_site_file
       with the last extension (should be '.site') removed.
    2. convert the records to a dmap dictionary and then write to file as the given filename,
       with extensions '.[borealis_filetype].hdf5.site' replaced with [dmap_filetype].dmap.
       The script will also bzip the resulting dmap file. 
    3. plot the antennas_iq for each antenna in a png file, in the same directory."""

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

def convert_and_plot_parser():
    """
    Parses the command line arguments passed when this script was called.

    :returns:   Object with all the arguments and flags supported by this script.
    :rtype:     ArgumentParser object
    """
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("borealis_site_directory", help="Path to the directory with files that "
                                                    "you wish to convert. "
                                                    "(e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("--plot-only", help="Flag to only plot from array files in the directory.",
                        action="store_true")
    parser.add_argument("--convert-only", help="Flag to only convert (not plot) the site files in the directory.",
                        action="store_true")
    parser.add_argument("--convert-all-first", help="Flag to convert all files first, then plot all files after.",
                        action="store_true")
    parser.add_argument("--convert-then-plot", help="Flag to convert then plot each file, one by one.",
                        action="store_true")

    return parser

def convert_all_files(directory):
    """
    Loops through all .antennas_iq.hdf5.site(.bz2) files in directory and converts 
    to array-formatted files.

    :param      directory: The directory specified by the user
    :type       directory: string
    """
    print("Converting...")
    for filename in os.listdir(directory):
        if filename.endswith(".antennas_iq.hdf5.site") or filename.endswith(".antennas_iq.hdf5.site.bz2"):
            filepath = os.path.join(directory, filename)
            convert_cmd = "python3 ~/data_flow/site-linux/borealis_convert_file.py {}".format(filepath)
            print(filename)
            execute_cmd(convert_cmd)

def plot_all_files(directory):
    """
    Loops through all .antennas_iq.hdf5 files in directory and plots
    antennas_iq for each antenna in each file.

    :param      directory:  The directory specified by the user.
    :type       directory:  string
    """
    print("Plotting...")
    for filename in os.listdir(directory):
        if filename.endswith(".antennas_iq.hdf5"):
            filepath = os.path.join(directory, filename)
            plot_cmd = "python3 ~/borealis-data-utils/SNR/plot_antennas_range_time.py {}".format(filepath)
            print(filename)
            execute_cmd(plot_cmd)

def convert_all_first(directory):
    """
    First converts all .site files, then plots all the converted antennas_iq data.
    
    :param      directory:  The directory specified by the user
    :type       directory:  string
    """
    convert_all_files(directory)
    plot_all_files(directory)

def convert_then_plot_sequentially(directory):
    """
    For each antennas_iq .site file in directory, converts to array-formatted then plots
    antennas_iq for each antenna, then moves onto next file.

    :param      directory:  The directory specified by the user
    :type       directory:  string
    """
    print("Running")
    for filename in os.listdir(directory):
        if filename.endswith(".antennas_iq.hdf5.site") or filename.endswith(".antennas_iq.hdf5.site.bz2"):
            sitepath = os.path.join(directory, filename)
            basename = os.path.basename(filename)
            print(filename)
            convert_cmd = "python3 ~/data_flow/site-linux/borealis_convert_file.py {}".format(sitepath)
            execute_cmd(convert_cmd)    # Convert the site file to array-formatted

            arrayname = '.'.join(basename.split('.')[:-1])
            arraypath = os.path.join(directory, arrayname)
            plot_cmd = "python3 ~/borealis-data-utils/SNR/plot_antennas_range_time.py {}".format(arraypath)
            execute_cmd(plot_cmd)       # Plot the array-formatted file

def main():
    """
    Takes a path to a directory, then converts all antennas_iq.hdf5.site(.bz2) files into
    array-formatted files (antennas_iq.hdf5). Then, plots the antennas_iq for each antenna
    in each file. Optional flags allow either conversion only or plotting only.
    
    Converted files are saved in the same directory, and are named identically sans the .site(.bz2) suffix.

    Plots are saved in the same directory, with the same naming convention.
    """
    parser = convert_and_plot_parser()
    args = parser.parse_args()
      
    # User error 
    if args.plot_only and args.convert_only:
        print(usage_msg())
        return(-1)

    # User error
    if args.convert_all_first and args.convert_then_plot:
        print(usage_msg())
        return(-2)

    if args.plot_only:
        plot_all_files(args.borealis_site_directory)
    
    elif args.convert_only:
        convert_all_files(args.borealis_site_directory)

    elif args.convert_all_first:
        convert_all_first(args.borealis_site_directory)

    elif args.convert_then_plot:
        convert_then_plot_sequentially(args.borealis_site_directory)
    
    else:
        convert_then_plot_sequentially(args.borealis_site_directory)


if __name__ == "__main__":
    main()

