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

    usage_message = """ convert_and_plot_site_antennas_iq.py [-h] [--plot-only] [--convert-only] borealis_site_file_directory

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
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("borealis_site_directory", help="Path to the directory with files that "
                                                    "you wish to convert. "
                                                    "(e.g. 20190327.2210.38.sas.0.bfiq.hdf5.site)")
    parser.add_argument("--plot-only", help="Flag to only plot from array files in the directory.",
                        action="store_true")
    parser.add_argument("--convert-only", help="Flag to only convert (not plot) the site files in the directory.",
                        action="store_true")

    return parser

def main():
    parser = convert_and_plot_parser()
    args = parser.parse_args()
    
    if args.plot_only and args.convert_only:
        print(usage_msg())
        return(-1)

    if not args.plot_only:
        for filename in os.listdir(args.borealis_site_directory):
            if filename.endswith(".antennas_iq.site") or filename.endswith(".antennas_iq.site.bz2"):
                filepath = os.path.join(args.borealis_site_directory, filename)
                convert_cmd = "python3 ~/data_flow/site-linux/borealis_convert_file.py {}".format(filepath)
                execute_cmd(convert_cmd)
                #print(convert_cmd)        
    
    if not args.convert_only:
        for filename in os.listdir(args.borealis_site_directory):
            if filename.endswith(".antennas_iq.hdf5"):
                filepath = os.path.join(args.borealis_site_directory, filename)
                plot_cmd = "python3 ~/borealis-data-utils/SNR/plot_antennas_range_time.py {}".format(filepath)
                execute_cmd(plot_cmd)
                #print(plot_cmd)


if __name__ == "__main__":
    main()

