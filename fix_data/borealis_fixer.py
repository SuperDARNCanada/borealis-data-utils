# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import argparse
import bz2
import sys
import os
import copy
import itertools
import subprocess as sp
import numpy as np
import warnings
import tables
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
import deepdish as dd

def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ borealis_fixer.py [-h] filename fixed_dat_dir

    **** NOT TO BE USED IN PRODUCTION ****
    **** USE WITH CAUTION ****

    Modify a borealis file with updated data fields. Modify the script where
    indicated to update the file. Used in commissioning phase of Borealis when
    data fields were not finalized."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("filename", help="Path to the file that you wish to modify")
    parser.add_argument("fixed_data_dir", help="Path to place the updated file in.")

    return parser


def update_file(filename, out_file):

    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))


    tmp_file = out_file + ".tmp"


    write_dict = {}

    def convert_to_numpy(data):
        """Converts lists stored in dict into numpy array. Recursive.

        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        return data

    for key_num, group_name in enumerate(sorted_keys):

        # APPLY CHANGE HERE
        recs[group_name]['blanked_samples'] = sorted(set(recs[group_name]['blanked_samples']))

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

        sp.call(cmd.split())
        os.remove(tmp_file)


def decompress_bz2(filename):
    basename = os.path.basename(filename)
    newfilepath = os.path.dirname(filename) + '/' + '.'.join(basename.split('.')[0:-1]) # all but bz2

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)

    return newfilepath


def compress_bz2(filename):
    bz2_filename = filename + '.bz2'

    with open(filename, 'rb') as file, bz2.BZ2File(bz2_filename, 'wb') as bz2_file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            bz2_file.write(data)

    return bz2_filename


def file_updater(filename, fixed_data_dir):
    """
    Checks if the file is bz2, decompresses if necessary, and
    writes to a fixed data directory. If the file was bz2, then the resulting
    file will also be compressed to bz2.

    Parameters
    ----------
    filename
        filename to update, can be bz2 compressed
    fixed_data_dir
        pathname to put the new file into
    """

    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        hdf5_file = decompress_bz2(filename)
        bzip2 = True
    else:
        hdf5_file = filename
        bzip2 = False

    if fixed_data_dir[-1] == '/':
        out_file = fixed_data_dir + os.path.basename(hdf5_file)
    else:
        out_file = fixed_data_dir + "/" + os.path.basename(hdf5_file)

    update_file(hdf5_file, out_file)

    if bzip2:
        # remove the input file from the directory because it was generated.
        os.remove(hdf5_file)
        # compress the updated file to bz2 if the input file was given as bz2.
        bz2_filename = compress_bz2(out_file)
        os.remove(out_file)
        out_file = bz2_filename

    return out_file


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    file_updater(args.filename, args.fixed_data_dir)
