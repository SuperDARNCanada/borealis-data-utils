"""
A script to fix hdf5 files. Removes, adds, or edits file fields in a site file.
"""

import argparse
import sys
import os
import copy
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

    :returns: the usage message
    """

    usage_message = """ borealis_fixer.py [-h] filename fixed_data_dir

    This script will read a file at the filename provided and write a
    new file in the fixed_data_dir. This script needs to be modified
    internally for the changes that you wish to make to the file
    (line 74).
    """

    return usage_message

def main():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("filename", help="The filename with path to the file that you "
                                          "want to fix. This file will not be edited "
                                          "but the data will be written in another "
                                          "directory.")
    parser.add_argument("fixed_data_dir", help="The directory in which to place the fixed "
                                          "file.")
    args=parser.parse_args()
    filename = args.filename
    fixed_data_dir = args.fixed_data_dir

    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))

    out_file = fixed_data_dir + "/" + os.path.basename(filename)
    tmp_file = fixed_data_dir + "/" + os.path.basename(filename) + ".tmp"


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

    for group_name in sorted_keys:

        # APPLY CHANGE HERE
        # recs[group_name]['data_dimensions'][0] = 2
        recs[group_name]['blanked_samples'] = sorted(set(recs[group_name]['blanked_samples']))
        # recs[group_name]['xcfs'] = recs[group_name]['xcfs'] * -1

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

        # TODO(keith): improve call to subprocess.
        sp.call(cmd.split())
        os.remove(tmp_file)


if __name__ == "__main__":
    main()
