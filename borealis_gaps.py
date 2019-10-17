# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to find gaps in Borealis data files.

This script should be piped to a file gaps.md or similar.

This script generates text in a markdown table format. The following
command line call can be used to generate a docx table from 
the printed output:

pandoc -o output_file.docx -f markdown -t docx input_file.md

Future Work
-----------
Update to use pydarn BorealisRead for getting the records.

"""


import sys
import os
import math
import numpy as np
import datetime
import deepdish
import bz2
import glob
import time
from multiprocessing import Pool, Queue, Manager, Process
import argparse

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    
    Returns
    ------- 
    usage_message: str
        The usage message on how to use this 
    """

    usage_message = """ borealis_gaps.py [-h] data_dir start_day end_day
    
    Pass in the data directory that you want to check for borealis gaps. This script uses 
    multiprocessing to check for gaps in the files of each day and gaps between the days. 

    The data directory passed in should have within it multiple directories named YYYYMMDD
    for a day's worth of data that is held there.
    """

    return usage_message


def borealis_gaps_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("data_dir", help="Path to the directory that holds the day directories "
                                         "containing *[filetype].hdf5.site.bz2 files.")
    parser.add_argument("start_day", help="First day directory to check, given as YYYYMMDD.")
    parser.add_argument("end_day", help="Last day directory to check, given as YYYYMMDD.")
    parser.add_argument("--filetype", help="The filetype that you want to check gaps in (bfiq or "
                                           "rawacf typical). Default 'rawacf'")
    parser.add_argument("--gap_spacing", help="The gap spacing that you wish to check the file"
                                              " for, in seconds. Default 7s.")
    parser.add_argument("--num_processes", help="The number of processes to use in the multiprocessing,"
                                                " default 4.")
    parser.add_argument("--file_structure", help="The file structure to use when reading the files, "
                                                " default 'site' but can also be 'array'")
    return parser


def decompress_bz2(filename):
    """
    Decompresses a bz2 file and returns the decompressed filename.

    Parameters
    ----------
    filename
        bz2 filename to decompress

    Returns
    -------
    newfilepath
        filename of decompressed file now in the path
    """
    basename = os.path.basename(filename)
    newfilepath = os.path.dirname(filename) + '/' + '.'.join(basename.split('.')[0:-1]) # all but bz2

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)

    return newfilepath


def get_record_timestamps(filename, record_dict, file_structure='site'):
    """
    Get the record timestamps from a file. These are what are used 
    to determine the gaps. 

    Also decompresses if the file is a bzip2 file before the read.

    Parameters
    ----------
    filename
        Filename to retrieve timestamps from.
    record_dict
        record_dictionary to append the entry filename: timestamps list
    file_structure 
        File structure of file provided. Default site, but can also
        be array structured. Determines how to retrieve the timestamps
        of the records.
    """
    print('Getting timestamps from file : ' + filename)
    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        borealis_hdf5_file = decompress_bz2(filename)
        bzip2 = True
    else:
        borealis_hdf5_file = filename
        bzip2 = False

    if file_structure == 'site':
        records = sorted(deepdish.io.load(borealis_hdf5_file).keys())
    elif file_structure == 'array':
        # get first timestamp per record in sqn_timestamps array of num_records x num_sequences
        records = sorted(deepdish.io.load(borealis_hdf5_file)['sqn_timestamps'][:0])
    else:
        raise Exception('Invalid file structure provided')

    inner_dict1 = record_dict 
    inner_dict1[filename] = records
    record_dict = inner_dict1 # reassign

    if bzip2:
        if borealis_hdf5_file.split('.')[-1] in ['bz2', 'bzip2']:
            print('Warning: attempted remove of original bzip file {}'.format(borealis_hdf5_file))
        else:
            os.remove(borealis_hdf5_file)


def combine_timestamp_lists(record_dict):
    """
    Combine and sort all timestamps within the dictionary.

    Parameters
    ----------
    record_dict
        Dictionary of filename: list of timestamps within file. Typically
        contains all files from within a single day.

    Returns
    -------
    sorted_timestamps
        A list of sorted timestamps from all filenames in the dictionary.
    """

    all_timestamps = []
    for filename, timestamp_list in record_dict.items():
        all_timestamps.extend(timestamp_list)

    sorted_timestamps = sorted(all_timestamps)

    return sorted_timestamps


def check_for_gaps_between_records(timestamp_list, gap_spacing):
    """
    Take in lists of record start times and find gaps between the record
    start times that are greater than gap spacing. Also includes finding
    gaps between files (between the list of record timestamps).

    Parameters
    ----------
    timestamp_list
        List of timestamps to check for gaps within, typically given 
        for a single day.
    gap_spacing
        Minimum spacing allowed between records, given in seconds.

    Returns
    -------
    gaps_list
        List of gaps, where the gap is a tuple of the first timestamp and 
        the following timestamp where the gap occurred ie.
        (timestamp1, timestamp2)
    """

    gaps_list = []

    sorted_list = sorted(timestamp_list)
    for record_num, record in enumerate(sorted_list):
        if record_num == len(sorted_list) - 1:
            continue
        this_record = datetime.datetime.utcfromtimestamp(float(record)/1000)
        expected_next_record = this_record + datetime.timedelta(seconds=float(gap_spacing))
        if datetime.datetime.utcfromtimestamp(float(sorted_list[record_num + 1])/1000) > expected_next_record:
            # append the gap to the dictionary list where key = filename,
            # value = list of gaps. Gaps are lists of (gap_start, gap_end)
            gaps_list = gaps_list + [(record, sorted_list[record_num + 1])]

    return gaps_list


def check_for_gaps_between_days(timestamps_dict, gap_spacing, gaps_dict):
    """
    Check for gaps between timestamp lists, which are passed in for each day
    in the timestamps_dict.

    Parameters
    ----------
    timestamps_dict 
        provided as day: list of timestamps for records in that day
    gap_spacing
        Minimum spacing allowed between records, given in seconds.
    gaps_dict
        existing gaps_dict, to append to if there are any gaps between days.

    Returns
    -------
    gaps_dict
        Updated gaps_dict with any extra gaps found added in.
    """

    sorted_days = sorted(timestamps_dict.keys())
    previous_last_record = timestamps_dict[sorted_days[0]][-1]
    for day_num, day in enumerate(sorted_days):
        if day_num == 0:
            continue # skip first one
        sorted_timestamps = sorted(timestamps_dict[day])
        # last record integration start time in the first file.
        previous_end_time = datetime.datetime.utcfromtimestamp(float(previous_last_record)/1000) 
        first_record = sorted_timestamps[0]
        last_record = sorted_timestamps[-1]
        start_time = datetime.datetime.utcfromtimestamp(float(first_record)/1000)
        end_time = datetime.datetime.utcfromtimestamp(float(last_record)/1000)
        if start_time > previous_end_time + datetime.timedelta(seconds=float(gap_spacing)):
            # append gap to this day's list of gaps. Dict key is day, list is list of (gap_start, gap_end)
            if day not in gaps_dict.keys():
                gaps_dict[day] = []
                print('Adding to dict') 
            #print(new_dict[filename])
            gaps_dict[day] = [(previous_last_record, first_record)] + gaps_dict[day]
        previous_last_record = last_record
    return gaps_dict


def daterange(start_date, end_date):
    """
    Generator for days between start_date and end_date inclusive
    """
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def print_gaps(gaps_dict):
    """
    Printer function for a dictionary of gaps. Prints a markdown
    table for easy integration into documents.

    Parameters
    ----------
    gaps_dict
        Dictionary of day: list of gaps to be printed to stdout
    """

    strf_format = '%Y%m%d %H:%M:%S'

    # need to print a new line at start of file otherwise markdown 
    # table won't generate.
    print(' ')
    print('| START TIME | END TIME | DURATION (min) | CAUSE |')
    print('| --- | --- | --- | --- |')
    for day in sorted(gaps_dict.keys()):
        gaps = gaps_dict[day]
        if gaps:  # not empty
            for (gap_start, gap_end) in gaps:
                gap_start_time = datetime.datetime.utcfromtimestamp(float(gap_start)/1000)
                gap_end_time = datetime.datetime.utcfromtimestamp(float(gap_end)/1000)
                gap_duration = gap_end_time - gap_start_time
                duration = gap_duration.total_seconds()
                duration_min = round(duration/60.0, 1)
                print('| ' + gap_start_time.strftime(strf_format) + ' | '  +
                      gap_end_time.strftime(strf_format) + ' | ' + 
                      str(duration_min) + ' |   |')


if __name__ == '__main__':
    parser = borealis_gaps_parser()
    args = parser.parse_args()

    if args.gap_spacing is None:
        gap_spacing = 7 # s
    else:
        gap_spacing = float(args.gap_spacing)

    if args.filetype is None:
        filetype = 'rawacf'
    else:
        filetype = args.filetype

    if args.num_processes is None:
        num_processes = 4
    else:
        num_processes = args.num_processes

    if args.file_structure is None:
        file_structure = 'site'
    else:
        file_structure = args.file_structure

    files = []
    
    data_dir = args.data_dir
    if data_dir[-1] != '/':
        data_dir += '/'

    start_day = datetime.datetime(year=int(args.start_day[0:4]), month=int(args.start_day[4:6]), day=int(args.start_day[6:8]))
    end_day = datetime.datetime(year=int(args.end_day[0:4]), month=int(args.end_day[4:6]), day=int(args.end_day[6:8]))    

    # this dictionary will be day: list of sorted timestamps in the day
    timestamps_dict = {}

    # this dictionary will be day: {dict of filename: list of timestamps in file}
    record_dict = {}

    # this dictionary will be day: [list of gaps in day, or between previous running day and this day]
    gaps_dict = {}

    for one_day in daterange(start_day, end_day):
        # Get all the filenames and then all the timestamps for this day.
        print(one_day.strftime("%Y%m%d"))
        if os.path.isdir(data_dir + one_day.strftime("%Y%m%d")):
            files = glob.glob(data_dir + one_day.strftime("%Y%m%d") + '/*.' + filetype + '*')
        else:
            continue

        jobs = []
        files_left = True
        filename_index = 0

        manager1 = Manager()
        filename_dict = manager1.dict()

        while files_left:
            for procnum in range(num_processes):
                try:
                    filename = files[filename_index + procnum]
                except IndexError:
                    if filename_index + procnum == 0:
                        print('No files found to check!')
                        raise
                    files_left = False
                    break
                p = Process(target=get_record_timestamps, args=(filename, filename_dict, file_structure))
                #p = Process(target=check_for_gaps_in_file, args=(filename, gap_spacing, gaps_dict, file_duration_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            filename_index += num_processes

        record_dict[one_day] = filename_dict
        
        timestamps_dict[one_day] = combine_timestamp_lists(record_dict[one_day])
        gaps_dict[one_day] = check_for_gaps_between_records(timestamps_dict[one_day], gap_spacing)
    
    # now that gaps_dict is entirely filled with each day in the range, find gaps between days
    gaps_dict = check_for_gaps_between_days(timestamps_dict, gap_spacing, gaps_dict)

    sorted_days = sorted(timestamps_dict.keys())

    # first timestamp is first day's first timestamp
    first_timestamp = datetime.datetime.utcfromtimestamp(float(sorted(timestamps_dict[sorted_days[0]])[0])/1000)
    # last timestamp is last day's last timestamp
    last_timestamp = datetime.datetime.utcfromtimestamp(float(sorted(timestamps_dict[sorted_days[-1]])[-1])/1000)
    print('GAPS BETWEEN {} and {} found:'.format(first_timestamp.strftime('%Y%m%d %H:%M:%S'), last_timestamp.strftime('%Y%m%d %H:%M:%S')))
    print_gaps(gaps_dict)

