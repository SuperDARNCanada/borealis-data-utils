# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller, Remington Rohel

"""
This script is used to find gaps in Borealis data files.
"""

import argparse
import datetime
import glob
from multiprocessing import get_context
import os

import h5py
import pydarnio


def get_record_timestamps(filename):
    """
    Get the record timestamps from a file. These are what are used
    to determine the gaps.

    Parameters
    ----------
    filename: str
        Filename to retrieve timestamps from.
    """
    print("Getting timestamps from file : " + filename)
    if (
        filename.endswith("hdf5.site")
        or filename.endswith("hdf5")
        or filename.endswith("h5")
    ):
        with h5py.File(filename, "r") as f:
            recs = sorted(list(f.keys()))
            if "sqn_timestamps" in recs:
                sqn_timestamps = [
                    datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc)
                    for x in f["sqn_timestamps"][:, 0]
                ]
            else:
                sqn_timestamps = []
                for r in recs:
                    rec = f[r]
                    sqn_timestamps.append(
                        datetime.datetime.fromtimestamp(
                            rec["sqn_timestamps"][0], tz=datetime.timezone.utc
                        )
                    )
    else:
        recs = pydarnio.read_dmap(filename, mode="strict")
        sqn_timestamps = []
        for r in recs:
            tstamp = datetime.datetime(
                r["time.yr"],
                r["time.mo"],
                r["time.dy"],
                r["time.hr"],
                r["time.mt"],
                r["time.sc"],
                r["time.us"],
                tzinfo=datetime.timezone.utc,
            )
            sqn_timestamps.append(tstamp)

    return sqn_timestamps


def check_for_gaps(timestamp_list, gap_spacing):
    """
    Take in lists of record start times and find gaps between the record start times that are greater than gap spacing.

    Parameters
    ----------
    timestamp_list
        Sorted list of timestamps to check for gaps within, typically given for a single day.
    gap_spacing
        Minimum spacing allowed between records, given in seconds.

    Returns
    -------
    gaps
        List of gaps, where the gap is a tuple of the first timestamp and the following timestamp where the gap
        occurred, i.e. (timestamp1, timestamp2)
    """
    gaps = [
        (t0, t1)
        for t0, t1 in zip(timestamp_list[:-1], timestamp_list[1:])
        if (t1 - t0).total_seconds() > gap_spacing
    ]
    return gaps


def print_gaps(gaps, first_timestamp, last_timestamp, gap_spacing, print_filename, uptime=False):
    """
    Printer function for a dictionary of gaps. Prints csv
    table for easy integration into documents.

    Parameters
    ----------
    gaps
        list of gaps as (start_of_gap, end_of_gap) tuples
    first_timestamp
        datetime of first timestamp in period of gaps
    last_timestamp
        datetime of last timestamp in period of gaps
    gap_spacing
        Gap spacing used, in s.
    print_filename
        filename to print the gaps table to, in addition to the stdout.
    uptime
        gaps actually represent times when data was found.
    """

    strf_format = "%Y%m%d %H:%M:%S"

    with open(print_filename, "w") as f:
        print(
            "GAPS GREATER THAN {} s BETWEEN {} and {}:,".format(
                str(gap_spacing),
                first_timestamp.strftime(strf_format),
                last_timestamp.strftime(strf_format),
            ),
            file=f,
        )
        # new line required for table to generate
        print(" ", file=f)
        print("START TIME, END TIME, DURATION (min), CAUSE,", file=f)

        total_duration_min = 0.0
        for gap_start, gap_end in gaps:
            gap_duration = gap_end - gap_start
            duration = gap_duration.total_seconds()
            duration_min = round(duration / 60.0, 1)
            print(
                f"{gap_start.strftime(strf_format)} ,{gap_end.strftime(strf_format)} ,{duration_min:.1f},,",
                file=f,
            )
            total_duration_min += duration_min

        # end table, print new line
        print(" ", file=f)
        total_duration_hrs = round(total_duration_min / 60.0, 1)
        total_duration_days = round(total_duration_hrs / 24.0, 1)
        timeperiod = last_timestamp - first_timestamp
        if uptime:
            uptime_percentage = (
                (total_duration_min * 60) / timeperiod.total_seconds() * 100.0
            )
            downtime_percentage = 100.0 - uptime_percentage
            time_type = "UPTIME"
        else:
            downtime_percentage = (
                (total_duration_min * 60) / timeperiod.total_seconds() * 100.0
            )
            uptime_percentage = 100.0 - downtime_percentage
            time_type = "DOWNTIME"
        
        print(
            "TOTAL {} DURATION IN PERIOD from {} to {}".format(
                time_type,
                first_timestamp.strftime(strf_format),
                last_timestamp.strftime(strf_format),
            ),
            file=f,
        )
        print(f"{total_duration_min:.1f} minutes,", file=f)
        print(f"{total_duration_hrs:.1f} hours,", file=f)
        print(f"{total_duration_days:.1f} days,", file=f)
        print(f"{uptime_percentage:.1f}% uptime,", file=f)
        print(f"{downtime_percentage:.1f}% downtime,", file=f)

    # Print the results to screen
    print(" ")
    with open(print_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())


def borealis_gaps_parser():
    parser = argparse.ArgumentParser(
        description="""Pass in the directory with files that you want to check for borealis gaps. This script uses 
        multiprocessing to check for gaps in the hdf5 and/or dmap files of each day and gaps between the days.
        """
    )
    parser.add_argument(
        "data_dir",
        help="Path to the directory that holds any directory structure which within contains all "
        "files from the dates you wish to get downtimes.",
    )
    parser.add_argument("start_day", help="First day to check, given as YYYYMMDD.")
    parser.add_argument("end_day", help="Last day to check, given as YYYYMMDD.")
    parser.add_argument(
        "--suffix",
        default="rawacf*",
        help="Pattern for matching (globbing) file suffixes. Default is files with 'rawacf' in the "
        "name, i.e. ending in 'rawacf*'.",
    )
    parser.add_argument(
        "--gap_spacing",
        type=float,
        default=120.0,
        help="The gap spacing that you wish to check the file for, in seconds. Default 120s.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="The number of processes to use in the multiprocessing, default 4.",
    )
    parser.add_argument(
        "--uptime",
        action="store_true",
        help="If given, returns the times when data WAS found."
    )
    parser.add_argument(
        "--gaps_table_file",
        help="The pathname of the file to print the gaps table to, default is placed in "
        "$HOME/borealis_gaps/",
    )
    return parser


if __name__ == "__main__":
    parser = borealis_gaps_parser()
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir[-1] != "/":
        data_dir += "/"

    lowest_dir = data_dir.split("/")[
        -2
    ]  # now has / at the end so must be second last element

    if args.gaps_table_file is None:
        print_filename = f"{os.environ['HOME']}/borealis_gaps/{args.start_day}_{args.end_day}_{lowest_dir}_gaps.csv"
    else:
        print_filename = args.gaps_table_file

    print_dir = os.path.dirname(print_filename)
    if not os.path.isdir(print_dir):
        raise OSError(f"Directory {print_dir} does not exist")

    start_day = datetime.datetime(
        year=int(args.start_day[0:4]),
        month=int(args.start_day[4:6]),
        day=int(args.start_day[6:8]),
        tzinfo=datetime.timezone.utc,
    )
    end_day = datetime.datetime(
        year=int(args.end_day[0:4]),
        month=int(args.end_day[4:6]),
        day=int(args.end_day[6:8]),
        tzinfo=datetime.timezone.utc,
    )

    all_days = [
        start_day + datetime.timedelta(n)
        for n in range(int((end_day - start_day).days) + 1)
    ]
    all_timestamps = []

    for one_day in all_days:
        # Get all the filenames and then all the timestamps for this day.
        date_str = one_day.strftime("%Y%m%d")
        print(f"{date_str}")
        
        files = sorted(
            glob.glob(f"{data_dir}**/{date_str}*{args.suffix}", recursive=True)
        )
        print(f"{len(files)} files found")
        daily_timestamps = list()

        with get_context("spawn").Pool(args.num_processes) as pool:
            for tstamps in pool.imap(get_record_timestamps, files):
                daily_timestamps.extend(tstamps)

        if one_day == start_day:
            daily_timestamps.insert(0, start_day)

        if one_day == end_day:
            last_timestamp = end_day + datetime.timedelta(
                seconds=59, minutes=59, hours=23
            )
            daily_timestamps.append(last_timestamp)

        all_timestamps.extend(sorted(daily_timestamps))

    gaps = check_for_gaps(all_timestamps, args.gap_spacing)
    if args.uptime:
        non_gaps = []
        for i in range(1, len(gaps)):
            non_gaps.append((gaps[i-1][1], gaps[i][0]))
        gaps = non_gaps

    print_gaps(
        gaps, all_timestamps[0], all_timestamps[-1], args.gap_spacing, print_filename, args.uptime
    )
