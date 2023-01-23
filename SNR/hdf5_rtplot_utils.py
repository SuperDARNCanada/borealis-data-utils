# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller
# Modified: Dec. 2, 2021 (Remington Rohel)
"""
This module contains plotting functions for range time plots 
from Borealis HDF5 files of the following types:
- antennas_iq
- bfiq
- rawacf

These functions can be used on site to plot Borealis HDF5 files
for testing.

Functions
---------
plot_unaveraged_range_time_data
    Used by the other functions. Takes the necessary arrays and plots 
    the data as a range-time and SNR plot. 
plot_antennas_range_time
    Uses plot_unaveraged_range_time_data and BorealisRead to read an
    antennas_iq file and plot the range-time data.
plot_arrays_range_time
    Uses plot_unaveraged_range_time_data and BorealisRead to read a
    bfiq file and plot the range-time data.

plot_averaged_range_time_data
    Used by other functions. Takes arrays of records x ranges and plots
    a range-time and an SNR plot.
plot_rawacf_lag_pwr 
    Uses plot_averaged_range_time_data and BorealisRead to read a
    rawacf file and plot the range-time data.
"""
import copy
import datetime
import matplotlib
# import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import deepdish as dd

from pydarnio import BorealisRead
from multiprocessing import Process

from minimal_restructuring import antennas_iq_site_to_array

matplotlib.use('Agg')


def plot_unaveraged_range_time_data(data_array, num_sequences_array, timestamps_array, dataset_descriptor,
                                    plot_filename, vmax, vmin, start_sample, end_sample, figsize):
    """
    Plots data as range time given an array with correct dimensions. Also
    plots SNR by finding the ratio of max power in the sequence to average 
    power of the 10 weakest range gates.

    Note that this plots unaveraged data. All sequences available from the 
    record will be plotted side by side.

    Parameters
    ----------
    data_array: ndarray
        Array with shape num_records x max_num_sequences x num_samps for some 
        dataset.
    num_sequences_array: ndarray
        Array with shape num_records with the number of sequences per record.
    timestamps_array: ndarray
        Array of timestamps with dimensions num_records x max_num_sequences.
    dataset_descriptor: str
        Name for dataset, to be included in plot title.
    plot_filename: str
        Where to save plot.
    vmax: float
        Max power for the color bar on the plot. 
    vmin: float
        Min power for the color bar on the plot. 
    start_sample: int
        The sample to start plotting at. 
    end_sample: int
        The last sample in the sequence to plot.
    figsize: tuple (float, float)
        The desired size (in inches) of the plotted figure.
    """
    print(dataset_descriptor)
    (num_records, max_num_sequences, num_samps) = data_array.shape

    power_list = []     # list of lists of power
    timestamps = []     # list of timestamps
    noise_list = []     # list of (average of ten weakest ranges in sample range)
    max_snr_list = []   # max power - sequence noise (ave of 10 weakest ranges)

    for record_num in range(num_records):
        num_sequences = int(num_sequences_array[record_num])

        # get data for all sequences up to num sequences for this record.
        voltage_samples = data_array[record_num, :num_sequences, :]

        for sequence in range(num_sequences):
            timestamp = float(timestamps_array[record_num, sequence])
            timestamps.append(timestamp)

            # Get the raw power from the voltage samples
            power = np.abs(voltage_samples)[sequence, start_sample:end_sample]
            power_db = 10 * np.log10(power)
            power_list.append(power_db)

            # Average the 10 lowest power samples for this sequence, and call this the noise level
            sequence_noise_db = 10 * np.log10(np.average(np.sort(power)[:10]))
            noise_list.append(sequence_noise_db)

            # Max SNR = maximum power - noise level
            max_snr_list.append(np.max(power_db[2:])-sequence_noise_db)

    power_array = np.array(power_list)
    
    start_time = datetime.datetime.utcfromtimestamp(timestamps[0])
    end_time = datetime.datetime.utcfromtimestamp(timestamps[-1])
    
    # take the transpose to get sequences x samps for the dataset
    new_power_array = np.transpose(power_array)

    kw = {'width_ratios': [95, 5], 'height_ratios': [1, 3]}
    fig, ((ax1, cax1), (ax2, cax2)) = plt.subplots(2, 2, figsize=figsize, gridspec_kw=kw)
    fig.suptitle(f'{dataset_descriptor} Raw Power Sequence Time {start_time.strftime("%Y%m%d")} '
                 f'{start_time.strftime("%H:%M:%S")} to {end_time.strftime("%H:%M:%S")} UT vs Range')

    # plot SNR and noise
    ax1.plot(range(len(max_snr_list)), max_snr_list)
    ax1.set_title('Max SNR in sequence')
    ax1.set_ylabel('SNR (dB)')

    img = ax2.imshow(new_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('plasma'), vmax=vmax, vmin=vmin)
    ax2.set_title(f'Range-time based on samples {start_sample} to {end_sample}')
    ax2.set_ylabel('Sample number (Range)')
    ax2.set_xlabel('Sequence number (spans time)')

    fig.colorbar(img, cax=cax2, label='Raw Power (dB)')
    cax1.axis('off') 

    ax2.get_shared_x_axes().join(ax1, ax2)
    print(plot_filename)
    plt.savefig(plot_filename)
    plt.close() 


def plot_antennas_range_time(antennas_iq_file, antenna_nums=None, num_processes=3, vmax=40.0, vmin=10.0, start_sample=0,
                             end_sample=70, plot_directory='', figsize=(12, 10)):
    """ 
    Plots unaveraged range time data from echoes received in every sequence
    for a single antenna.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses plasma color map.

    Parameters 
    ----------
    antennas_iq_file: str
        The filename that you are plotting data from for plot title.
    antenna_nums: list[int]
        List of antennas you want to plot. This is the antenna number 
        as listed in the antenna_arrays_order. The index into the data array
        is determined by finding the index of the antenna number into the 
        antenna_arrays_order list. The data array is organized main antennas 
        first consecutively, followed by interferometer antennas consecutively. 
        Default None, which allows the algorithm to plot all antennas available 
        in the dataset.
    num_processes: int
        The number of processes to use to plot the data. Default 3.
    vmax: float
        Max power for the color bar on the plot, in dB. Default 40 dB.
    vmin: float
        Min power for the color bar on the plot, in dB.  Default 10 dB.
    start_sample: int
        The sample to start plotting at. Default 0th range (first sample).
    end_sample: int
        The last sample in the sequence to plot. Default 70 so ranges 0-69
        will plot.
    plot_directory: str
        The directory that generated plots will be saved in. Default '', which
        will save plots in the same location as the input file.
    figsize: tuple (float, float)
        The size of the figure to create, in inches across by inches tall. Default (12, 10)
    """
    basename = os.path.basename(antennas_iq_file)

    if plot_directory == '':
        directory_name = os.path.dirname(antennas_iq_file)
    elif not os.path.exists(plot_directory):
        directory_name = os.path.dirname(antennas_iq_file)
        print(f"Plot directory {plot_directory} does not exist. Using directory {directory_name} instead.")
    else:
        directory_name = plot_directory

    time_of_plot = '.'.join(basename.split('.')[0:6])

    # Try to guess the correct file structure
    basename = os.path.basename(antennas_iq_file)
    is_site_file = 'site' in basename

    if is_site_file:
        arrays, antenna_names, antenna_indices = antennas_iq_site_to_array(antennas_iq_file, antenna_nums)
    else:
        reader = BorealisRead(antennas_iq_file, 'antennas_iq', 'array')
        arrays = reader.arrays

        (num_records, num_antennas, max_num_sequences, num_samps) = arrays['data'].shape

        # typically, antenna names and antenna indices are the same except
        # where certain antennas were skipped in data writing for any reason.
        if antenna_nums is None or len(antenna_nums) == 0:
            antenna_indices = list(range(0, num_antennas))
            antenna_names = list(arrays['antenna_arrays_order'])
        else:
            antenna_indices = []
            antenna_names = [f'antenna_{a}' for a in antenna_nums]
            for antenna_num in antenna_nums:
                antenna_indices.append(list(arrays['antenna_arrays_order']).index('antenna_' + str(antenna_num)))

    sequences_data = arrays['num_sequences']
    timestamps_data = arrays['sqn_timestamps']

    arg_tuples = []
    print(antennas_iq_file)

    if is_site_file:
        iterable = enumerate(antenna_names)
    else:
        iterable = zip(antenna_indices, antenna_names)

    plotted = False
    for antenna_num, antenna_name in iterable:
        antenna_data = arrays['data'][:, antenna_num, :, :]
        plot_filename = f'{directory_name}/{time_of_plot}.{antenna_name}_{start_sample}_{end_sample}.jpg'
        if num_processes == 1:
            # If the system is memory-limited, we can save memory by plotting in this thread
            plot_unaveraged_range_time_data(antenna_data, sequences_data, timestamps_data, antenna_name, plot_filename,
                                            vmax, vmin, start_sample, end_sample, figsize)
            plotted = True
        else:
            arg_tuples.append((copy.copy(antenna_data), sequences_data, timestamps_data, antenna_name, plot_filename,
                               vmax, vmin, start_sample, end_sample, figsize))

    if plotted:
        # Already plotted in this thread
        return

    jobs = []
    antennas_index = 0
    antennas_left = True
    while antennas_left:
        for procnum in range(num_processes):
            try:
                antenna_args = arg_tuples[antennas_index + procnum]
            except IndexError:
                if antennas_index + procnum == 0:
                    print('No antennas found to plot')
                    raise
                antennas_left = False
                break
            p = Process(target=plot_unaveraged_range_time_data, args=antenna_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        antennas_index += num_processes


def plot_arrays_range_time(bfiq_file, beam_nums=None, num_processes=3, vmax=50.0, vmin=10.0, start_sample=0,
                           end_sample=70, plot_directory='', figsize=(12, 10)):
    """ 
    Plots unaveraged range time data from echoes received in every sequence
    for a single beam.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses plasma color map.

    Parameters 
    ----------
    bfiq_file: str
        The filename that you are plotting data from for plot title.
    beam_nums: list[int]
        The list of beam numbers to plot. Default None which allows all beams 
        available in the file to be plotted.
    num_processes: int
        The number of processes to use to plot the data from the beam_nums. Default 3.
    vmax: float
        Max power for the color bar on the plot. Default 50 dB.
    vmin: float
        Min power for the color bar on the plot.  Default 10 dB. 
    start_sample: int
        The sample to start plotting at. Default 0th range (first sample).
    end_sample: int
        The last sample in the sequence to plot. Default 70 so ranges 0-69
        will plot.
    plot_directory: str
        The directory that generated plots will be saved in. Default '', which
        will save plots in the same location as the input file.
    figsize: tuple (float, float)
        The size of the figure to create, in inches across by inches tall. Default (12, 10)
    """
    # Try to guess the correct file structure
    if "site" in bfiq_file:
        reader = BorealisRead(bfiq_file, 'bfiq', 'site')
    else:
        reader = BorealisRead(bfiq_file, 'bfiq', 'array')
    arrays = reader.arrays

    (num_records, num_antenna_arrays, max_num_sequences, max_num_beams, num_samps) = arrays['data'].shape

    basename = os.path.basename(bfiq_file)

    if plot_directory == '':
        directory_name = os.path.dirname(bfiq_file)
    elif not os.path.exists(plot_directory):
        directory_name = os.path.dirname(bfiq_file)
        print(f"Plot directory {plot_directory} does not exist. Using directory {directory_name} instead.")
    else:
        directory_name = plot_directory

    time_of_plot = '.'.join(basename.split('.')[0:6])

    # find the number of unique beams and their azimuths
    if beam_nums is None or len(beam_nums) == 0:
        # arrays['beam_nums'] is of shape num_records x max_num_beams
        # Note we do not want to include appended zeroes in the unique calc.
        all_beams = np.empty(0, dtype=np.uint32)

        for record_num in range(0, num_records):
            num_beams_in_record = int(arrays['num_beams'][record_num])
            all_beams = np.concatenate((all_beams, arrays['beam_nums'][record_num, :num_beams_in_record]))

        beam_names = np.unique(all_beams)

    else:
        beam_names = np.array(beam_nums, dtype=np.uint32)

    arg_tuples = []
    print(bfiq_file)

    for beam_name in beam_names:
        # find only the data with this beam name
        # data will be num_records avail (with this beam and array)
        # to plot x num_sequences x num_samps
        # also have to remake the timestamps and sequences data only for 
        # those records that contain this beam.
        beam_timestamps_data = np.empty((0, max_num_sequences))
        beam_sequences_data = np.empty(0)
        beam_arrays_data = np.empty((0, num_antenna_arrays, max_num_sequences, num_samps))

        for record_num in range(0, num_records):
            if beam_name in arrays['beam_nums'][record_num]:
                beam_index = list(arrays['beam_nums'][record_num]).index(beam_name)
                beam_timestamps_data = np.concatenate((beam_timestamps_data,
                                                       np.reshape(arrays['sqn_timestamps'][record_num],
                                                                  (1, max_num_sequences))))
                beam_sequences_data = np.concatenate((beam_sequences_data,
                                                      np.reshape(arrays['num_sequences'][record_num], 1)))
                beam_arrays_data = np.concatenate((beam_arrays_data,
                                                   np.reshape(arrays['data'][record_num, :, :, beam_index, :],
                                                              (1, num_antenna_arrays, max_num_sequences, num_samps))))

        for array_num, array_name in enumerate(arrays['antenna_arrays_order']):
            plot_filename = f'{directory_name}/{time_of_plot}.{array_name}_beam{str(beam_name)}_{start_sample}_' \
                            f'{end_sample}.jpg'
            descriptor = f'{array_name} beam {str(beam_name)}'
            arg_tuples.append((copy.copy(beam_arrays_data[:, array_num, :, :]), beam_sequences_data,
                               beam_timestamps_data, descriptor, plot_filename, vmax, vmin, start_sample, end_sample,
                               figsize))

    jobs = []
    plots_index = 0
    plots_left = True
    while plots_left:
        for procnum in range(num_processes):
            try:
                plot_args = arg_tuples[plots_index + procnum]
            except IndexError:
                if plots_index + procnum == 0:
                    print('No data found to plot')
                    raise
                plots_left = False
                break
            p = Process(target=plot_unaveraged_range_time_data, args=plot_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        plots_index += num_processes


def plot_averaged_range_time_data(data_array, timestamps_array, dataset_descriptor, plot_filename, vmax, vmin, figsize):
    """
    Plots data as range time given an array with correct dimensions. Also
    plots SNR by finding the ratio of max power in the sequence to average 
    power of the 10 weakest range gates.

    Note that this plots averaged data. 

    Parameters
    ----------
    data_array: ndarray
        Array with shape num_records x num_samps (or num_ranges) for some 
        correlations dataset.
    timestamps_array: ndarray
        Array of timestamps with dimensions num_records x max_num_sequences.
    dataset_descriptor: str
        Name for dataset, to be included in plot title.
    plot_filename: str
        Where to save plot.
    vmax: float
        Max power for the color bar on the plot. 
    vmin: float
        Min power for the color bar on the plot.
    figsize: tuple (float, float)
        The desired size of the figure, in inches across by inches tall.
    """
    print(dataset_descriptor)
    (num_records, num_samps) = data_array.shape

    power_list = []     # list of lists of power
    timestamps = []     # list of timestamps
    noise_list = []     # list of (average of ten weakest ranges in sample range)
    max_snr_list = []   # max power - noise (ave of 10 weakest ranges)

    for record_num in range(num_records):
        # get all ranges for this record
        voltage_samples = data_array[record_num, :]

        # get first timestamp in this record.
        timestamp = float(timestamps_array[record_num, 0])
        timestamps.append(timestamp)

        # Raw power only, no averaging
        power = np.sqrt(abs(voltage_samples))
        power_db = 10 * np.log10(power)
        power_list.append(power_db)

        # set noise to average of 10 lowest ranges
        record_noise_db = 10 * np.log10(np.average(np.sort(power)[:10]))
        noise_list.append(record_noise_db)

        # Maximum Power - noise for the sequence
        max_snr_list.append(np.max(power_db[2:]) - record_noise_db)

    # want records x ranges
    new_power_array = np.transpose(np.array(power_list))
    
    start_time = datetime.datetime.utcfromtimestamp(timestamps[0])
    end_time = datetime.datetime.utcfromtimestamp(timestamps[-1])

    kw = {'width_ratios': [95, 5], 'height_ratios': [1, 3]}
    fig, ((ax1, cax1), (ax2, cax2)) = plt.subplots(2, 2, figsize=figsize, gridspec_kw=kw)
    fig.suptitle(f'{dataset_descriptor} PWR Time {start_time.strftime("%Y%m%d")} {start_time.strftime("%H:%M:%S")} to '
                 f'{end_time.strftime("%H:%M:%S")} UT vs Range')

    # plot SNR and noise (10 weakest ranges average)
    ax1.plot(range(len(max_snr_list)), max_snr_list)
    ax1.set_title('Max SNR in record')
    ax1.set_ylabel('SNR (dB)')

    img = ax2.imshow(new_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('plasma'), vmax=vmax, vmin=vmin)
    ax2.set_title('Range-time plot')
    ax2.set_ylabel('Range gate number')
    ax2.set_xlabel('Record number (spans time)')

    fig.colorbar(img, cax=cax2, label='Raw Power (dB)')
    cax1.axis('off') 

    ax2.get_shared_x_axes().join(ax1, ax2)
    print(plot_filename)
    plt.savefig(plot_filename)
    plt.close() 


def plot_rawacf_lag_pwr(rawacf_file, beam_nums=None, lag_nums=None, datasets=None, num_processes=3, vmax=50.0,
                        vmin=10.0, plot_directory='', figsize=(12, 10)):
    """
    Plots the lag xcf phase of rawacf.hdf5 file, lag number found via lag_index.
    
    Plots averaged range time data from echoes received for given beams
    and given lags (default all available beams lag0). Can plot all beams 
    and all lags. 

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses plasma color map.

    Parameters 
    ----------
    rawacf_file: str
        The filename that you are plotting data from for plot title.
    beam_nums: list[int]
        The list of beam numbers to plot. Default None which allows all beams 
        available in the file to be plotted.
    lag_nums: list[int]
        The list of lag numbers to plot. Default [0] to plot only lag0pwr.
        lag_nums=None allows all lags available in the file to be plotted.
    datasets: list[str]
        The datasets to plot. Default all [acfs, intf_acfs, and xcfs].
    num_processes: int
        The number of processes to use to plot the data from the beam_nums
        and lag_nums. Default 3.
    vmax: float
        Max power for the color bar on the plot, in dB. Default 50 dB.
    vmin: float
        Min power for the color bar on the plot, in dB.  Default 10 dB.
    plot_directory: str
        The directory that generated plots will be saved in. Default '', which
        will save plots in the same location as the input file.
    figsize: tuple (float, float)
        The size of the figure to create, in inches across by inches tall. Default (12, 10)
    """
    # Try to guess the correct file structure
    basename = os.path.basename(rawacf_file)
    is_site_file = 'site' in basename

    if is_site_file:
        reader = BorealisRead(rawacf_file, 'rawacf', 'site')
    else:
        reader = BorealisRead(rawacf_file, 'rawacf', 'array')
    arrays = reader.arrays

    (num_records, max_num_beams, num_ranges, num_lags) = arrays['main_acfs'].shape
    max_num_sequences = arrays['sqn_timestamps'].shape[1]

    basename = os.path.basename(rawacf_file)

    if plot_directory == '':
        directory_name = os.path.dirname(rawacf_file)
    elif not os.path.exists(plot_directory):
        directory_name = os.path.dirname(rawacf_file)
        print(f"Plot directory {plot_directory} does not exist. Using directory {directory_name} instead.")
    else:
        directory_name = plot_directory

    time_of_plot = '.'.join(basename.split('.')[0:6])

    # find the number of unique beams and their azimuths
    if beam_nums is None or len(beam_nums) == 0:
        # arrays['beam_nums'] is of shape num_records x max_num_beams
        # Note we do not want to include appended zeroes in the unique calc.
        all_beams = np.empty(0, dtype=np.uint32)
        for record_num in range(0, num_records):
            num_beams_in_record = int(arrays['num_beams'][record_num])
            all_beams = np.concatenate((all_beams, arrays['beam_nums'][record_num, :num_beams_in_record]))
        beam_names = np.unique(all_beams)
    else:
        beam_names = np.array(beam_nums, dtype=np.uint32)

    all_lag_nums = [lag[1] - lag[0] for lag in list(arrays['lags'])]

    # No lag nums specified, try plotting all of them.
    if lag_nums is None or len(lag_nums) == 0:
        lag_nums = all_lag_nums
        lag_indices = list(range(len(lag_nums)))
    else:
        lag_indices = []
        for lag_num in lag_nums:
            try:
                lag_indices.append(all_lag_nums.index(lag_num))
            except ValueError:
                raise ValueError(f'Lag number {lag_num} is not found in the file')

    # No datasets specified, try to get all of them
    if datasets is None or len(datasets) == 0:
        datasets = ['main_acfs', 'intf_acfs', 'xcfs']

    for dataset in datasets:
        if dataset not in ['main_acfs', 'intf_acfs', 'xcfs']:
            raise ValueError(f'Dataset {dataset} not available in rawacf file {rawacf_file}')

    arg_tuples = []
    print(rawacf_file)

    for beam_name in beam_names:
        # find only the data with this beam name
        # data will be num_records avail (with this beam and lag number)
        # to plot x num_ranges
        beam_timestamps_data = np.empty((0, max_num_sequences))
        for lag_index, lag_num in zip(lag_indices, lag_nums):
            beam_lag_dict = {}
            for dataset in datasets:
                beam_lag_dict[dataset] = np.empty((0, num_ranges))
            for record_num in range(0, num_records):
                if beam_name in arrays['beam_nums'][record_num]:
                    beam_index = list(arrays['beam_nums'][record_num]).index(beam_name)
                    beam_timestamps_data = np.concatenate((beam_timestamps_data,
                                                           np.reshape(arrays['sqn_timestamps'][record_num],
                                                                      (1, max_num_sequences))))
                    for dataset in datasets:
                        beam_lag_dict[dataset] = np.concatenate(
                            (beam_lag_dict[dataset], np.reshape(arrays[dataset][record_num, beam_index, :, lag_index],
                                                                (1, num_ranges))))

            for dataset in datasets:
                plot_filename = f'{directory_name}/{time_of_plot}.{dataset}_beam{beam_name}_lag{lag_num}.jpg'
                descriptor = f'{dataset} beam {str(int(beam_name))} lag {str(lag_num)}'
                arg_tuples.append((copy.copy(beam_lag_dict[dataset]), beam_timestamps_data, descriptor, plot_filename,
                                   vmax, vmin, figsize))

    jobs = []
    plots_index = 0
    plots_left = True
    while plots_left:
        for procnum in range(num_processes):
            try:
                plot_args = arg_tuples[plots_index + procnum]
            except IndexError:
                if plots_index + procnum == 0:
                    print('No data found to plot')
                    raise
                plots_left = False
                break
            p = Process(target=plot_averaged_range_time_data, args=plot_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        plots_index += num_processes


def plot_rawrf_data(rawrf_file, antenna_nums=None, num_processes=3, sequence_nums=None, plot_directory='',
                    figsize=(12, 10)):
    """
    Plots a sequence of samples from a rawrf file.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses plasma color map.

    Parameters
    ----------
    rawrf_file: str
        The filename that you are plotting data from for plot title.
    antenna_nums: list[int]
        List of antennas you want to plot. Rawrf files do not store
        antennas_array_order, so this is simply indexed from 0. If your
        radar has a different antenna order, this will fail silently, but
        still naively plot the antennas just with the wrong labels. Default None, which plots all antennas.
    num_processes: int
        The number of processes to use to plot the data. Default 3.
    sequence_nums: list[int]
        The index of sequences in the first record to plot. Default None, which plots all sequences.
    plot_directory: str
        The directory that generated plots will be saved in. Default '', which
        will save plots in the same location as the input file.
    figsize: tuple (float, float)
        The size of the figure to create, in inches across by inches tall. Default (12, 10)
    """
    basename = os.path.basename(rawrf_file)

    if plot_directory == '':
        directory_name = os.path.dirname(rawrf_file)
    elif not os.path.exists(plot_directory):
        directory_name = os.path.dirname(rawrf_file)
        print(f"Plot directory {plot_directory} does not exist. Using directory {directory_name} instead.")
    else:
        directory_name = plot_directory

    time_of_plot = '.'.join(basename.split('.')[0:6])

    records = dd.io.load(rawrf_file)
    record_keys = sorted(list(records.keys()))

    record = records[record_keys[0]]

    # This little hack was made to deal with rawrf files afflicted by Issue #258 on the Borealis GitHub, which
    # has since been solved. It should work for all rawrf files regardless.
    num_sequences, num_antennas, num_samps = record['data_dimensions']
    total_samples = record['data'].size
    sequences_stored = int(total_samples / num_samps / num_antennas)
    data = record['data'].reshape((sequences_stored, num_antennas, num_samps))

    # typically, antenna names and antenna indices are the same except
    # where certain antennas were skipped in data writing for any reason.
    if antenna_nums is None or len(antenna_nums) == 0:
        antenna_indices = list(range(0, num_antennas))
    else:
        antenna_indices = antenna_nums
    antenna_names = [f'antenna_{a}' for a in antenna_nums]

    if sequence_nums is None:
        sequence_indices = [i for i in range(num_sequences)]
    else:
        sequence_indices = [num for num in sequence_nums if 0 < num < num_sequences]
        invalid_indices = [num for num in sequence_nums if num not in sequence_indices]
        if len(invalid_indices) != 0:
            print(f'Warning: sequence numbers {invalid_indices} not in file {rawrf_file}.')

        if len(sequence_indices) == 0:  # None of the input sequences were valid
            print(f'No requested sequences from {sequence_nums} found in file {rawrf_file}.')
            return

    timestamps_data = record['sqn_timestamps']
    sampling_rate = record['rx_sample_rate']

    arg_tuples = []
    print(rawrf_file)

    for antenna_num, antenna_name in zip(antenna_indices, antenna_names):
        antenna_data = data[sequence_indices, antenna_num, :]
        plot_filename_prefix = f'{directory_name}/{time_of_plot}.{antenna_name}_rawrf'
        arg_tuples.append((copy.copy(antenna_data), timestamps_data, antenna_name, plot_filename_prefix, sampling_rate,
                           figsize))

    jobs = []
    antennas_index = 0
    antennas_left = True
    while antennas_left:
        for procnum in range(num_processes):
            try:
                antenna_args = arg_tuples[antennas_index + procnum]
            except IndexError:
                if antennas_index + procnum == 0:
                    print('No antennas found to plot')
                    raise
                antennas_left = False
                break
            p = Process(target=plot_iq_data, args=antenna_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        antennas_index += num_processes


def plot_iq_data(voltage_samples, timestamps_array, dataset_descriptor, plot_filename_prefix, sample_rate, figsize):
    """
    Plots data as range time given an array with correct dimensions. Also
    plots SNR by finding the ratio of max power in the sequence to average
    power of the 10 weakest range gates.

    Note that this plots unaveraged data. All sequences available from the
    record will be plotted side by side.

    Parameters
    ----------
    voltage_samples: ndarray
        Array with shape num_sequences x num_samps for some
        dataset.
    timestamps_array: ndarray
        Array of timestamps with dimensions num_sequences.
    dataset_descriptor: str
        Name for dataset, to be included in plot title.
    plot_filename_prefix: str
        Path and beginning of filename to save plot. Since several plots are generated, there will be
        multiple plots saved, all sharing this same prefix.
    sample_rate: float
        Sampling rate of the data in the file. Hz
    figsize: tuple (float, float)
        Size of the plotted figure, in inches.
    """
    print(dataset_descriptor)

    start_time = datetime.datetime.utcfromtimestamp(timestamps_array[0])
    end_time = datetime.datetime.utcfromtimestamp(timestamps_array[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle(f'{dataset_descriptor} Raw Voltage Sequence Time {start_time.strftime("%Y%m%d")} '
                 f'{start_time.strftime("%H:%M:%S")} to {end_time.strftime("%H:%M:%S")} UT')

    # plot one sequence (real and imaginary))
    ax1.plot(np.arange(0, len(voltage_samples[0, :]))/sample_rate*1e6, np.real(voltage_samples[0, :]),
             label='In-Phase')
    ax1.plot(np.arange(0, len(voltage_samples[0, :]))/sample_rate*1e6, np.imag(voltage_samples[0, :]),
             label='Quadrature')
    ax1.legend()
    ax1.set_ylabel('Arbitrary Units')

    ax2.plot(np.arange(0, len(voltage_samples[0, :]))/sample_rate*1e6, 10 * np.log10(np.abs(voltage_samples[0, :])))
    ax2.set_ylabel('Power (dB)')
    ax2.set_xlabel('Time (us)')
    ax2.get_shared_x_axes().join(ax1, ax2)

    plot_name = plot_filename_prefix + '_time.jpg'
    print(plot_name)
    plt.savefig(plot_name)
    plt.close()

    # Take FFT
    N = voltage_samples[0, :].size
    fft_data = np.fft.fft(voltage_samples[0, :])
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(voltage_samples.size, d=1/sample_rate)) / 1e6
    zero_index = int(np.where(np.isclose(fft_freqs, 0.0))[0])

    fft_fixed = np.zeros(N, dtype=np.complex64)
    fft_fixed[:zero_index] = fft_data[zero_index:]
    fft_fixed[zero_index:] = fft_data[:zero_index]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(f'{dataset_descriptor} Sequence FFT Time {start_time.strftime("%Y%m%d")} '
                 f'{start_time.strftime("%H:%M:%S")} to {end_time.strftime("%H:%M:%S")} UT')

    # plot one sequence
    ax.plot(fft_freqs, 10 * np.log10(fft_fixed))
    ax.set_ylabel('Power (dB)')
    ax.set_xlabel('Frequency (MHz)')

    plot_name = plot_filename_prefix + '_freq.jpg'
    print(plot_name)
    plt.savefig(plot_name)
    plt.close()
