# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

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
    array-structured antennas_iq file and plot the range-time data.
plot_arrays_range_time
    Uses plot_unaveraged_range_time_data and BorealisRead to read an 
    array-structured bfiq file and plot the range-time data.

plot_averaged_range_time_data
    Used by other functions. Takes arrays of records x ranges and plots
    a range-time and an SNR plot.
plot_rawacf_lag_pwr 
    Uses plot_averaged_range_time_data and BorealisRead to read an 
    array-structured rawacf file and plot the range-time data.
"""
import copy
import datetime
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from pydarnio import BorealisRead, BorealisWrite
from multiprocessing import Pool, Process
from scipy.fftpack import fft

matplotlib.use('Agg')
plt.rcParams.update({'font.size': 28})


def plot_unaveraged_range_time_data(data_array, num_sequences_array, timestamps_array, dataset_descriptor,
                                    plot_filename, vmax, vmin, start_sample, end_sample):
    """
    Plots data as range time given an array with correct dimensions. Also
    plots SNR by finding the ratio of max power in the sequence to average 
    power of the 10 weakest range gates.

    Note that this plots unaveraged data. All sequences available from the 
    record will be plotted side by side.

    Parameters
    ----------
    data_array
        Array with shape num_records x max_num_sequences x num_samps for some 
        dataset.
    num_sequences_array
        Array with shape num_records with the number of sequences per record.
    timestamps_array
        Array of timestamps with dimensions num_records x max_num_sequences.
    dataset_descriptor
        Name for dataset, to be included in plot title.
    plot_filename
        Where to save plot.
    vmax
        Max power for the color bar on the plot. 
    vmin
        Min power for the color bar on the plot. 
    start_sample
        The sample to start plotting at. 
    end_sample
        The last sample in the sequence to plot. 
    """
    print(dataset_descriptor)
    (num_records, max_num_sequences, num_samps) = data_array.shape

    power_list = [] # list of lists of power
    timestamps = [] # list of timestamps
    noise_list = [] # list of (average of ten weakest ranges in sample range)
    max_snr_list = [] # max power - sequence noise (ave of 10 weakest ranges)
    for record_num in range(num_records):
        num_sequences = int(num_sequences_array[record_num])
        # get all antennas, up to num sequences, all samples for this record.
        voltage_samples = data_array[record_num,:num_sequences,:]

        for sequence in range(num_sequences):
            timestamp = float(timestamps_array[record_num, 
                                                       sequence])
            # power only. no averaging done. 
            power = np.sqrt(voltage_samples.real**2 + 
                            voltage_samples.imag**2)[sequence,
                            start_sample:end_sample]
            power_db = 10 * np.log10(power)
            sequence_noise_db = 10 * np.log10(np.average(
                                np.sort(power)[:10]))
            power_list.append(power_db)
            noise_list.append(sequence_noise_db)
            max_snr_list.append(np.max(power_db[2:])-sequence_noise_db)
            timestamps.append(float(timestamp))
    power_array = np.array(power_list)
    
    start_time = datetime.datetime.utcfromtimestamp(timestamps[0])
    end_time = datetime.datetime.utcfromtimestamp(timestamps[-1])

    # x_lims = mdates.date2num([start_time, end_time])
    # y_lims = [start_sample, end_sample]
    
    # take the transpose to get sequences x samps for the antenna num
    new_power_array = np.transpose(power_array)

    kw = {'width_ratios': [95,5]}
    fig, ((ax1, cax1), (ax2, cax2)) = plt.subplots(2, 2, figsize=(32,16), 
                gridspec_kw=kw)
    fig.suptitle('{} PWR Sequence Time {} {} to {} UT vs Range'.format(
            dataset_descriptor, start_time.strftime('%Y%m%d'), 
            start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')))

    # plot SNR and noise (10 weakest ranges average)
    ax1.plot(range(len(max_snr_list)), max_snr_list)
    ax1.set_title('Max SNR in sequence')
    ax1.set_ylabel('SNR (dB)')

    img = ax2.imshow(new_power_array, aspect='auto', origin='lower', 
                    cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin)
    ax2.set_title('Range-time based on samples {} to {}'.format(start_sample,
                                end_sample))
    ax2.set_ylabel('Sample number (Range)')
    ax2.set_xlabel('Sequence number (spans time)')

    # extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]], 

    # Using datetimes as below can be misleading because it is just using
    # a range and our sequences are not necessarily evenly spaced across 
    # the time range. Not going to plot this way until a better method can
    # be found.
    # ax2.xaxis_date()
    # date_format = mdates.DateFormatter('%H:%M:%S')
    # ax2.xaxis.set_major_formatter(date_format)
    # fig.autofmt_xdate()
    # ax2.tick_params(axis='x', which='major', labelsize='15')
    fig.colorbar(img, cax=cax2, label='Raw Power (dB)')
    cax1.axis('off') 

    ax2.get_shared_x_axes().join(ax1, ax2)
    print(plot_filename)
    plt.savefig(plot_filename)
    plt.close() 


def plot_antennas_range_time(antennas_iq_file, antenna_nums=None, num_processes=3, vmax=40.0, vmin=10.0,
                             start_sample=0, end_sample=80):
    """ 
    Plots unaveraged range time data from echoes received in every sequence
    for a single antenna.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses gnuplot2 color map. 

    Parameters 
    ----------
    antennas_iq_file
        The filename that you are plotting data from for plot title. The file
        should be array restructured.
    antenna_nums
        List of antennas you want to plot. This is the antenna number 
        as listed in the antenna_arrays_order. The index into the data array
        is determined by finding the index of the antenna number into the 
        antenna_arrays_order list. The data array is organized main antennas 
        first consecutively, followed by interferometer antennas consecutively. 
        Default None, which allows the algorithm to plot all antennas available 
        in the dataset.
    num_processes
        The number of processes to use to plot the data.
    vmax
        Max power for the color bar on the plot. Default 40 dB.
    vmin
        Min power for the color bar on the plot.  Default 10 dB. 
    start_sample
        The sample to start plotting at. Default 0th range (first sample).
    end_sample
        The last sample in the sequence to plot. Default 70 so ranges 0-69
        will plot.
    """ 

    reader = BorealisRead(antennas_iq_file, 'antennas_iq', 
                          borealis_file_structure='array')
    arrays = reader.arrays

    (num_records, num_antennas, max_num_sequences, num_samps) = \
        arrays['data'].shape

    basename = os.path.basename(antennas_iq_file)
    directory_name = os.path.dirname(antennas_iq_file)
    time_of_plot = '.'.join(basename.split('.')[0:6])

    # typically antenna names and antenna indices are the same except 
    # where certain antennas were skipped in data writing for any reason.
    if antenna_nums is None:
        antenna_indices = list(range(0, num_antennas))
        antenna_names =  list(arrays['antenna_arrays_order'])
    else:
        antenna_indices = []
        antenna_names = antenna_nums
        for antenna_name in antenna_nums:
            antenna_indices.append(arrays['antenna_arrays_order'].index(
                'antenna_' + str(antenna_name)))

    sequences_data = arrays['num_sequences']
    timestamps_data = arrays['sqn_timestamps']

    arg_tuples = []
    print(antennas_iq_file)
    for antenna_num, antenna_name in zip(antenna_indices, antenna_names):
        antenna_data = arrays['data'][:,antenna_num,:,:]   
        plot_filename = directory_name + '/' + time_of_plot + \
                   '.{}_{}_{}.png'.format(antenna_name, start_sample,
                                          end_sample)
        arg_tuples.append((copy.copy(antenna_data), sequences_data, 
            timestamps_data, antenna_name, plot_filename, vmax, vmin, 
            start_sample, end_sample))

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
            p = Process(target=plot_unaveraged_range_time_data, 
                args=antenna_args)
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        antennas_index += num_processes


def plot_arrays_range_time(bfiq_file, beam_nums=None, num_processes=3, 
    vmax=50.0, vmin=10.0, start_sample=0, end_sample=70):
    """ 
    Plots unaveraged range time data from echoes received in every sequence
    for a single beam.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses gnuplot2 color map. 

    Parameters 
    ----------
    bfiq_file
        The filename that you are plotting data from for plot title. The file
        should be array restructured.
    beam_nums
        The list of beam numbers to plot. Default None which allows all beams 
        available in the file to be plotted.
    num_processes
        The number of processes to use to plot the data from the beam_nums.
    vmax
        Max power for the color bar on the plot. Default 60 dB.
    vmin
        Min power for the color bar on the plot.  Default 10 dB. 
    start_sample
        The sample to start plotting at. Default 0th range (first sample).
    end_sample
        The last sample in the sequence to plot. Default 70 so ranges 0-69
        will plot.
    """ 

    reader = BorealisRead(bfiq_file, 'bfiq', 
                          borealis_file_structure='array')
    arrays = reader.arrays

    (num_records, num_antenna_arrays, max_num_sequences, max_num_beams, 
        num_samps) = arrays['data'].shape

    basename = os.path.basename(bfiq_file)
    directory_name = os.path.dirname(bfiq_file)
    time_of_plot = '.'.join(basename.split('.')[0:6])

    # find the number of unique beams and their azimuths
    if beam_nums is None:
        # arrays['beam_nums'] is of shape num_records x max_num_beams
        # Note we do not want to include appended zeroes in the unique calc.
        all_beams = np.empty(0, dtype=np.uint32)
        for record_num in range(0, num_records):
            num_beams_in_record = int(arrays['num_beams'][record_num])
            all_beams = np.concatenate((all_beams, arrays['beam_nums'][
                record_num,:num_beams_in_record]))
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
        beam_timestamps_data = np.empty((0,max_num_sequences))
        beam_sequences_data = np.empty(0)
        beam_arrays_data = np.empty((0, num_antenna_arrays, 
            max_num_sequences, num_samps))
        for record_num in range(0, num_records):
            if beam_name in arrays['beam_nums'][record_num]:
                beam_index = list(arrays['beam_nums'][record_num]).index(beam_name)
                beam_timestamps_data = np.concatenate((beam_timestamps_data,
                    np.reshape(arrays['sqn_timestamps'][record_num], (1, 
                        max_num_sequences))))
                beam_sequences_data = np.concatenate((beam_sequences_data,
                    np.reshape(arrays['num_sequences'][record_num], 1)))
                beam_arrays_data = np.concatenate((beam_arrays_data, 
                    np.reshape(arrays['data'][record_num,
                        :,:,beam_index,:], (1, num_antenna_arrays, 
                            max_num_sequences, num_samps))))

        for array_num, array_name in enumerate(arrays['antenna_arrays_order']):
            plot_filename = directory_name + '/' + time_of_plot + \
                       '.{}_beam{}_{}_{}.png'.format(array_name, str(beam_name),
                                            start_sample, end_sample)
            descriptor = array_name + ' beam ' + str(beam_name)
            arg_tuples.append((copy.copy(beam_arrays_data[:,array_num,:,:]), 
                beam_sequences_data, beam_timestamps_data, descriptor, 
                plot_filename, vmax, vmin, start_sample, end_sample))

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

    
    # for beam in main_power_dict.keys():
    #     main_power_array = np.transpose(np.array(main_power_dict[beam])[:,0:69])
    #     intf_power_array = np.transpose(np.array(intf_power_dict[beam])[:,0:69])
    #     print(main_power_array.shape)
    #     fig, (ax1, ax2, cax) = plt.subplots(nrows=3, figsize=(32,24), gridspec_kw={'height_ratios':[0.4,0.4,0.1]})
    #     img1 = ax1.imshow(main_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin) #
    #     img2 = ax2.imshow(intf_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin)
    #     fig.colorbar(img1, cax=cax, orientation='horizontal')

    #     basename = bfiq_file.split('/')[-1]
    #     time_of_plot = '.'.join(basename.split('.')[0:6])
    #     plotname = time_of_plot + '.beam{}.png'.format(beam)
    #     print(plotname)
    #     plt.savefig(plotname)
    #     plt.close()
    #     snr = np.max(main_power_array) - np.mean(main_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
    #     print('Ave of array: {}'.format(np.average(main_power_array)))
    #     print('FILE {} beam {} main array snr: {}'.format(bfiq_file,beam,snr))


def plot_averaged_range_time_data(data_array, timestamps_array, 
    dataset_descriptor, plot_filename, vmax, vmin):
    """
    Plots data as range time given an array with correct dimensions. Also
    plots SNR by finding the ratio of max power in the sequence to average 
    power of the 10 weakest range gates.

    Note that this plots averaged data. 

    Parameters
    ----------
    data_array
        Array with shape num_records x num_samps (or num_ranges) for some 
        correlations dataset.
    timestamps_array
        Array of timestamps with dimensions num_records x max_num_sequences.
    dataset_descriptor
        Name for dataset, to be included in plot title.
    plot_filename
        Where to save plot.
    vmax
        Max power for the color bar on the plot. 
    vmin
        Min power for the color bar on the plot. 
    """
    print(dataset_descriptor)
    (num_records, num_samps) = data_array.shape

    power_list = [] # list of lists of power
    timestamps = [] # list of timestamps
    noise_list = [] # list of (average of ten weakest ranges in sample range)
    max_snr_list = [] # max power - noise (ave of 10 weakest ranges)
    for record_num in range(num_records):
        # get all ranges for this record
        voltage_samples = data_array[record_num,:]
        # get first timestamp in this record.
        timestamp = float(timestamps_array[record_num, 0])
        # power only. no averaging done. 
        power = np.sqrt(abs(voltage_samples))
        power_db = 10 * np.log10(power)
        # set noise to ave of 10 lowest ranges
        record_noise_db = 10 * np.log10(np.average(
                            np.sort(power)[:10]))
        power_list.append(power_db)
        noise_list.append(record_noise_db)
        max_snr_list.append(np.max(power_db[2:]) - record_noise_db)
        timestamps.append(timestamp)

    # want records x ranges
    new_power_array = np.transpose(np.array(power_list))
    
    start_time = datetime.datetime.utcfromtimestamp(timestamps[0])
    end_time = datetime.datetime.utcfromtimestamp(timestamps[-1])

    # x_lims = mdates.date2num([start_time, end_time])
    # y_lims = [start_sample, end_sample]

    kw = {'width_ratios': [95,5]}
    fig, ((ax1, cax1), (ax2, cax2)) = plt.subplots(2, 2, figsize=(32,16), 
                gridspec_kw=kw)
    fig.suptitle('{} PWR Time {} {} to {} UT vs Range'.format(
            dataset_descriptor, start_time.strftime('%Y%m%d'), 
            start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S')))

    # plot SNR and noise (10 weakest ranges average)
    ax1.plot(range(len(max_snr_list)), max_snr_list)
    ax1.set_title('Max SNR in record')
    ax1.set_ylabel('SNR (dB)')

    img = ax2.imshow(new_power_array, aspect='auto', origin='lower', 
                    cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin)
    ax2.set_title('Range-time plot')
    ax2.set_ylabel('Range gate number')
    ax2.set_xlabel('Record number (spans time)')

    # extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]], 

    # Using datetimes as below can be misleading because it is just using
    # a range and our sequences are not necessarily evenly spaced across 
    # the time range. Not going to plot this way until a better method can
    # be found.
    # ax2.xaxis_date()
    # date_format = mdates.DateFormatter('%H:%M:%S')
    # ax2.xaxis.set_major_formatter(date_format)
    # fig.autofmt_xdate()
    # ax2.tick_params(axis='x', which='major', labelsize='15')
    fig.colorbar(img, cax=cax2, label='SNR')
    cax1.axis('off') 

    ax2.get_shared_x_axes().join(ax1, ax2)
    print(plot_filename)
    plt.savefig(plot_filename)
    plt.close() 


def plot_rawacf_lag_pwr(rawacf_file, beam_nums=None, lag_nums=[0], 
    datasets=['main_acfs', 'intf_acfs', 'xcfs'], num_processes=3, vmax=50.0, 
    vmin=10.0):
    """
    Plots the lag xcf phase of rawacf.hdf5 file, lag number found via lag_index.
    
    Plots averaged range time data from echoes received for given beams
    and given lags (default all available beams lag0). Can plot all beams 
    and all lags. 

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses gnuplot2 color map. 

    Parameters 
    ----------
    rawacf_file
        The filename that you are plotting data from for plot title. The file
        should be array restructured.
    beam_nums
        The list of beam numbers to plot. Default None which allows all beams 
        available in the file to be plotted.
    lag_nums
        The list of lag numbers to plot. Default [0] to plot only lag0pwr.
        lag_nums=None allows all lags available in the file to be plotted.
    datasets
        The datasets to plot. Default all (acfs, intf_acfs, and xcfs). 
    num_processes
        The number of processes to use to plot the data from the beam_nums
        and lag_nums
    vmax
        Max power for the color bar on the plot. Default 60 dB.
    vmin
        Min power for the color bar on the plot.  Default 10 dB. 
    """

    reader = BorealisRead(rawacf_file, 'rawacf', 
                          borealis_file_structure='array')
    arrays = reader.arrays

    (num_records, max_num_beams, num_ranges, num_lags) = \
        arrays['main_acfs'].shape
    max_num_sequences = arrays['sqn_timestamps'].shape[1]

    basename = os.path.basename(rawacf_file)
    directory_name = os.path.dirname(rawacf_file)
    time_of_plot = '.'.join(basename.split('.')[0:6])

    # find the number of unique beams and their azimuths
    if beam_nums is None:
        # arrays['beam_nums'] is of shape num_records x max_num_beams
        # Note we do not want to include appended zeroes in the unique calc.
        all_beams = np.empty(0, dtype=np.uint32)
        for record_num in range(0, num_records):
            num_beams_in_record = int(arrays['num_beams'][record_num])
            all_beams = np.concatenate((all_beams, arrays['beam_nums'][
                record_num,:num_beams_in_record]))
        beam_names = np.unique(all_beams)
    else:
        beam_names = np.array(beam_nums, dtype=np.uint32)

    lag_names = arrays['lags']
    all_lag_nums = [lag[1] - lag[0] for lag in list(arrays['lags'])]
    if lag_nums is None:
        lag_nums = all_lag_nums
        lag_indices = list(range(len(lag_nums)))
    else:
        lag_indices = []
        for lag_num in lag_nums:
            try:
                lag_indices.append(all_lag_nums.index(lag_num))
            except ValueError as e:
                raise ValueError('Lag number {} is not found in the file'
                                 .format(lag_num))

    for dataset in datasets:
        if dataset not in ['main_acfs', 'intf_acfs', 'xcfs']:
            raise ValueError('Dataset not available in rawacf file: {}'
                             .format(dataset))
        

    arg_tuples = []
    print(rawacf_file)

    for beam_name in beam_names:
        # find only the data with this beam name
        # data will be num_records avail (with this beam and lag number)
        # to plot x num_ranges
        beam_timestamps_data = np.empty((0,max_num_sequences))
        for lag_index, lag_num in zip(lag_indices, lag_nums):
            beam_lag_dict = {}
            for dataset in datasets:
                beam_lag_dict[dataset] = np.empty((0, num_ranges))
            for record_num in range(0, num_records):
                if beam_name in arrays['beam_nums'][record_num]:
                    beam_index = list(arrays['beam_nums'][record_num]).index(beam_name)
                    beam_timestamps_data = np.concatenate((beam_timestamps_data,
                        np.reshape(arrays['sqn_timestamps'][record_num], (1, 
                            max_num_sequences))))
                    for dataset in datasets:
                        beam_lag_dict[dataset] = np.concatenate((
                            beam_lag_dict[dataset],
                            np.reshape(arrays[dataset][record_num,beam_index,
                                :,lag_index], (1, num_ranges))))

            for dataset in datasets:
                plot_filename = directory_name + '/' + time_of_plot + \
                           '.{}_beam{}_lag{}.png'.format(dataset, beam_name,
                                                lag_num)
                descriptor = dataset + ' beam ' + str(int(beam_name)) + ' lag ' + \
                    str(lag_num)
                arg_tuples.append((copy.copy(beam_lag_dict[dataset]), 
                    beam_timestamps_data, descriptor, plot_filename, vmax, 
                    vmin))

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

