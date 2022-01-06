# Copyright 2021 SuperDARN Canada, University of Saskatchewan
# Author: Remington Rohel
"""
This file contains functions for "restructuring" site to array Borealis HDF5 files on memory-limited systems,
for the express purpose of plotting with hdf5_rtplot_utils.py.

Functions
---------
"""
import copy
import numpy as np
import os
import deepdish as dd


def antennas_iq_site_to_array(antennas_iq_file, antenna_nums=None):
    """
    Gets the minimal fields necessary to build up an array-structured representation of antennas_iq_file
    for plotting.

    Parameters
    ----------
    antennas_iq_file:   str
        Path to an antennas_iq site-structured file.
    antenna_nums:       list
        List of antenna indices to retrieve. Entries should be ints.

    Returns
    -------
    arrays: dict
        Dictionary of necessary fields for plotting.
    antenna_names: list
        List of all the antenna names.
    antenna_indices: list
        List of all the indices in the data corresponding to antenna_indices.
    """
    # Dictionary of fields that will be returned
    arrays = {}

    group = dd.io.load(antennas_iq_file)
    timestamps = sorted(list(group.keys()))

    arrays['antenna_arrays_order'] = group[timestamps[0]]['antenna_arrays_order']

    num_records = len(timestamps)
    num_sequences_array = np.zeros(num_records, dtype=np.uint32)

    # typically, antenna names and antenna indices are the same except
    # where certain antennas were skipped in data writing for any reason.
    if antenna_nums is None or len(antenna_nums) == 0:
        antenna_indices = list(range(0, arrays['antenna_arrays_order'].size))
        antenna_names = list(arrays['antenna_arrays_order'])
    else:
        antenna_indices = []
        antenna_names = [f'antenna_{a}' for a in antenna_nums]
        for antenna_num in antenna_nums:
            antenna_indices.append(list(arrays['antenna_arrays_order']).index('antenna_' + str(antenna_num)))

    # Get the maximum number of sequences for a record in the file
    max_sequences = 0
    for i, timestamp in enumerate(timestamps):
        record = group[timestamp]
        _, num_seqs, _ = record['data_dimensions']
        num_sequences_array[i] = num_seqs
        if num_seqs > max_sequences:
            max_sequences = num_seqs

    # We will need these dimensions for pre-allocating arrays
    data_dimensions = group[timestamps[0]]['data_dimensions']

    # Create arrays for each of the needed fields
    timestamps_data = np.zeros((num_records, max_sequences), dtype=np.float64)
    data_array = np.zeros((num_records, len(antenna_indices), max_sequences, data_dimensions[2]), dtype=np.complex64)

    # Copy the relevant data into the arrays
    for i, timestamp in enumerate(timestamps):
        record = group[timestamp]
        num_antennas, num_seqs, num_samps = record['data_dimensions']
        timestamps_data[i, :num_seqs] = record['sqn_timestamps']
        data = record['data'].reshape(record['data_dimensions'])
        data_array[i, :, :num_seqs, :] = \
            data[antenna_indices, ...]

    # Copy to dictionary and return
    arrays['data'] = data_array
    arrays['sqn_timestamps'] = timestamps_data
    arrays['num_sequences'] = num_sequences_array

    return arrays, antenna_names, antenna_indices
