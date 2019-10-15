import matplotlib
matplotlib.use('Agg')

import sys
import os
import math
import numpy as np
import struct
import random
import deepdish
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
import glob
from scipy.fftpack import fft
from multiprocessing import Pool, Process

iq_file_1 = '/data1/borealis_data/20190802/20190802.0000.02.sas.0.antennas_iq.hdf5'
#iq_file_1 = '/data/borealis_data/20190710/20190710.1609.44.sas.0.output_ptrs_iq.hdf5'
#glob.glob('/data/tempdat/detwiller/20190402.17*')

#colour_ranges = { iq_file : {'vmax' : 110.0, 'vmin' : 30.0} for iq_file in iq_files} # 

def plot_antennas_iq_file_power(iq_file, antenna_num, vmax=80.0, vmin=10.0):
    """
    Plots the output_ptrs_iq file. Takes the first 70 samples from all records and plots them out.
    No averaging is done, but instead sequences are plotted side by side. Plots a figure per antenna.
    """
    #record_names = {}
    #power_dict = {}
    
    print(iq_file, antenna_num)
    records = sorted(deepdish.io.load(iq_file).keys())
    #power_dict[iq_file] = {}
    power_list = []
    for record_name in records:
        groupname = '/' + record_name
        data = deepdish.io.load(iq_file,group=groupname)
        voltage_samples = data['data'].reshape(data['data_dimensions'])
        #print(data['data_descriptors'])
        num_antennas, num_sequences, num_samps = data['data_dimensions']
        for sequence in range(num_sequences):
            timestamp = float(data['sqn_timestamps'][sequence])
            #print(timestamp)
            # power only. no averaging done. 
            power = (voltage_samples.real**2 + voltage_samples.imag**2)[antenna_num,sequence,0:69]
            power_db = 10 * np.log10(power)
            #power_dict[timestamp] = power_db
            power_list.append(power_db)
    power_array = np.array(power_list)

    # take the transpose to get sequences x samps for the antenna num
    new_power_array = np.transpose(power_array)

    #power_dict[iq_file][array] = new_power_array

    fig, ax = plt.subplots(figsize=(32,16))
    img = ax.imshow(new_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax = vmax, vmin = vmin) #
    fig.colorbar(img)

    basename = iq_file.split('/')[-1]
    time_of_plot = '.'.join(basename.split('.')[0:6])
    plotname = time_of_plot + '.antenna{}.png'.format(antenna_num)
    print(plotname)
    plt.savefig(plotname)
    plt.close() 
    snr = np.max(new_power_array) - np.mean(new_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
    print('antenna {} SNR: {}'.format(antenna_num, snr))


if __name__ == '__main__':
    
    antenna_range = list(range(0,20))
    
    arg_tuples = []

    for antenna_num in antenna_range:
        arg_tuples.append((iq_file_1, antenna_num))
 
#    arg_tuples.append((iq_file_1, 19))   
    pool = Pool(processes=2)  # 8 worker processes
    pool.starmap(plot_antennas_iq_file_power, arg_tuples)
        
        #p = Process(target=plot_output_ptrs_iq_file_power, args=(iq_file_1, antenna_num,))
        #p.start()

    #p.join()

