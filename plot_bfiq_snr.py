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
from multiprocessing import Pool




iq_files = glob.glob('/data/tempdat/detwiller/20190407*.bfiq.hdf5')
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.18*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190403*'))

colour_ranges = { iq_file : {'vmax' : 110.0, 'vmin' : 30.0} for iq_file in iq_files} # 40 dB difference between scaling factors.


def plot_bfiq_file_power(iq_file, vmax=-50.0, vmin=-120.0, beam_num=7):
    """
    Plots the bfiq file. Takes the first 70 samples from first (often only) beam for each array in all records and plots them out.
    No averaging is done, but instead sequences are plotted side by side. Plots two figures in total, one for each array.
    DOes not take into account varying beam directions between records!!

    """
    record_names = {}
    power_dict = {}
    beam_num = 0

    print(iq_file)
    records = sorted(deepdish.io.load(iq_file).keys())
    record_names[iq_file] = records
    power_dict[iq_file] = {}
    for array in range(0,2):
        power_list = []
        for record_name in record_names[iq_file]:
            groupname = '/' + record_name
            data = deepdish.io.load(iq_file,group=groupname)
            if data['beam_nums'][0] != beam_num:
                continue
            voltage_samples = data['data'].reshape(data['data_dimensions'])
            #print(data['data_descriptors'])
            num_arrays, num_sequences, num_beams, num_samps = data['data_dimensions']
            for sequence in range(num_sequences):
                timestamp = float(data['sqn_timestamps'][sequence])
                #print(timestamp)
                # power only. no averaging done. 
                power = (voltage_samples.real**2 + voltage_samples.imag**2)[array,sequence,beam_num,0:69]
                power_db = 10 * np.log10(power)
                power_dict[timestamp] = power_db
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
        plotname = time_of_plot + '.array{}.png'.format(array)
        print(plotname)
        plt.savefig(plotname)
        plt.close() 
        snr = np.max(new_power_array) - np.mean(new_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
        print(snr)


if __name__ == '__main__':
    pool = Pool(processes=8)  # 8 worker processes
    pool.map(plot_bfiq_file_power, iq_files)


