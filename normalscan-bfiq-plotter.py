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

iq_files = glob.glob('/data/tempdat/detwiller/*.bfiq.hdf5')
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190409.*.bfiq.hdf5'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190403*'))

#colour_ranges = { iq_file : {'vmax' : 30.0, 'vmin' : -50.0} for iq_file in iq_files} # 40 dB difference between scaling factors.


def plot_normalscan_bfiq_averaged_power_by_beam(iq_file, vmax=110.0, vmin=30.0):
    """
    Plots the bfiq file. Takes the first 70 samples from first (often only) beam for each array in all records and plots them out.
    No averaging is done, but instead sequences are plotted side by side. Plots two figures in total, one for each array.
    DOes not take into account varying beam directions between records!!

    """
    record_names = {}
    main_power_dict = {}
    intf_power_dict = {}
    beam_num = 0

    print(iq_file)
    records = sorted(deepdish.io.load(iq_file).keys())
    record_names[iq_file] = records
    for record_name in record_names[iq_file]:
        power_list = []
        groupname = '/' + record_name
        data = deepdish.io.load(iq_file,group=groupname)
        voltage_samples = data['data'].reshape(data['data_dimensions'])
            #print(data['data_descriptors'])
        num_arrays, num_sequences, num_beams, num_samps = data['data_dimensions']
        timestamps = [float(i) for i in data['sqn_timestamps']]
                #print(timestamp)
                # averaging here
        for array in range(0,2):
            averaged_power = np.mean(((voltage_samples.real**2 + voltage_samples.imag**2)/50.0)[array,:,beam_num,:], axis=0)
#            print(averaged_power.shape)
            power_db = 10 * np.log10(averaged_power)
            #print(np.max(power_db))
            beam_dir = data['beam_nums'][0]
            if array==0:
                if beam_dir not in main_power_dict.keys():
                    main_power_dict[beam_dir] =[list(power_db)]
                else:
                    main_power_dict[beam_dir].append(list(power_db))
            else:
                if beam_dir not in intf_power_dict.keys():
                    intf_power_dict[beam_dir] = [list(power_db)]
                else:
                    intf_power_dict[beam_dir].append(list(power_db))
    
    for beam in main_power_dict.keys():
        main_power_array = np.transpose(np.array(main_power_dict[beam])[:,0:69])
        intf_power_array = np.transpose(np.array(intf_power_dict[beam])[:,0:69])
        print(main_power_array.shape)
        fig, (ax1, ax2, cax) = plt.subplots(nrows=3, figsize=(32,24), gridspec_kw={'height_ratios':[0.4,0.4,0.1]})
        img1 = ax1.imshow(main_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin) #
        img2 = ax2.imshow(intf_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin)
        fig.colorbar(img1, cax=cax, orientation='horizontal')

        basename = iq_file.split('/')[-1]
        time_of_plot = '.'.join(basename.split('.')[0:6])
        plotname = time_of_plot + '.beam{}.png'.format(beam)
        print(plotname)
        plt.savefig(plotname)
        plt.close()
        snr = np.max(main_power_array) - np.mean(main_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
        print('Ave of array: {}'.format(np.average(main_power_array)))
        print('FILE {} beam {} main array snr: {}'.format(iq_file,beam,snr))


if __name__ == '__main__':
    pool = Pool(processes=8)  # 8 worker processes
    pool.map(plot_normalscan_bfiq_averaged_power_by_beam, iq_files)


