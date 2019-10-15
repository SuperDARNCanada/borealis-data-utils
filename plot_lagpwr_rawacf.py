import matplotlib
matplotlib.use('Agg')

import sys
import os
import math
import numpy as np
import struct
import random
import deepdish
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
import glob
from scipy.fftpack import fft
from multiprocessing import Pool




acf_files = glob.glob('/data/tempdat/detwiller/20190523/20190523.*.rawacf.hdf5')

def plot_lag_pwr(acf_file):
    """
    Plots the lag pwer of rawacf.hdf5 file, lag number found via lag_index.
    """

    try:
        with h5py.File(acf_file, 'r') as f:
            group_names = sorted(list(f.keys()))
    except RuntimeError:
        print('FILE not read due to RuntimeError: {}'.format(acf_file))
        return
    first_record_lags = deepdish.io.load(acf_file, group= '/' + group_names[0])['lags']
    lag_numbers = [lags[1] - lags[0] for lags in first_record_lags]

    #lag_indices = range(0,len(lag_numbers))
    lag_indices = [0]
    for lag_index in lag_indices:
        lag_number = lag_numbers[lag_index]
        lag_powers = []
        for group_name in group_names:
            record = deepdish.io.load(acf_file, group='/' + group_name)
            try:
                acf = record['main_acfs'].reshape(record['correlation_dimensions'])
                dimensions_len = len(record['correlation_dimensions'])
            except (KeyError, AttributeError):
                if record['main_acfs'] is not None: # not None
                    acf = record['main_acfs'].reshape([75,-1])   
                    dimensions_len = 2
                else:
                    return       
     
            if dimensions_len == 3:
                beam_index=0
                lag = acf[beam_index,:,lag_index] # this beam, all ranges lag 
                lag_power = 10*np.log10(lag.real**2 + lag.imag**2)
                lag_powers.append(lag_power)
            else:
                lag = acf[:,lag_index] # this beam, all ranges lag 0
                lag_power = 10*np.log10(lag.real**2 + lag.imag**2)
                lag_powers.append(lag_power)  
    
        lagpwr = np.transpose(np.array(lag_powers, dtype=np.float32))
        
        fig, ax = plt.subplots(figsize=(32,16)) 
        img = ax.imshow(lagpwr, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'))
        fig.colorbar(img)
        
        basename = acf_file.split('/')[-1]
        time_of_plot = '.'.join(basename.split('.')[0:6])
        plotname = time_of_plot + '.lagind' + str(lag_index) + 'pwr.png'
        ax.set_title(time_of_plot + ' Lag index {}, lag num {}'.format(lag_index, lag_number))
        print(plotname)
        plt.savefig(plotname)
        plt.close()

if __name__ == '__main__':
    pool = Pool(processes=16)  # worker processes
    # arguments = [(acf, range(0,23)) for acf in acf_files]
    pool.map(plot_lag_pwr, acf_files)


