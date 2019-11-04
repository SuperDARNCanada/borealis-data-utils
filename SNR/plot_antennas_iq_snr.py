# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to 
"""


import matplotlib
matplotlib.use('Agg')

import copy
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

from hdf5_rtplot_utils import plot_antennas_range_time
from pydarn import BorealisRead

iq_file_1 = '/sddata/sas_borealis_data/antennas_iq/20190911/20190911.1600.02.sas.0.antennas_iq.hdf5'
#iq_file_1 = '/data1/borealis_data/20190802/20190802.0000.02.sas.0.antennas_iq.hdf5'
#iq_file_1 = '/data/borealis_data/20190710/20190710.1609.44.sas.0.output_ptrs_iq.hdf5'
#glob.glob('/data/tempdat/detwiller/20190402.17*')

#colour_ranges = { iq_file : {'vmax' : 110.0, 'vmin' : 30.0} for iq_file in iq_files} # 

if __name__ == '__main__':
    
    antenna_range = list(range(0,20))
    
    arg_tuples = []

    reader = BorealisRead(iq_file_1, 'antennas_iq', 'array')
    arrays = reader.arrays

    for antenna_num in antenna_range:
        arg_tuples.append((iq_file_1, copy.deepcopy(arrays), antenna_num))
 
#    arg_tuples.append((iq_file_1, 19))   
    pool = Pool(processes=3)  # 8 worker processes
    pool.starmap(plot_antennas_range_time, arg_tuples)
        
        #p = Process(target=plot_output_ptrs_iq_file_power, args=(iq_file_1, antenna_num,))
        #p.start()

    #p.join()

