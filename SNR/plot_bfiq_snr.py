# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This script is used to 
"""

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

from hdf5_rtplot_utils import plot_bfiq_file_power

iq_files = glob.glob('/data/tempdat/detwiller/20190407*.bfiq.hdf5')
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.18*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190402.2*'))
#iq_files.extend(glob.glob('/data/tempdat/detwiller/20190403*'))

colour_ranges = { iq_file : {'vmax' : 110.0, 'vmin' : 30.0} for iq_file in iq_files} # 40 dB difference between scaling factors.

#colour_ranges = { iq_file : {'vmax' : 30.0, 'vmin' : -50.0} for iq_file in iq_files} # 40 dB difference between scaling factors.



if __name__ == '__main__':
    pool = Pool(processes=8)  # 8 worker processes
    pool.map(plot_bfiq_file_power, iq_files)



# if __name__ == '__main__':
#     pool = Pool(processes=8)  # 8 worker processes
#     pool.map(plot_normalscan_bfiq_averaged_power_by_beam, iq_files)