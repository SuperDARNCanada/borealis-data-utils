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
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
import glob
from scipy.fftpack import fft
from multiprocessing import Pool




acf_files = glob.glob('/data/tempdat/detwiller/20190523/20190523.00*.rawacf.hdf5')



if __name__ == '__main__':
    pool = Pool(processes=16)  # worker processes
    # arguments = [(acf, range(0,23)) for acf in acf_files]
    pool.map(plot_lag_xcfphase, acf_files)


