import pydarn
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os

plt.rcParams.update({'font.size': 28})

rawacf_file = sys.argv[1]
darn_read = pydarn.DarnRead(rawacf_file)
rawacf_data = darn_read.read_rawacf()

rawacf_file_basename = os.path.basename(rawacf_file)

lognorm = colors.LogNorm

plt.figure(figsize=(32,16))

pydarn.RTP.plot_range_time(rawacf_data, beam_num=0, parameter='pwr0',
                          color_bar_label="pwr0", color_norm=lognorm,
                          color_map='gnuplot2')

plt.title("Borealis " + rawacf_file_basename + " pwr0")
plt.ylabel('Range Gates')
plt.xlabel('Date (UTC)')

plt.savefig(rawacf_file+'.rangetime.png')
plt.show()