import pydarn
import matplotlib.pyplot as plt
from matplotlib import colors
import sys
import os

plt.rcParams.update({'font.size': 28})

fitacf_file = sys.argv[1]
fitacf_data = pydarn.SuperDARNRead.read_dmap(fitacf_file)

fitacf_file_basename = os.path.basename(fitacf_file)

lognorm = colors.LogNorm

plt.figure(figsize=(32,16))

pydarn.RTP.plot_range_time(fitacf_data, beam_num=0, parameter='p_l',
                          color_bar_label="p_l", color_norm=lognorm,
                          color_map='gnuplot2')

plt.title("Borealis " + fitacf_file_basename + " pwr0")
plt.ylabel('Range Gates')
plt.xlabel('Date (UTC)')

plt.show()
