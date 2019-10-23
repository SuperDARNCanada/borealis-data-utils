
import sys
import pydarn
import matplotlib.pyplot as plt

fitacf_file = sys.argv[1]

pydarn_reader = pydarn.DarnRead(fitacf_file)
fitacf_data = pydarn_reader.read_fitacf()


pydarn.RTP.plot_summary(fitacf_data, figsize=(22, 17), beam_num=0, groundscatter=True)
plt.savefig(fitacf_file + '.summaryplot.png', dpi=400)
#plt.show()

