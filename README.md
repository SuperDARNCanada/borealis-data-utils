# borealis-data-utils

Scripts and utilities for analyzing Borealis data (operations focused). 

**Dependencies**
pydarn
pydarnio

**Utilities**

*borealis_gaps*
Reads borealis files (that are organized in day subdirectories) and finds
any gaps in sequence timestamps in the file. Reports these gaps in a 
table of downtimes with duration in minutes. This table is written in 
markdown format so that the command line utility pandoc can be used to 
convert the table into a word document or latex table for use in reports.

*borealis_fixer*
A script for fixing hdf5 files. Can be modified to edit Borealis HDF5 files 
if an error in data field or format is found that can be easily fixed.
Should be used with caution and only by data producers internally.

*SNR-plotters*
A collection of plotting utilities for plotting signal to noise ratio
of the bfiq and antennas_iq 'data' fields organized in range vs time, 
and for plotting lag power of the rawacf 'acfs' field. This can be used 
to quickly verify data.

*pydarn-plotters*
A collection of plotting scripts using the RTP class in pydarn, for use
after Borealis data has been converted to SDARN filetype (.dmap). 
Includes scripts for range time plots of both iqdat and rawacf file types 
and scripts for fitacf summary plots. 

*time-domain-plotters*
Basic plotters for plotting amplitude vs time of sequences sampled in 
antennas_iq, bfiq, txdata, or rawrf data. For Borealis testing. 

