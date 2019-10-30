# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
This module contains plotting functions for range time plots 
from Borealis HDF5 files of the following types:
- antennas_iq
- bfiq
- rawacf

These functions can be used on site to plot Borealis HDF5 files
for testing.

"""

from pydarn import BorealisRead, BorealisWrite

def plot_antennas_range_time(antennas_iq_file, antennas_iq_arrays,
                             antenna_num, vmax=80.0, vmin=10.0, 
                             start_sample=0, end_sample=70):
    """ 
    Plots unaveraged range time data from echoes received in every sequence
    for a single antenna.

    Gets the samples between start_sample and end_sample for every
    sequence in the file, calculates their power, and plots these sequences side
    by side. Uses gnuplot2 color map. 

    Parameters 
    ----------
    antennas_iq_file
        The filename that you are plotting data from for plot title. The 
        file should be read before being passed in, using the antennas_iq_arrays
        parameter. 
    antennas_iq_arrays
        Arrays from the BorealisRead instance that read the file. (e.g. 
        reader.arrays)
    antenna_num
        The antenna that you want to plot. Used to index into the data array, 
        which is organized main antennas first consecutively, followed by 
        interferometer antennas consecutively. 
    vmax
        Max power for the color bar on the plot. 
    vmin
        Min power for the color bar on the plot. 
    start_sample
        The sample to start plotting at. Default 0th range (first sample).
    end_sample
        The last sample in the sequence to plot. Default 70 so ranges 0-69
        will plot.
    """ 
    print(antennas_iq_file, antenna_num)

    arrays = antennas_iq_arrays

    (num_records, num_antennas, max_num_sequences, num_samps) = \
        arrays['data'].shape

    power_list = [] # list of lists of power
    timestamps = [] # list of timestamps
    noise_list = [] # list of (average of ten weakest ranges in sample range)
    max_snr_list = [] # max power - sequence noise (ave of 10 weakest ranges)
    for record_num in range(num_records):
        num_sequences = arrays['num_sequences'][record_num]
        # get all antennas, up to num sequences, all samples for this record.
        voltage_samples = arrays['data'][record_num,:,:num_sequences,:]

        for sequence in range(num_sequences):
            timestamp = float(arrays['sqn_timestamps'][record_num, sequence])
            # power only. no averaging done. 
            power = np.sqrt(voltage_samples.real**2 + voltage_samples.imag**2)[
                        antenna_num,sequence,start_sample:end_sample]
            power_db = 10 * np.log10(power)
            sequence_noise_db = 10 * np.log10(np.average(np.sort(power)[:10]))
            power_list.append(power_db)
            noise_list.append(sequence_noise_db)
            max_snr_list.append(np.max(power_db)-sequence_noise_db)
            timestamps.append(float(timestamp))
    power_array = np.array(power_list)
    
    start_time = datetime.datetime.fromtimestamp(timestamps[0])
    end_time = datetime.datetime.fromtimestamp(timestamps[-1])

    x_lims = mdates.date2num([start_time, end_time])
    y_lims = [start_sample, end_sample]
    # take the transpose to get sequences x samps for the antenna num
    new_power_array = np.transpose(power_array)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(32,16))
    img = ax1.imshow(new_power_array, extent=[x_lims[0], x_lims[1], y_lims[0], 
                    y_lims[1]], aspect='auto', origin='lower', 
                    cmap=plt.get_cmap('gnuplot2'), vmax=vmax, vmin=vmin)
    
    plt.title('Antenna {} PWR Sequence Time {} {} to {} vs Range'.format(
            antenna_num, start_time.strftime('%Y%m%d'), 
            start_time.strftime('%H%M%S'), end_time.strftime('%H%M%S')))
    ax1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax1.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    ax1.tick_params(axis='x', which='major', labelsize='15')
    fig.colorbar(img)

    basename = antennas_iq_file.split('/')[-1]
    time_of_plot = '.'.join(basename.split('.')[0:6])
    plotname = time_of_plot + '.antenna{}_{}_{}.png'.format(antenna_num, 
                start_sample, end_sample)
    print(plotname)
    plt.savefig(plotname)
    plt.close() 

    # plot SNR and noise (10 weakest ranges average)
    ax2.plot(range(len(max_snr_list)), max_snr_list)


def plot_bfiq_file_power(bfiq_file, vmax=-50.0, vmin=-120.0, beam_num=7):
    """
    Plots the bfiq file. Takes the first 70 samples from first (often only) beam for each array in all records and plots them out.
    No averaging is done, but instead sequences are plotted side by side. Plots two figures in total, one for each array.
    DOes not take into account varying beam directions between records!!

    """
    record_names = {}
    power_dict = {}
    beam_num = 0

    print(bfiq_file)
    records = sorted(deepdish.io.load(bfiq_file).keys())
    record_names[bfiq_file] = records
    power_dict[bfiq_file] = {}
    for array in range(0,2):
        power_list = []
        for record_name in record_names[bfiq_file]:
            groupname = '/' + record_name
            data = deepdish.io.load(bfiq_file,group=groupname)
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

        #power_dict[bfiq_file][array] = new_power_array

        fig, ax = plt.subplots(figsize=(32,16))
        img = ax.imshow(new_power_array, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'), vmax = vmax, vmin = vmin) #
        fig.colorbar(img)

        basename = bfiq_file.split('/')[-1]
        time_of_plot = '.'.join(basename.split('.')[0:6])
        plotname = time_of_plot + '.array{}.png'.format(array)
        print(plotname)
        plt.savefig(plotname)
        plt.close() 
        snr = np.max(new_power_array) - np.mean(new_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
        print(snr)


def plot_normalscan_bfiq_averaged_power_by_beam(bfiq_file, vmax=110.0, vmin=30.0):
    """
    Plots the bfiq file. Takes the first 70 samples from first (often only) beam for each array in all records and plots them out.
    No averaging is done, but instead sequences are plotted side by side. Plots two figures in total, one for each array.
    DOes not take into account varying beam directions between records!!

    """
    record_names = {}
    main_power_dict = {}
    intf_power_dict = {}
    beam_num = 0

    print(bfiq_file)
    records = sorted(deepdish.io.load(bfiq_file).keys())
    record_names[bfiq_file] = records
    for record_name in record_names[bfiq_file]:
        power_list = []
        groupname = '/' + record_name
        data = deepdish.io.load(bfiq_file,group=groupname)
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

        basename = bfiq_file.split('/')[-1]
        time_of_plot = '.'.join(basename.split('.')[0:6])
        plotname = time_of_plot + '.beam{}.png'.format(beam)
        print(plotname)
        plt.savefig(plotname)
        plt.close()
        snr = np.max(main_power_array) - np.mean(main_power_array[:,10], axis=0) # use average of a close range as noise floor, see very little there (45 * 7 = 315 km range)
        print('Ave of array: {}'.format(np.average(main_power_array)))
        print('FILE {} beam {} main array snr: {}'.format(bfiq_file,beam,snr))


def plot_lag_xcfphase(acf_file):
    """
    Plots the lag xcf phase of rawacf.hdf5 file, lag number found via lag_index.
    """

    try:
        with h5py.File(acf_file, 'r') as f:
            group_names = sorted(list(f.keys()))
    except RuntimeError:
        print('FILE not read due to RuntimeError: {}'.format(acf_file))
        return
    first_record_lags = deepdish.io.load(acf_file, group= '/' + group_names[0])['lags']
    lag_numbers = [lags[1] - lags[0] for lags in first_record_lags]

    for lag_index in range(0,len(lag_numbers)):
        lag_number = lag_numbers[lag_index]
        xcf_phases = []
        for group_name in group_names:
            record = deepdish.io.load(acf_file, group='/' + group_name)
            xcf = record['xcfs'].reshape(record['correlation_dimensions'])
            dimensions_len = len(record['correlation_dimensions'])
     
            beam_index=0
            lag = xcf[beam_index,:,lag_index] # this beam, all ranges lag 
            xcf_phase = np.angle(lag, deg=True) # degrees
            xcf_phases.append(xcf_phase)
        
        xcf_to_plot = np.transpose(np.array(xcf_phases, dtype=np.float32))
        
        fig, ax = plt.subplots(figsize=(32,16)) 
        img = ax.imshow(xcf_to_plot, aspect='auto', origin='lower', cmap=plt.get_cmap('gnuplot2'))
        fig.colorbar(img)
        
        basename = acf_file.split('/')[-1]
        time_of_plot = '.'.join(basename.split('.')[0:6])
        plotname = time_of_plot + '.lagind' + str(lag_index) + 'xcfphase.png'
        ax.set_title(time_of_plot + ' Lag index {}, lag num {}'.format(lag_index, lag_number))
        print(plotname)
        plt.savefig(plotname)
        plt.close()


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

