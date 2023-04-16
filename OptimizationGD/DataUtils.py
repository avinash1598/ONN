"""
TIP: 
    1. Never use  [[0]*3]*3 for array assignment - This creates three duplicate list. Check out below code to view the weird result it gives:
        X = [[2]*3]*2
        X[0][0] = 3
        print(X)
        #Extremely weired!!! This is assigning to all the coloumn 0th element to 3.
        
        TODO:
        Reconstruct between -1 and 1
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from tbd_eeg.data_analysis.eegutils import EEGexp
#pip install -e .
from glob import glob
from os import path
import scipy
from scipy import signal
import pandas as pd
        
class DataUtils:
    
    """
    Variables holding data:
        1. self.exp
        2. self.eegdata (EEG data for all 30 channels)
        3. self.processedData (filtered EEg signal)
        4. self.segmented_data (EEG signal cut into 3s intervals)
        5. self.ERP (ERP for each of each of the stimulus group)
    """
    
    #Constants
    #TODO: include contralateral as well
    INTER_REGION_CONN_FILE = 'inter-region-p-value-filtered-labelled.csv'
    INTER_REGION_PVAL_FILE = 'regions-p-values - unlabelled.csv'
    DATASTORE = '/home/jupyter-avinash/datastore/allen_mouse_eeg'
    MOUSE = 'mouse599975'
    EXP = 'estim_vis_2022-03-31_12-03-06'
    STIM_DATA = DATASTORE + '/' + MOUSE + '/' + EXP + '/experiment1/recording1/all_stim_log.csv'
    SAMPLING_FREQ = 500 #Sampling frequency
    NOTCH_FLT_FREQ = 30 #Notch filter frequency
    AC_FREQ = 50        #Frequency to remove
    LPF_FC = 100        #Cutoff frequency for low pass filter
    
    def __init__(self):        
        self.init_inter_region_connectivity()
        
        #self.available_session()
        #self.experiment_info()
        #self.load_data()
        #self.filter_data()
        #self.preprocess_data()
    
    def available_session(self):
        print('Available sessions:')
        print([path.basename(f) for f in glob(path.join(DataUtils.DATASTORE, DataUtils.MOUSE, '*'))])
        
    def experiment_info(self):
        rec_folder = path.join(DataUtils.DATASTORE, DataUtils.MOUSE, DataUtils.EXP, 'experiment1/recording1')
        self.exp = EEGexp(rec_folder, preprocess=False, make_stim_csv=False)
        print(self.exp.ephys_params['EEG'])
    
    def load_data(self):
        print(self.exp.stimulus_log_file)
        #Deafult sampling rate si 250 and not 2500 - rhythm_info, structure.oebin
        ## loading the EEG data
        self.eegdata = self.exp.load_eegdata(return_type='pd', downsamplefactor=1)
        display(self.eegdata.head())
    
    def filter_data(self):
        #Design notch filter
        b_notch, a_notch = signal.iirnotch(DataUtils.AC_FREQ, DataUtils.NOTCH_FLT_FREQ, DataUtils.SAMPLING_FREQ)
        #Design low pass filter
        b_lpf, a_lpf = signal.butter(9, DataUtils.LPF_FC/(DataUtils.SAMPLING_FREQ/2), btype='low')
        #Frequency response
        freq_notch, h_notch = signal.freqz(b_notch, a_notch, fs=DataUtils.SAMPLING_FREQ)
        freq_lpf, h_lpf = signal.freqz(b_lpf, a_lpf, fs=DataUtils.SAMPLING_FREQ)

        f = plt.figure()
        plt.plot(freq_notch, abs(h_notch))
        plt.plot(freq_lpf, abs(h_lpf))
        plt.show()
        
        #Pre-process and update panda dataframe
        fsig = []
        for col in self.eegdata:
            #Apply notch filter
            fltsig = signal.filtfilt(b_notch, a_notch, self.eegdata.iloc[:,col])
            #Apply low pass filter
            fltsig = signal.filtfilt(b_lpf, a_lpf, fltsig)
            fsig.append(fltsig)

        fsig = np.array(fsig).T
        self.processedData = pd.DataFrame(fsig, index=self.eegdata.index)
    
    def preprocess_data(self):
        """
        Variables containing data: self.eegdata, self.processedData
        """
        """
            1. Get segments
        """
        stim_data = pd.read_csv(DataUtils.STIM_DATA)
        df = stim_data.groupby(['sweep', 'stim_type', 'parameter'])
        print(df.groups.keys())

        grp_idxs = []
        for key in df.groups.keys():
            grp_idxs.append(df.get_group(key))

        """
        Set this value to choose specific group
        ([(0, 'biphasic', '30'), (0, 'biphasic', '50'), (0, 'biphasic', '70'), 
          (0, 'circle', 'white'), (1, 'biphasic', '30'), (1, 'biphasic', '50'), 
          (1, 'biphasic', '70'), (1, 'circle', 'white')]
        """
        self.GRP = 7

        #biphasic - slectrical stimulus
        #circle - visual
        NUM_STIM = len(stim_data)
        idx = np.where(np.in1d(np.array(stim_data.onset), np.array(grp_idxs[self.GRP].onset)))[0]
        stim_data.loc[idx[0]]
        display(stim_data)
        
        """
            2. Segment EEG data into 3s chunk
        """
        labels = stim_data['onset']
        start_dur = labels + 0.0005
        end_dur = labels + 3.0005
        bins = pd.IntervalIndex.from_arrays(start_dur, end_dur)

        #No need to use processed data if signal averaging is done
        #segmented = pd.cut(eegdata.index, bins)
        segmented = pd.cut(self.processedData.index, bins)
        #groups = eegdata.groupby(segmented)
        groups = self.processedData.groupby(segmented)
        display(groups)
        
        """
            3. Calculate ERP (using eeg segments and available groups)
            
            TODO: Ignore first few stimulation
        """
        unique_keys = np.array(list(groups.groups.keys())) #pd.unique(groups.keys)
        #unique_keys = unique_keys[1:NUM_STIM] #Ignoring the NaN group
        print("No. of eeg data segments: ", len(unique_keys))
        ERP_KEYS = unique_keys[idx]

        print(ERP_KEYS[0])
        #TODO: fix 1106.186437 timestamp
        self.segmented_data = groups.get_group(ERP_KEYS[0]) #3s chuck of data, 1500 samples
        display(self.segmented_data)
        
        #Adding only for one timestamp
        #self.segmented_data = self.segmented_data[1:,:]
        #s2 = groups.get_group(unique_keys[1])
        #dictionary = dict(zip(cur_df.index, self.segmented_data.index))
        #add_df = self.segmented_data.rename(index=dictionary)
        #add_df = add_df.add(self.segmented_data)

        idx_to_plot = self.segmented_data.keys
        ERP_KEYS = unique_keys[idx]
        self.ERP = pd.DataFrame()
        ref_df = self.segmented_data
        for key in ERP_KEYS:
            #Panda dataframe
            cur_df = groups.get_group(key)
            dictionary = dict(zip(cur_df.index, ref_df.index))
            cur_df = cur_df.rename(index=dictionary)
            self.ERP = self.ERP.add(cur_df, fill_value=0)

        self.ERP = self.ERP / len(ERP_KEYS)
        display(self.ERP)
    
    #def scaled_eeg_data(self):
        
    def display_groups(self):
        """
        Variables containing data: self.eegdata, self.processedData, self.GRP
        """
        #TODO: Align to onset of stimulus
        #313.17998	313.18038
        #313.17998 + 0.002 = 313.18198
        #Response - 313.185
        #1684.88338 + 0.0005 = 1684.88388
        #343,biphasic,30,1684.88338,1684.88378,0.0004,0
        
        stim_data = pd.read_csv(DataUtils.STIM_DATA)
        df = stim_data.groupby(['sweep', 'stim_type', 'parameter'])
        print(df.count())
        stim_start = 311.17998
        stim_end = 5279.83147

        eeg_data_1 = self.eegdata[self.eegdata.index < stim_start] #Before start of stimulus
        eeg_data_ = self.eegdata[self.eegdata.index >= stim_start] 
        eeg_data_2 = eeg_data_[eeg_data_.index <= stim_end] #Contains multiple stimulations
        eeg_data_3 = self.eegdata[self.eegdata.index > stim_end] #After End of stimulus

        test =  self.eegdata[self.eegdata.index > 1684.88388]
        test = test[test.index <= 1687.88388]
        print(len(test))
        
    def init_inter_region_connectivity(self):
        file = open(DataUtils.INTER_REGION_CONN_FILE)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        #Connectivity matrix normalized between 0 and 1
        self.LABELS = rows[0,:]
        self.INTR_RGN_CONN = (rows[1:,:]).astype('float')
        self.N_REGIONS = len(self.INTR_RGN_CONN)
        
        file = open(DataUtils.INTER_REGION_PVAL_FILE)
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)
        rows = np.array(rows);
        file.close()
        
        self.P_VAL = rows.astype('float')
        
    def show_inter_region_connectivity(self):
        f = plt.figure()
        #f.set_figwidth(8)
        #f.set_figheight(8)
        print(self.P_VAL.shape, self.INTR_RGN_CONN.shape)
        plt.imshow(self.INTR_RGN_CONN, alpha=(1-self.P_VAL))
        #plt.xticks(np.linspace(1, self.N_REGIONS, self.N_REGIONS), self.LABELS, rotation=90)
        #plt.yticks(np.linspace(1, self.N_REGIONS, self.N_REGIONS), self.LABELS)
        plt.title("Inter-region connectivity matrix")
        #plt.colorbar()
        plt.show()
    

"""
so_many_mouse = glob(datastore+"/mouse*")

for mouse in so_many_mouse:
    sessions = [path.basename(f) for f in glob(path.join(mouse, '*'))]
    print(sessions)
    
    for expt in sessions:
        if expt != 'histology':
            rec_folder = path.join(mouse, expt, 'experiment1/recording1')
            print(rec_folder)
            exp = EEGexp(rec_folder, preprocess=False, make_stim_csv=False)
            print(exp.ephys_params['EEG'])
"""