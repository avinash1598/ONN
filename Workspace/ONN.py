"""
TIP: 
    1. Never use  [[0]*3]*3 for array assignment - This creates three duplicate list. Check out below code to view the weird result it gives:
        X = [[2]*3]*2
        X[0][0] = 3
        print(X)
        #Extremely weired!!! This is assigning to all the coloumn 0th element to 3.
        
        TODO:
        Reconstruct between -1 and 1
        
        TODO:
        1. Analyze relationship between phi, theta, frequency of reconstructed and test signal (First check with fewer number of oscillator per node)
        1. Fourier spectrum of error signal
        2. Initialization scheme to get better reconstruction
        3. Do we need many datapoints to fix this? Fine for test signal but what about EEG which is non-stationary?
        4. How is the convergence affected with different parameters?
        5. Does the phi intialization affects quality of convergence(self.PHI = 2*pi*np.ones([self.N, 1]))
        6. Will connectivity make a difference
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import *
from sklearn import preprocessing
import networkx as nx
np.seterr(all='raise')
from deprecated import deprecated
np.set_printoptions(suppress=True)
import DataUtils as dutl

class ONN:
    
    #Constants
    DEBUG = 1
    MU = 0.01                  # Oscillator parameter (>0 for supercritical)
    EPS = 0.1 #2#, 0.5        # ε - coupling strength for error signal e(t) = D(t) - P(t)
    ETA_OMEGA = 0.1#0.1     #0.1 ηω - ωi' = -ηω*e(t)*sin(Фi) ; 1/η = 0.0001
    ETA_ALPHA = 0.05         #0.0001 0.0001 ηα - αi' = ηα*e(t)*ri*cos(Фi)
    TAU = 10000#10000#10000       #10000 τ - τ*Wij' = -Wij + zi(zj*)^(ωi/ωj)
    TAU_J = 10000
    TAU_THETA = 10000 #1000
    #How to deal with non-stationarity
    FREQ_BANDS = np.array([[1, 3], [4, 7], [8, 12], [13, 30]]) #, [8, 12], [13, 30]
    #FREQ_BANDS = np.array([[8, 12], [13, 30]])
    #[1, 3], [4, 7], [8, 12], [13, 18], [19, 24], [25, 30]
    #np.array([[1, 3], [4, 7], [8, 12], [13, 18], [19, 24], [25, 30]]) #[13, 30], [30, 100]
    #More granular frequency band
    #np.array([[1, 3], [4, 7], [8, 12], [13, 30], [30, 100]])

    def __init__(self, N=30, NEPOCHS=100, DT=0.01, fc=30):
        self.N_NODES = N
        self.NEPOCHS = NEPOCHS
        self.DT = DT
        self.fc = fc
        
        self.initialize()
    
    def initialize(self):
        """
        Parameters:
        R:     Initialize everything to 1
        PHI:   2*pi*rand(n, 1) - Uniformly distributed random number from 0 - 2ℼ 
        W:     { 0.01 + 45*2*pi*rand(n,1) - Given} For now assign in steps of 5
        A:     Randomly assign betwwn 0-1 uniformly distributed (TODO: constrain this later)
        THETA: Randomly assign between 0 - 2ℼ uniformly distributed
        APLHA: Initialize all to 0.09

        Constrain:
        θ12 = -θ21 = θ
        A12 = A21 = A
        MU = 1
        
        Use coloumn vector to avoid ambiguity
        
        #Initialized at startup
        R = np.zeros([N, 1])          # R vector in polar coordinates for N oscillators
        PHI = np.zeros([N, 1])        # Ф vector for N oscillators
        OMEGA = np.zeros([N, 1])      # Natural frequencies of N oscillators
        A = np.zeros([N, N])          # NxN connectivity matrix for N oscillators
        THETA = np.zeros([N, N])      # NxN phase angle difference in complex coupling
        ALPHA = np.zeros([N, 1])      # NxN weight matrix for phase 1 teaching
        IEXT = np.zeros([N, 3])       # For each oscillator external input freq, phase and amplitude
        """
        mb_atlas = dutl.DataUtils()
        self.A = mb_atlas.INTR_RGN_CONN
        np.fill_diagonal(self.A, 0) #Few regions have connection to themselves! What does this mean?
        self.N_NODES = mb_atlas.N_REGIONS
        
        #Dummy connectivity matrix
        """
            (i) Aij = Aji (ii) Aii = 0
            (ii) θ12 = -θ21 = θ
            Note: Make sure Aii is always zero to avoid numerical issues
            All oscillator connected to all other oscillator
        """
        # self.A = np.random.rand(self.N_NODES, 1)
        # self.A = np.dot(self.A, self.A.T)
        # self.A.fill(0.0001)
        # #Larger the 'A' value more is the spatial information contained in the signal
        # np.fill_diagonal(self.A, 0)
        # #print(self.A)
        #Don't comment is during testing spatial constrain
        #self.A = np.zeros([self.N_NODES, self.N_NODES])
        
        self.N_SUB = int(np.floor(np.count_nonzero(ONN.FREQ_BANDS <= self.fc)/2))
        self.N = self.N_NODES*self.N_SUB
        
        self.J = np.zeros([self.N, self.N])
        self.J[:] = self.get_J_matrix(self.A, mb_atlas.P_VAL)
        self.bin_conn = np.zeros([self.N, self.N])
        self.bin_conn[self.J > 0] = 1
        
        """
        For each freq band initialize list containing 
        frequency uniformly distributed from specific band.
        Ust the list to assign frequency to each subnode
        """
        self.OMEGA = np.array([np.random.uniform(ONN.FREQ_BANDS[i, 0], 
                                                 ONN.FREQ_BANDS[i, 1], 
                                                 self.N_NODES) 
                               for i in range(self.N_SUB)])
        self.OMEGA = (self.OMEGA.T).flatten()[:,None]
        
        self.R = 0.8*np.ones([self.N, 1]) 
        #Everything initialized to 2pi to get better reconstruction
        #Does this affect quality of convergence?
        self.PHI = 2*pi*np.random.rand(self.N, 1) 
        #2*pi*np.ones([self.N, 1]) #2*pi*np.random.rand(self.N, 1)  #2*pi*np.ones([self.N, 1])
        
        # θ12 = -θ21 = θ
        self.THETA = 2*pi*np.random.rand(self.N, self.N)
        self.THETA = (self.THETA - self.THETA.T)
        #print("Initial theta \n", self.THETA)
        
        #What if we set alpha according to amplitude from magnitude spectrum of fourier transform
        self.ALPHA = 0.02*np.ones([self.N, 1]) #0.05
    
    def get_J_matrix(self, A, p_val):
        #TODO: Uncomment when using inter-region connectivity
        self.P_VAL = np.zeros([self.N, self.N])
        
#         #TODO: Replace with inter-region connectivity matrix
#         """
#             (i) Aij = Aji (ii) Aii = 0
#             (ii) θ12 = -θ21 = θ
#             Note: Make sure Aii is always zero to avoid numerical issues
#             Set Aij to for which (OMEGA[i] - OMEGA[j])<5
#         """
#         self.A = np.random.rand(self.N_NODES, 1)
#         self.A = np.dot(self.A, self.A.T)
#         self.A.fill(0.01)
#         np.fill_diagonal(self.A, 0)
        
        #Add all possible nodes here and later add edge
        G = nx.Graph()
        for i in range(self.N_NODES):
            for j in range(self.N_SUB):
                G.add_node(i*self.N_SUB + j)
        
        #Modify connections(edges) as per varying architecture
        for i in range(self.N_NODES):
            for j in range(self.N_NODES):
                """
                Equally distribute weight between all pair of connections
                Total no of connections are 'N_SUB' between each pair 
                in current scenario
                """
                wt = A[i,j]/self.N_SUB
                for k in range(self.N_SUB):
                    if A[i,j] > 0: G.add_edge(i*self.N_SUB + k, 
                                              j*self.N_SUB + k, 
                                              weight=wt)
                    #TODO: Uncomment when using inter-region connectivity
                    self.P_VAL[i*self.N_SUB + k, j*self.N_SUB + k] = p_val[i,j]
        
        #f = plt.figure()
        #nx.draw(G, pos=nx.circular_layout(G), with_labels = True)
        J = np.array(nx.adjacency_matrix(G).todense())
        #print("p-val shape: ", self.P_VAL.shape)
        return J
        
    def generate_samples(self, NSAMPLES, dt, R, PHI, OMEGA, A, 
                         THETA, ALPHA_Test, IEXT=[], natural_activity = False):
        samples = []
        time = np.linspace(0, (NSAMPLES-1)*dt, NSAMPLES)
        
        R_ = R
        PHI_ = PHI
        
        for idx in range(NSAMPLES):
            
            NORM_FREQ = np.multiply(OMEGA, 1/OMEGA.T) 
            NORM_FREQ[np.isnan(NORM_FREQ)] = 0
            
            NORM_PHASE = (PHI_/OMEGA).T - (PHI_/OMEGA) + THETA/(OMEGA*OMEGA.T)
            PHASE_DIFF = (PHI_/OMEGA) - (PHI_/OMEGA).T - THETA/(OMEGA*OMEGA.T) 
            
            R_POW_NF = np.power(R_.T, NORM_FREQ)
            ARG = OMEGA*NORM_PHASE
            ARG2 = OMEGA*PHASE_DIFF
            R_COUPLED = A*R_POW_NF*np.cos(ARG)
            PHI_COUPLED = A*(R_POW_NF/R_)*np.sin(ARG)
            R_COUPLED = np.sum(R_COUPLED, axis=1) 
            PHI_COUPLED = np.sum(PHI_COUPLED, axis=1)
            R_COUPLED = R_COUPLED[:,None]
            PHI_COUPLED = PHI_COUPLED[:,None]
            
            TRAINABLE_A_TEST = R_*R_POW_NF*np.cos(ARG2)
            
            dR = ( (ONN.MU - R_**2)*R_ + R_COUPLED)*dt
            dPHI = ( OMEGA + PHI_COUPLED)*dt
            #J_temp = A
            #J_temp[J_temp==0] = 1 #Filling zeros with 1 to avoid division by zero error
            #dTHETA = self.bin_conn*( OMEGA.T*R_*R_POW_NF*np.sin(OMEGA*PHASE_DIFF)/J_temp )*dt
            
            R_ = R_ + dR
            PHI_ = PHI_ + dPHI
            #THETA = THETA + dTHETA/ONN.TAU 
            
            #TODO: multiply it be R instead
            #samples_ncpl.append(np.angle(Z_CMPLX_NCPL))
            #Contains the coupling information
            samples.append(R_*np.cos(PHI_))
        
        samples = np.array(samples).T
        samples = np.sum(np.sum(samples, axis=0), axis=0)
        
        return samples[:,None]#, signal
    
    def generate_test_samples(self, NSAMPLES=10000, sampling_rate=1000):
        """
            Test signal (to be fed as external input to the oscillator)
            parameters.
            OMEGA_Test : Frequency of components
            PHI_Test: Phase offset of indivoidual components
            Amplitude is defined by MU (in this case fixed by class constant)
        """
        self.A_Test = np.random.rand(self.N_NODES, 1)
        self.A_Test = np.dot(self.A_Test, self.A_Test.T)
        self.A_Test.fill(0.00001)
        np.fill_diagonal(self.A_Test, 0)
        #Custom connectivity matrix for N=3
        self.A_Test = np.zeros([self.N_NODES, self.N_NODES])
        val = 0.00001 #2 #0.01
        self.A_Test[0,1] = val
        self.A_Test[0,2] = val
        self.A_Test[1,0] = val
        self.A_Test[2,0] = val
        print("A Matrix 1: ", self.A_Test)
        
        dt = 1/sampling_rate
        self.OMEGA_Test = np.array([np.random.uniform(ONN.FREQ_BANDS[i, 0], 
                                                 ONN.FREQ_BANDS[i, 1], 
                                                 self.N_NODES) 
                               for i in range(self.N_SUB)])
        self.OMEGA_Test = (self.OMEGA_Test.T).flatten()[:,None]
        self.PHI_Test = 2*pi*np.random.rand(self.N, 1)
        self.R_Test = 1*np.ones([self.N, 1])
        self.J_Test = np.zeros([self.N, self.N])
        self.J_Test[:] = self.get_J_matrix(self.A_Test)
        self.THETA_Test = self.OMEGA_Test.T*self.PHI_Test - self.OMEGA_Test*self.PHI_Test.T
        self.ALPHA_Test = 1*np.ones([self.N, 1])
        
        return self.generate_samples(NSAMPLES, dt, self.R_Test, self.PHI_Test, 
                            self.OMEGA_Test, self.J_Test, self.THETA_Test, self.ALPHA_Test)
        
    def fit(self, data, sampling_rate=500):
        """
        Parameters:
        NEPOCHS: How long to run the model

        TODO: Later add Iext as well

        Return:
        W: Change in W
        PHI: Change in pahse
        N_PHI_DIFF: Normalized pahse difference

        Miscellaneous:
            1. Normalized pahse difference approaches zero ss individual oscillators attain equillibrium 
            2. No dynamics for Amplitude of lateral coupling i.e. dA = 0
        """
        self.sampling_rate = sampling_rate
        #self.DT = 1.0/self.sampling_rate
        #Reduce DT to increase convergence time
        self.DT = 0.001#0.001 
        self.NSAMPLES = len(data)
        
        self.progress = 0
        
        # Be careful while changing these assignment!! Check out TIP section
        self.OMEGA_ = np.zeros([self.N, self.NSAMPLES+1])
        self.PHI_ = np.zeros([self.N, self.NSAMPLES+1])
        self.R_ = np.zeros([self.N, self.NSAMPLES+1])
        self.SIG_ = np.zeros(self.NSAMPLES)
        self.ERR_ = np.zeros(self.NSAMPLES)
        self.ALPHA_ = np.zeros([self.N, self.NSAMPLES+1])
        self.THETA_ = np.zeros([self.N, self.N, self.NSAMPLES+1])
        self.NORM_PHASE_DIFF_ = np.zeros([self.N, self.N, self.NSAMPLES])
        self.J_ = np.zeros([self.N, self.N, self.NSAMPLES])
        
        #Capturing the discrepancy in R due to coupling
        self.dR_ = np.zeros([self.N, self.NSAMPLES])
        self.dPHI_ = np.zeros([self.N, self.NSAMPLES])
        self.dOMEGA_ = np.zeros([self.N, self.NSAMPLES])
        self.dALPHA_ = np.zeros([self.N, self.NSAMPLES])
        self.dW_ = np.zeros([self.N, self.N, self.NSAMPLES])
        self.dTHETA_ = np.zeros([self.N, self.N, self.NSAMPLES])
        
        self.OMEGA_[:,0] = self.OMEGA.T
        self.ALPHA_[:,0] = self.ALPHA.T
        self.THETA_[:,:,0] = self.THETA
        self.PHI_[:,0] = self.PHI.T
        self.R_[:,0] = self.R.T
        
        #These three are pairs. Don't remove them
        #self.J[:] = self.J_Test#*0.01
        #self.bin_conn = np.zeros([self.N, self.N])
        #self.bin_conn[self.J > 0] = 1
        #self.OMEGA[:] = self.OMEGA_Test
        #self.THETA[:] = self.THETA_Test
        
        Reconstructed = np.zeros(self.NSAMPLES) 
        ERR = np.zeros(self.NSAMPLES)
        PTEACH = np.zeros(self.NSAMPLES)
        
        counter = 0
        for S in range(self.NSAMPLES):
            
            """
            !!!
            Careful while using direct assignment with arrays i.e.
            A = B. A and B will reference same memory if A is not initialized already.
            """
            
            PTEACH[S] = data[S]
            Reconstructed[S] = np.sum(self.ALPHA*np.cos(self.PHI)) #np.sum(self.R*np.cos(self.PHI))
            ERR[S] = PTEACH[S] - Reconstructed[S]

            self.progress = 100*(S+1)/(self.NSAMPLES)

            """
                1. Vector form - do simultaneous update 
                2. Below system handles complex sinusiodal input
                3. In all matrix rows contain source oscillator, and 
                   columns represent coupling to corresponding target oscillator

                Caution:
                Use coloumn vector for each oscillator to avoid confusion
            """

            NORM_FREQ = np.multiply(self.OMEGA, 1/self.OMEGA.T) 
            NORM_FREQ[np.isnan(NORM_FREQ)] = 0

            #Nij = ∅j/ωj - ∅i/ωi + θij/ωiωj
            NORM_PHASE = (self.PHI/self.OMEGA).T - (self.PHI/self.OMEGA) + self.THETA/(self.OMEGA*self.OMEGA.T) 

            #To be used for training lateral weights Nij 
            # ∅i/ωi - ∅j/ωj - θij/ωiωj
            PHASE_DIFF = (self.PHI/self.OMEGA) - (self.PHI/self.OMEGA).T - self.THETA/(self.OMEGA*self.OMEGA.T)

            """
                Causes of NaN
                1. Negative entries in R matrix. Any negative entry in R results in NaN i.e. complex value
                Replacing with 'self.R.T' to avoid numerical issues
            """
            R_POW_NF = np.power(self.R.T, NORM_FREQ) #self.R.T

            ARG = self.OMEGA*NORM_PHASE
            ARG_2 = self.OMEGA*PHASE_DIFF
            R_COUPLED = self.J*R_POW_NF*np.cos(ARG)
            PHI_COUPLED = self.J*(R_POW_NF/self.R)*np.sin(ARG)
            R_COUPLED = np.sum(R_COUPLED, axis=1) 
            PHI_COUPLED = np.sum(PHI_COUPLED, axis=1)
            #Converting to coloumn vector
            R_COUPLED = R_COUPLED[:,None]
            PHI_COUPLED = PHI_COUPLED[:,None]

            """
            When error signal is in real form.
            """
            dR = ( (ONN.MU - self.R**2)*self.R + R_COUPLED + ONN.EPS*ERR[S]*np.cos(self.PHI) )*self.DT 
            dPHI = ( self.OMEGA + PHI_COUPLED - ONN.EPS*(ERR[S]/self.R)*np.sin(self.PHI) )*self.DT 
            
            J_temp = np.zeros([self.N, self.N])
            J_temp[:] = self.J
            J_temp[J_temp==0] = 1 #Filling zeros with 1 to avoid division by zero error
            dTHETA = self.bin_conn*( 
                self.OMEGA.T*self.R*R_POW_NF*np.sin(self.OMEGA*PHASE_DIFF)/J_temp )#*self.DT
            
            #dJ = ( -self.J + self.R*R_POW_NF*np.cos(self.OMEGA*PHASE_DIFF) )*self.DT
            
            """
            Equation set 2: OMEGA and ALPHA update
            """
            dOMEGA = -ONN.ETA_OMEGA*ERR[S]*np.sin(self.PHI)*self.DT
            dALPHA = ONN.ETA_ALPHA*ERR[S]*self.R*np.cos(self.PHI)*self.DT
            
            self.R = self.R + dR
            self.PHI = self.PHI + dPHI
            #self.OMEGA = self.OMEGA + dOMEGA
            self.ALPHA = self.ALPHA + dALPHA
            self.THETA = self.THETA + dTHETA/ONN.TAU
            
            #self.OMEGA[:] = self.OMEGA_Test #Fixing omega
            #self.THETA[:] = self.THETA_Test #Fixing THETA

            """
                Corrections:
                1. Non-negative oscillator frequencies
                2. Entries in R mtrix should not be negative
            """
            #Restrict frequencies of oscillator to positive range only
            self.OMEGA[self.OMEGA < 0.01] = 0.01 
            #This avoids NaN restricting R to non-negative values
            self.R[self.R < 0.01] = 0.01
            #This avoids NaN due to 'A' term in the denominator in dTHETA
            np.fill_diagonal(self.THETA, 0)
            #Negative value of amplitudes not allowed
            self.ALPHA[self.ALPHA < 0.01] = 0.01
            #Restrict theta (resulting from 1e-300 value of theta 
            #returned by function np.angle) to avoid underflow error
            #Might need tro handle the negative case as well
            #self.THETA[self.THETA < 1e-200] = 1e-20
            #self.THETA = np.around(self.THETA, decimals=2)
            #Restrict value of R to avoid overflow
            self.R[self.R>100] = 100 #Gives weird frequency convergence
            
            #Consider sum of squares error
            self.OMEGA_[:,S+1] = self.OMEGA.T
            self.SIG_ = Reconstructed.flatten()
            self.ALPHA_[:,S+1] = self.ALPHA.T
            self.THETA_[:,:,S+1] = self.THETA
            self.PHI_[:,S+1] = self.PHI.T
            self.R_[:,S+1] = self.R.T
            self.ERR_[S] = ERR[S]
           
            #Value corresponding to last sample is important
            #Wrong place for assignment, Change later.
            self.NORM_PHASE_DIFF_[:,:,S] = NORM_PHASE
            self.dR_[:,S] = dR.T #self.R.T - self.R_[:,S]
            self.dPHI_[:,S] = dPHI.T #self.PHI.T - self.PHI_[:,S]
            self.dOMEGA_[:,S] = dOMEGA.T
            self.dALPHA_[:,S] = dALPHA.T
            self.dW_[:,:,S] = np.sin(self.OMEGA*(NORM_PHASE)) #This term should be zero
            self.dTHETA_[:,:,S] = dTHETA
            
            if counter%10000==0:
                print("End of epoch {}". format(S))
            counter = counter + 1
        
    def Test_Signal(self):
        self.sampling_rate = 500
        signal = 0
        NSAMPLES = 100000
        dt = 1.0/self.sampling_rate

        time = np.linspace(0, (NSAMPLES-1)*dt, NSAMPLES)

        Amplitude = np.random.rand(self.N)[:,None]
        Freq = np.array([1 + i*1 for i in range(self.N)])[:,None]
        Phase = np.random.uniform(low=-pi, high=pi, size=(self.N, 1)) 
        
        self.testAmpl = Amplitude
        #print("Amplitude: ", Amplitude)
        #print("Frequency: ", Freq)
        """
            !!!Note!!!
            Don't multiply by 2pi. Here Freq represents angular freq or omega
        """
        signal = np.sum(Amplitude*np.sin( Freq*time + Phase), axis=0)
        scalar = preprocessing.MinMaxScaler(feature_range=(-2, 2))
        scaled_data = scalar.fit_transform(signal[:,None])
        scaled_data = scaled_data.flatten() #Important to avoid fft error
        
        self.signal = signal
        self.scaled_data = scaled_data
        
        return scaled_data#scaled_data#signal
    
    def find_fft(self, SIGNAL, SAMPLING_RATE, is_eeg=False):
        #(2*pi*f0*n/fs
        #f0 = fs*n0/N
        X = np.fft.fft(SIGNAL)
        N = len(X)
        n = np.arange(int(N/2))
        T = N/SAMPLING_RATE
        freq = n/T
        # Get the one-sided specturm
        n_oneside = N//2
        #TODO: Remove 2*pi for EEG signal
        #f_oneside = freq[:n_oneside]
        if is_eeg==True:
            f_oneside = freq[:n_oneside]
        else:
            #Conver frequency to angular frequency component
            f_oneside = 2*np.pi*freq[:n_oneside]
        
        return np.abs(X[:n_oneside]), f_oneside