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
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import *
from sklearn import preprocessing
import networkx as nx
np.seterr(all='raise')

class ONN:
    
    #Constants
    DEBUG = 1
    MU = 0.0001#0.000001, 0.0001#0.0025                   # Oscillator parameter (>0 for supercritical)
    EPS = 0.9                          # ε - coupling strength for error signal e(t) = D(t) - P(t)
    ETA_OMEGA = 0.1                    #ηω - ωi' = -ηω*e(t)*sin(Фi) ; 1/η = 0.0001
    ETA_ALPHA = 0.01   #0.0001              #0.0001 ηα - αi' = ηα*e(t)*ri*cos(Фi)
    TAU = 10000 #10000                        #τ - τ*Wij' = -Wij + zi(zj*)^(ωi/ωj)
    
    def __init__(self, N=30, NEPOCHS=100, DT=0.01, fc=30):
        self.N = N
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
        self.R = np.ones([self.N, 1]) 
        #self.PHI = 2*pi*np.random.rand(self.N, 1) 
        self.PHI = 2*pi*np.ones([self.N, 1]) #Everything initialized to 2pi to get better reconstruction
        #self.OMEGA = self.N*np.random.rand(self.N, 1)
        self.OMEGA = self.fc*np.random.rand(self.N, 1) #chnage it to self.fc
        #self.OMEGA = np.array([3 + i*8 for i in range(self.N)])[:,None]
        #self.OMEGA[0] = 6
        self.OMEGA = np.sort(self.OMEGA) #Sorting frequencies in increasing order to apply constrain later
        
        """
            (i) Aij = Aji (ii) Aii = 0
            (ii) θ12 = -θ21 = θ
            Note: Make sure Aii is always zero to avoid numerical issues
            Set Aij to for which (OMEGA[i] - OMEGA[j])<5
        """
        self.A = np.random.rand(self.N, 1)
        self.A = np.dot(self.A, self.A.T)
        self.A.fill(0.00001) #0.00001
        np.fill_diagonal(self.A, 0)
        #Couple nearby oscillator's only
        #idx = np.linspace(1, self.N, self.N)[:,None]
        #diff = abs(idx - idx.T)
        #self.A = np.where(diff < 5, 0.1, 0)
        #np.fill_diagonal(self.A, 0)
        #print(self.A)
        self.bin_conn = np.zeros([self.N, self.N])
        self.bin_conn[self.A > 0] = 1
        
        self.THETA = 2*pi*np.random.rand(self.N, self.N)
        self.THETA = (self.THETA - self.THETA.T)
        
        self.ALPHA = 0.1*np.ones([self.N, 1]) #0.05*
    
    def generate_signal(self):
        self.sampling_rate = 500
        NSAMPLES = 10000
        dt = 1.0/self.sampling_rate
        time = np.linspace(0, (NSAMPLES-1)*dt, NSAMPLES)
        print(self.OMEGA)
        #Wrong way to generate signal
        signal = np.sum(self.ALPHA*np.cos(self.OMEGA*time + self.PHI), axis=0)
        print(signal.shape)
        #spatial distribution
        return signal
    
    def oscillate_kr_bhai(self, idx, dt, R, PHI, OMEGA, A, THETA, IEXT):
        Z_CMPLX = R*np.exp(1j*PHI)
                
        Term1 = Z_CMPLX*(ONN.MU + 1j*OMEGA - np.absolute(Z_CMPLX)**2)
        Term2 = np.sum(A*np.exp(1j*( THETA/OMEGA.T ))*np.power(Z_CMPLX.T, OMEGA/OMEGA.T), axis=1)[:,None]
        #Term3 is when oscillator is forced by external input 
        # and is generally present during learning phase
        Term3 = 0
        if len(IEXT) > 0:
            Term3 = ONN.EPS*IEXT[idx]
                
        dZ_CMPLX = (Term1 + Term2 + Term3)*dt
        Z_CMPLX = Z_CMPLX + dZ_CMPLX
        return Z_CMPLX
    
    def generate_samples(self, NSAMPLES=10000, sampling_rate=1000):
        samples = []
        dt = 1/sampling_rate
        
        for idx in range(NSAMPLES):
            samples.append(
                self.oscillate_kr_bhai(idx, dt, self.R, self.PHI, 
                                       self.OMEGA, self.A, self.THETA))
    
    def Preprocess(self, data):
        print("Preprocessing done...")
        return data
    
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
        self.DT = 1.0/self.sampling_rate
        #self.DT = 0.001
        data = self.Preprocess(data)
        self.NSAMPLES = len(data)
        
        self.progress = 0
        
        # Be careful while changing these assignment!! Check out TIP section
        self.OMEGA_ = np.zeros([self.N, self.NEPOCHS+1])
        self.SIG_ = np.zeros([self.NEPOCHS, self.NSAMPLES])
        self.ERR_ = np.zeros(self.NEPOCHS)
        self.ALPHA_ = np.zeros([self.N, self.NEPOCHS])
        self.NORM_PHASE_DIFF_ = np.zeros([self.N, self.N, self.NEPOCHS])
        
        self.OMEGA_[:,0] = self.OMEGA.T
        
        for t in range(self.NEPOCHS):
            # Why are they initializing R and THETA in every epoch?
            #self.R = 1*np.ones([self.N, 1]) 
            #self.PHI = 2*pi*np.random.rand(self.N, 1)
            
            Reconstructed = np.zeros(self.NSAMPLES) 
            ERR = np.zeros(self.NSAMPLES)
            PTEACH = np.zeros(self.NSAMPLES)
            
            for S in range(self.NSAMPLES):
                
                PTEACH[S] = data[S]
                Reconstructed[S] = np.sum(self.ALPHA*np.cos(self.PHI))
                ERR[S] = PTEACH[S] - Reconstructed[S]
                
                self.progress = 100*((t+1)*(S+1))/(self.NEPOCHS*self.NSAMPLES)

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
                """
                R_POW_NF = self.R.T
                
                ARG = self.OMEGA*NORM_PHASE
                ARG_2 = self.OMEGA*PHASE_DIFF
                R_COUPLED = self.A*R_POW_NF*np.cos(ARG)
                PHI_COUPLED = self.A*(R_POW_NF/self.R)*np.sin(ARG)
                R_COUPLED = np.sum(R_COUPLED, axis=1) 
                PHI_COUPLED = np.sum(PHI_COUPLED, axis=1)
                #Converting to coloumn vector
                R_COUPLED = R_COUPLED[:,None]
                PHI_COUPLED = PHI_COUPLED[:,None]
                
                # dR = ( (ONN.MU - self.R**2)*self.R + R_COUPLED + ONN.EPS*ERR[S]*np.cos(self.PHI) )*self.DT 
                # dPHI = ( self.OMEGA + PHI_COUPLED - ONN.EPS*(ERR[S]/self.R)*np.sin(self.PHI) )*self.DT 
                
#                 """
#                 TODO: write in complex form
#                 """
                Z_CMPLX = self.R*np.exp(1j*self.PHI)
                
                Term1 = Z_CMPLX*(ONN.MU + 1j*self.OMEGA - np.absolute(Z_CMPLX)**2)
                Term2 = np.sum(self.A*np.exp(1j*( self.THETA/self.OMEGA.T ))*np.power(Z_CMPLX.T, self.OMEGA/self.OMEGA.T), axis=1)[:,None]
                Term3 = ONN.EPS*ERR[S]
                
                dZ_CMPLX = (Term1 + Term2 + Term3)*self.DT
                Z_CMPLX = Z_CMPLX + dZ_CMPLX

#                 #Equation2
                COMPLEX_W = self.A*np.exp(1j*self.THETA/self.OMEGA.T)
                Z1 = self.R*np.exp(1j*self.PHI)
                Z2 = self.R.T*np.exp(-1j*self.PHI.T)
                Z2_SP = np.power(Z2, self.OMEGA/self.OMEGA.T)
                
                dCOMPLEX_W = (-COMPLEX_W + Z1*Z2_SP)
                COMPLEX_W = COMPLEX_W + dCOMPLEX_W/ONN.TAU #ONN.ETA_OMEGA
                
                
                dOMEGA = -ONN.ETA_OMEGA*ERR[S]*np.sin(self.PHI)*self.DT
                dALPHA = ONN.ETA_ALPHA*ERR[S]*self.R*np.cos(self.PHI)*self.DT
                
                J_temp = np.zeros([self.N, self.N])
                J_temp[:] = self.A
                J_temp[J_temp==0] = 1 #Filling zeros with 1 to avoid division by zero error
                dTHETA = self.bin_conn*( 
                    self.OMEGA.T*self.R*R_POW_NF*np.sin(self.OMEGA*PHASE_DIFF)/J_temp )#*self.DT

                # self.R = self.R + dR
                # self.PHI = self.PHI + dPHI
                self.R = np.absolute(Z_CMPLX)
                self.PHI = np.angle(Z_CMPLX)
                #self.OMEGA = self.OMEGA + dOMEGA
                self.ALPHA = self.ALPHA + dALPHA
                self.THETA = np.angle(COMPLEX_W)*self.OMEGA.T
                #self.THETA = self.THETA + dTHETA/ONN.TAU
                
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
                np.fill_diagonal(self.A, 0)
                
                self.NORM_PHASE_DIFF_[:,:,t] = np.sin(NORM_PHASE)
            
            #Consider sum of squares error
            self.ERR_[t] = np.sum(np.abs(ERR))
            self.OMEGA_[:,t+1] = self.OMEGA.T
            self.SIG_[t] = Reconstructed
            self.ALPHA_[:,t] = self.ALPHA.T
            
            print("End of epoch {}". format(t))
        
    def Test_Signal(self):
        self.sampling_rate = 1000
        signal = 0
        NSAMPLES = 500000
        dt = 1.0/self.sampling_rate

        time = np.linspace(0, (NSAMPLES-1)*dt, NSAMPLES)

        Amplitude = np.ones([self.N, 1]) #np.random.rand(self.N)[:,None]
        Freq = np.array([5 + i*5 for i in range(self.N)])[:,None] #i*(self.fc/self.N)
        Phase = np.random.uniform(low=0, high=2*pi, size=(self.N, 1))
        
        print("Frequency in test signal:", Freq)
        
        self.testAmpl = Amplitude
        #print("Amplitude: ", Amplitude)
        #print("Frequency: ", Freq)
        """
            !!!Note!!!
            Don't multiply by 2pi. Here Freq represents angular freq or omega
        """
        signal = np.sum(Amplitude*np.sin( Freq*time + Phase), axis=0)
        scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scalar.fit_transform(signal[:,None])
        scaled_data = scaled_data.flatten() #Important to avoid fft error
        
        self.signal = signal
        self.scaled_data = scaled_data
        
        #Subtract mean to avoid DC component in FFT
        self.scaled_data = scaled_data - np.mean(scaled_data)
        
        return signal#scaled_data#scaled_data#signal