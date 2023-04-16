"""
TIP: 
    1. Never use  [[0]*3]*3 for array assignment - This creates three duplicate list. Check out below code to view the weird result it gives:
        X = [[2]*3]*2
        X[0][0] = 3
        print(X)
        #Extremely weired!!! This is assigning to all the coloumn 0th element to 3.
        
        TODO:
        Reconstruct between -1 and 1

------------------
Model description:
------------------

(∅_i (t)) ̇= ω_i (t)+ ∑_(j=1)^N〖A_ij sin⁡((∅_j (t))/ω_j - (∅_i (t))/ω_i )〗
Minimize: ∑_(t=1)^M〖( s(t)- ∑_(i=1)^N〖α_i  sin⁡(∅_i (t)) 〗  )〗^2  

-------
Points:
-------

• For oscillator to be in sync normalized phase difference has to zero
• A_ij correlates with oscillators synchronizability. Higher the value of A, more is the tendency of the oscillators to synchronize.
• Oscillators are already in sync suggest that normalized phase difference between them is zero
• Choice of intial condition of omega and phi is important
• Bounds of search variables should be set according to the biological data to get better reconstruction

"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import *
from sklearn import preprocessing
import networkx as nx
import DataUtils as dutl
from scipy.optimize import minimize, basinhopping

class ONN:
    
    #Constants
    FREQ_BANDS = np.array([[1, 3], [4, 7], [8, 12], [13, 30]]) #, [8, 12], [13, 30]
    MU = 1
    NUM_CHANNELS = 3
    
    def __init__(self, N=30, NEPOCHS=100, DT=0.01):
        self.N = N
        self.NEPOCHS = NEPOCHS
        self.DT = DT
        
        self.initialize()
    
    def initialize(self):
        """
        """
        # Create a random NxN adjacency matrix
        self.A = np.random.rand(self.N, self.N)
        self.A = (self.A + self.A.T) / 2 # Make it symmetric
        self.A = np.where(self.A > 0.5, 1.0, 0.0) # Threshold to get binary matrix
        np.fill_diagonal(self.A, 0)
        
        #Uniformly pick from some frequency band
        self.OMEGA = np.random.uniform(10, 20, [self.N, 1]) 
        self.OMEGA = (self.OMEGA.T).flatten()[:,None]
        
        #Initialize everythingt to zero to have same starting phase
        self.PHI = 2*np.pi*np.random.rand(self.N, 1) 
        self.R = np.ones([self.N, 1])
        
        #Later set this weights according to volume condution model
        self.ALPHA = np.zeros([ONN.NUM_CHANNELS, self.N])
        self.ALPHA[:] = self.SampleFFWeights()
    
    def SampleFFWeights(self):
        means = np.random.uniform(0, self.N, ONN.NUM_CHANNELS) 
        means = np.sort(means)
        sigma = self.N*0.2
        
        x = np.random.uniform(0, self.N, [ONN.NUM_CHANNELS, self.N]) 
        x = np.sort(x)
        samples = self.BellCurveFn(x, means, sigma)
        return np.around(samples, 2)
    
    def BellCurveFn(self, x, mean, sigma):
        mean = mean[:,None]
        return np.exp(-((x - mean)/sigma)**2/2) / np.sqrt(2*np.pi)
    
    def generate_samples(self, NSAMPLES, dt, R, PHI, OMEGA, A, ALPHA):
        samples = []
        self.phase_diff = []
        PHI_ = PHI
        R_ = R
        
        for idx in range(NSAMPLES):
            
            NORM_PHASE = (PHI_/OMEGA).T - (PHI_/OMEGA)
            PHI_COUPLED = A*np.sin(OMEGA*NORM_PHASE)
            PHI_COUPLED = np.sum(PHI_COUPLED, axis=1)
            PHI_COUPLED = PHI_COUPLED[:,None]
            R_COUPLED = A*np.cos(OMEGA*NORM_PHASE)
            R_COUPLED = np.sum(R_COUPLED, axis=1)
            R_COUPLED = R_COUPLED[:,None]
            
            dPHI = ( OMEGA + PHI_COUPLED)*dt
            dR = ( (ONN.MU - R_**2)*R_ + R_COUPLED)*dt
            PHI_ = PHI_ + dPHI
            R_ = R_ + dR
            
            self.phase_diff.append(NORM_PHASE)
            data = np.sum(ALPHA.T*R_*np.cos(PHI_), axis=0)
            samples.append(data)
        
        self.phase_diff = np.array(self.phase_diff)
        return np.array(samples).T #Num channels X Num data points
    
    def generate_test_samples(self, NSAMPLES=10000, sampling_rate=1000):
        """
            Test signal (to be fed as external input to the oscillator)
            parameters.
            OMEGA_Test : Frequency of components
            PHI_Test: Phase offset of indivoidual components
            Amplitude is defined by MU (in this case fixed by class constant)
        """
        dt = 1/sampling_rate
        self.NSAMPLES = NSAMPLES
        self.SAMPLING_RATE = sampling_rate
        
        # Create a random NxN adjacency matrix
        self.A_Test = np.random.rand(self.N, self.N)
        self.A_Test = (self.A_Test + self.A_Test.T) / 2 # Make it symmetric
        self.A_Test = np.where(self.A_Test > 0.5, 1, 0) # Threshold to get binary matrix
        np.fill_diagonal(self.A_Test, 0)
        
        #Uniformly pick from some frequency band
        self.OMEGA_Test = np.random.uniform(10, 20, [self.N, 1]) 
        #self.OMEGA_Test = (self.OMEGA_Test.T).flatten()[:,None]
        self.R_Test = np.ones([self.N, 1])
        
        #Initialize everythingt o zero to have same starting phase
        #Phase difference is important for non-zero weight
        self.PHI_Test = 2*np.pi*np.random.rand(self.N, 1)
        
        #Later set this weights according to volume condution model
        self.ALPHA_Test = np.zeros([ONN.NUM_CHANNELS, self.N])
        self.ALPHA_Test[:] = self.SampleFFWeights()
        
        return self.generate_samples(NSAMPLES, dt, self.R_Test, self.PHI_Test, 
                            self.OMEGA_Test, self.A_Test, self.ALPHA_Test)
        
        
    def fit(self, data):
        self.TestData = data
        
        #TODO: Apply symmetry constrain as well.
        
        #weights = self.A.ravel()  # convert to 1D array
        weights = self.A[np.tril_indices(self.N, k=-1)]  # convert lower triangular part to 1D array
        
        # create a mask for non-zero values in the adjacency matrix
        mask = weights != 0
        x0 = weights[mask].astype(float)  # set initial guess for x as the non-zero weights
        bounds = [(0.1, 15)] * len(x0)  # bounds for each element of x 
        options = {'maxiter': self.NEPOCHS}  # maximum number of iterations
        
        print(x0)
        self.ERR = []
        #Run SPSA optimizer
        result = basinhopping(self.objective, x0, niter=self.NEPOCHS, minimizer_kwargs={"method":"SLSQP", "bounds":bounds}, callback=self.my_callback)
        
        self.A = self.GetAdjMatrix(result.x)
        #weights[mask] = result.x
        #self.A = weights.reshape(self.N, self.N)
        
        print('Status : %s' % result['message'])
        print('Total Evaluations: %d' % result['nfev'])
        print("Solution found:")
        print("x =", result.x)
        print("f(x) =", result.fun)
        
    def objective(self, updated_weight):
        dt = 1/self.SAMPLING_RATE
        A_updated = self.GetAdjMatrix(updated_weight)
        
        # Gradient calculation for weight can be parallalized instead of 
        # running objective function serially for each weight
        # NUM_CHANNELS X NUM_SAMPLES
        curSignal = self.generate_samples(self.NSAMPLES, dt, self.R, self.PHI, 
                                          self.OMEGA, A_updated, self.ALPHA)
        err = 0
        for i in range(ONN.NUM_CHANNELS):
            err = err + np.sum((self.TestData - curSignal[i,:])**2)
        self.ERR.append(err)
        if (len(self.ERR)%50==0):
            print("Evaluating...")
            print("Current weight snapshot: ", updated_weight)
        return err
    
    def my_callback(self, updated_weight, fun, boolean):
        # What to do when weights get updated?
        #print("Updated weight: ", updated_weight)
        return
    
    def GetAdjMatrix(self, x0):
        """
        This function imposes symmetry condition on adjacency matrix A
        """
        # convert lower triangular part to 1D array
        weights = self.A[np.tril_indices(self.N, k=-1)]  
        # create a mask for non-zero values in the adjacency matrix
        mask = weights != 0
        weights[mask] = x0
        
        # Create a new array with the upper triangular part mirrored
        upper_tri = np.zeros((self.N, self.N))
        upper_tri[np.triu_indices(self.N, k=1)] = weights
        lower_tri = upper_tri.T

        # Set the lower triangular part of the adjacency matrix
        # to be equal to the upper triangular part
        A_updated = np.triu(upper_tri) + np.tril(lower_tri, k=-1)

        return A_updated