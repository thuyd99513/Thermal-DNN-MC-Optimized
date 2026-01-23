#===============
#Super Module
#===============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '..'
main_dir_local = '..'
def determine_module_filename(mf_index):
    if mf_index == 0:
        module_filename = main_dir +'/py_functions/'
    elif mf_index == 1:
        module_filename = main_dir_local + '/py_functions_local/'
    return module_filename
module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)
#======
#Module
#======
from random import choice
import copy
from functools import wraps
import math
import numpy as np
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
from random import randrange
import scipy as sp
from time import time
from utilities import calc_ener

SAMPLE_START = 1 # SET THIS PARAMETER
SAMPLE_END = 100 # SET THIS PARAMETER
num_cores = 64

num_cores = 64
Bound_tw = 8 # All tw which is smaller than Bound_tw will save seeds for. 
l_list = [1,2,3,4,5,6,7,8]
l_index_list = [0,1,2,3,4,5,6,7]
l_S_list = [1,2,3,4,5,6,7,8,9]
l_S_index_list = [0,1,2,3,4,5,6,7,8]
init_list = [i for i in range(SAMPLE_START,SAMPLE_END+1)] # For M480 tw32, 64, 128, 256, 512, 1024
init_list_all = np.array(range(20*num_cores))
M_list = [30,60,120,240,480,960,1920]
QS_cutoff = 0.02
QJ_cutoff = 0.02

# The parameter ratio is used for generating the sampling time point.
qq = 5.0
pp = qq + 1
ratio= pp/qq

# general ratio 
step_list_v2 = [int((ratio)**n) for n in range(200)]
tw_list_test = [0,1,2,3] # MC steps
  
def generate_tw_list(tot_steps):
    if tot_steps < 4 + 1:
        tw_list = tw_list_test 
    else:
        #tw_list = [1,1,1,1]
        #tw_list = [64,64,64,64]
        #tw_list = [128,128,128,128]
        #tw_list = [256,256,256,256]
        #tw_list = [512,512,512,512]
        tw_list = [1024,1024,1024,1024]
    return tw_list

def timethis(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time()
        result = fun(*args,**kwargs)
        end = time()
        print(fun.__name__, end-start)
        return result
    return wrapper

def generate_traj_sampling_num(num_steps,BIAS,qq):
    try:
        if qq==1:
            sampled_steps = int(np.log2(num_steps + BIAS))
        elif qq > 0 and qq != 1:
            sampled_steps = int(np.log(num_steps + BIAS) / np.log( (qq+1)/qq ))
    except:
        print("The parameter qq should be positive.")
    print("sampled_steps")
    print(sampled_steps)
    return sampled_steps

class Network:
    def __init__(self,sample_index,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp,h,qq=2):
        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only calculate the part affected by the flip of a SPIN (S)  or a shift of
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected. 
           h : index of the host machine.
        """
        # Parameters used in the host machine (No. 0)
        self.init = int(sample_index)
        self.tw = int(tw)
        self.L = int(L)
        self.M = int(M)
        self.N = int(N)
        self.N_in = int(N_in)
        self.N_out = int(N_out)
        self.tot_steps = int(tot_steps)
        self.beta = beta
        self.timestamp = timestamp
        self.num_hidden_node_layers = self.L - 1 # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
        self.num_hidden_bond_layers = self.L - 2
        # Define new parameters; T (technically required, to save memory)
        self.BIAS = 1 # For avoiding x IS NOT EQUAL TO ZERO in log2(x)
        self.BIAS2 = 5 # For obtaing long enough list list_k.
        #self.T = int(np.log2(self.tot_steps+self.BIAS)) # we keep the initial state in the first step
        #self.T = generate_traj_sampling_num(self.tot_steps,self.BIAS,qq)
        self.H = 0 # for storing energy
        self.new_H = 0 # for storing temperay energy when update

        # Energy difference caused by update of sample mu
        self.delta_H = 0

        # Intialize S,J_hidden,J_in and J_out by the saved arrays, which are saved from the host machine. 
        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
        data_dir = '../data'
        str_timestamp = str(timestamp)
        self.S = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,sample_index,tw,L,M,N,beta))
        self.J_hidden = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,sample_index,tw,L,N,beta))
        self.J_in = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(h,L,M,N,sample_index, tw, N, N_in, beta))
        self.J_out = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,sample_index,tw,N_out,N,beta))
        
        #=========================================================================================================
        #The following copy operations must be done in this __init__() function.
        #NOTE that one can NOT delay the COPY operations, for example, just putting it in the function flip_spin(), etc.
        #However, we ALSO NEED the COPY operation in functions flip_spin() and shift_bond_in(), etc.
        #=========================================================================================================
        self.new_S = copy.copy(self.S) # for storing temperay array when update
        self.new_J_hidden = copy.copy(self.J_hidden) # for storing temperay array when update
        self.new_J_in = copy.copy(self.J_in) 
        self.new_J_out = copy.copy(self.J_out) 
      
        # Initialize the inner parameters: num_bonds, num_variables
        self.SQ_N = (self.N) ** 2
        num_variables = self.N * self.M * self.num_hidden_node_layers 
        num_bonds = self.N * self.N_in + self.SQ_N * self.num_hidden_bond_layers + self.N_out * self.N
        num_hidden_bonds = self.SQ_N * self.num_hidden_bond_layers

        self.num_variables = int(num_variables) 
        self.num_bonds = int(num_bonds)
        self.num_hidden_bonds = int(num_hidden_bonds)
        self.num = self.num_variables + self.num_bonds

        self.ind_save = 0
        self.count_MC_step = 0 
        self.T_2 = generate_traj_sampling_num(self.tot_steps * self.num, self.BIAS, qq)
        print("self.T_2")
        print(self.T_2)
        #self.T_2 = int(np.log2(self.tot_steps * self.num + self.BIAS)) # we keep the initial state in the first step
        #======================================================================================================================================== 
        # The arrays for storing MC trajectories of S, J and H (hyperfine)
        # Note that we DO NOT nedd to define arrays self.S_in_traj_hyperfine, or S_out_traj_hyperfine, because self.S_in and self.S_out are fixed.
        self.J_hidden_traj_hyperfine = np.zeros((self.T_2, self.num_hidden_bond_layers, self.N, self.N), dtype='float32') 
        self.S_traj_hyperfine = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
        self.H_hidden_traj_hyperfine = np.zeros(self.T_2, dtype='float32') 
        self.J_in_traj_hyperfine = np.zeros((self.T_2,self.N,self.N_in), dtype='float32')
        self.J_out_traj_hyperfine = np.zeros((self.T_2,self.N_out,self.N), dtype='float32')
        #======================================================================================================================================== 

        # DEFINE SOME PARAMETERS
        self.EPS = 0.000001 
        self.RAT = 0.1 # r: Yoshino2019 Eq(35), RAT=0.1
        self.RESCALE_J = 1.0/np.sqrt(1 + self.RAT**2)
        self.SQRT_N = np.sqrt(self.N)
        self.SQRT_N_IN = np.sqrt(self.N_in)
        self.RAT_in = self.RAT * self.N_in/self.N  # r: Yoshino2019 Eq(35), RAT=0.1, we rescale it for input bond layer.
        self.RAT_out = self.RAT * self.N_out/self.N  # r: Yoshino2019 Eq(35), RAT=0.1, we rescale it for output bond layer.
        self.PROB = self.num_variables/self.num 
        self.cutoff1 = self.N*self.N_in
        self.cutoff2 = self.cutoff1 + self.num_hidden_bond_layers * self.SQ_N

        # EXAMPLE: list_k = [0,1,2,3,4,8,16,64,256,1024,4096,16384,65536] # list_k is used for waiting time.
        temp_set1 = set([2**(i) for i in range(int(np.log2(self.tot_steps+self.BIAS) + self.BIAS2))])
        temp_set2 = set([i for i in range(1,Bound_tw)]) # For small time step, we save seeds for each step (IMPORTANT: tw=0 should be excluded).
        temp_list = list(temp_set1.union(temp_set2))
        temp_list.sort() # Sort the list. This is important.
        # HARD version
        self.list_k_4_hyperfine = [2**(i) for i in range(int(np.log2(self.tot_steps * self.num + self.BIAS) + self.BIAS2))]
        #SIMPLE version
        self.list_k = [0,8,16,32,64,128,256,512,1024] 
        
        #For saving S and J accurately, I need know the exact update steps.
        self.update_index = 0

        # Define J_in and J_out
        self.S_in = np.zeros((self.M,self.N_in))  
        self.S_out = np.zeros((self.M,self.N_out))  
        #==============================
        # For testing
        self.Delta_E_node_hidden = []
        self.Delta_E_node_in = []
        self.Delta_E_node_out = []
        self.Delta_E_bond_hidden = []
        self.Delta_E_bond_in = []
        self.Delta_E_bond_out = []
    def gap_in_init(self):
        '''Ref: Yoshino2019, eqn (31b):'''
        r_in = np.zeros((self.M, self.N),dtype='float32')
        for mu in range(self.M):
            for n2 in range(self.N):
                r_in[mu,n2] = (np.sum(self.J_in[n2,:] * self.S_in[mu,:])/self.SQRT_N_IN) * self.S[mu,0,n2]
        return r_in

    def gap_hidden_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        L = self.L
        M = self.M
        N = self.N
        r_hidden = np.zeros((M,L,N),dtype='float32')
        for mu in range(M):
            for l in range(self.num_hidden_bond_layers): # l = 2,...,L-1
                index_node_layer = l # Distinguish different index for S and r is important!
                for n2 in range(N):
                    r_hidden[mu,l,n2] = (np.sum(self.J_hidden[l,n2,:] * self.S[mu,index_node_layer,:])/self.SQRT_N) * self.S[mu,index_node_layer + 1, n2]
        return r_hidden

    def gap_out_init(self):
        M = self.M
        N = self.N
        N_out = self.N_out
        r_out = np.zeros((M,N_out),dtype='float32')
        for mu in range(M):
            for n2 in range(N_out):
                r_out[mu,n2] = (np.sum(self.J_out[n2,:] * self.S[mu,-1,:])/self.SQRT_N) * self.S_out[mu,n2]
        return r_out

    def flip_spin(self,mu,l,n):
        '''flip_spin() will flip S at a given index tuple (l,mu,n). We add l,mu,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
        # Update self.new_S
        self.new_S = copy.copy(self.S) # for storing temperay array when update
        self.new_S[mu][l][n] = -self.S[mu][l][n]

    def load_S_and_J(self):
        beta = self.beta
        init = self.init
        L = self.L
        M = self.M
        N = self.N
        N_in = self.N_in
        N_out = self.N_out
        tw = self.tw
        self.S_in = np.load('../data/host/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(M,N_in,beta))
        self.S_out = np.load('../data/host/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(M,N_out,beta))
        #=========================================================================================================================================
        # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of the host machine is the same as the guest machines.
        self.J_in = np.load('../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init,tw,N,N_in,beta))
        self.J_out = np.load('../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init,tw,N_out,N,beta))
        self.J_hidden = np.load('../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init,tw,L,N,beta))
        self.S = np.load('../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init,tw,L,M,N,beta))

    def load_S_and_J_2g(self):
        beta = self.beta
        init = self.init
        L = self.L
        M = self.M
        N = self.N
        N_in = self.N_in
        N_out = self.N_out
        tw = self.tw
        h = int(init/num_cores) + 1
        self.S_in = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(h,L,M,N,M,N_in,beta))
        self.S_out = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(h,L,M,N,M,N_out,beta))
        #=========================================================================================================================================
        # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of the host machine is the same as the guest machines.
        self.J_in = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(h,L,M,N,init,tw,N,N_in,beta))
        self.J_out = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,init,tw,N_out,N,beta))
        self.J_hidden = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,init,tw,L,N,beta))
        self.S = np.load('../../host{:d}_ir_hf_L{:d}_M{:d}_N{:d}_mp/data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(h,L,M,N,init,tw,L,M,N,beta))

    def random_update_S(self):
        # Const.s
        EPS = self.EPS
        rand2 = np.random.random(1)

        mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
        self.flip_spin(mu,l,n)
        if l == 0:
            self.decision_node_in_by_mu_n(mu,n,EPS,rand2)
        elif l == self.num_hidden_node_layers - 1:
            self.decision_node_out_by_mu_n(mu,n,EPS,rand2)
        else:
            self.decision_by_mu_l_n(mu,l,n,EPS,rand2)
    def random_update_J(self):
        """Method 1"""
        # Const.s
        EPS = self.EPS
        x = np.random.normal(loc=0,scale=1.0,size=None)
        x = round(x,10)
        cutoff1 = self.cutoff1        
        cutoff2 = self.cutoff2        
        P1 = cutoff1/self.num_bonds
        P2 = cutoff2/self.num_bonds
        rand1, rand2 = np.random.random(1), np.random.random(1)

        if rand1 < P1:
            n2,n1=randrange(self.N),randrange(self.N_in)
            # Update J_in[n2,n1]
            self.shift_bond_in_v2(n2,n1,x) # Note1/2: shift_bond_in_v2() will give normalized result.
            #self.shift_bond_in(n2,n1,x) # Note1/2: shift_bond_in() will give non-normalized result. 
            self.decision_bond_in_by_n2_n1(n2,n1,EPS,rand2)
        elif rand1 > P2:
            n2,n1=randrange(self.N_out),randrange(self.N)
            self.shift_bond_out_v2(n2,n1,x)
            self.decision_bond_out_by_n2_n1(n2,n1,EPS,rand2)
        else:
            l,n2,n1 = randrange(self.num_hidden_bond_layers),randrange(self.N),randrange(self.N) 
            self.shift_bond_hidden_v2(l,n2,n1,x)
            self.decision_bond_hidden_by_l_n2_n1(l,n2,n1,EPS,rand2)
    def set_vars(self):
        """ Define some variables. 
        Ref: He Yujian's book, Fig. 3.2, m-layer network; Yoshino2020, Fig.1. L-layer network.
        We ASSUME that each hidden layer has N neurons.
        """
        self.r_hidden = self.gap_hidden_init() # The initial gap
        self.r_in = self.gap_in_init() # The initial gap
        self.r_out = self.gap_out_init() # The initial gap
        
        self.S_traj_hyperfine[0,:,:,:] = self.S # Note that self.S_traj will independent of self.S from now on. This o.S is the state of S at the end of last epoch of training
        self.J_hidden_traj_hyperfine[0,:,:,:] = self.J_hidden # This o.J is the state of J at the end of last epoch of training
        self.J_in_traj_hyperfine[0,:,:] = self.J_in # The shape of J_in : (N, N_in) 
        self.J_out_traj_hyperfine[0,:,:] = self.J_out # The shape of o.J_out:  (N_out, N) 
        
        self.H_hidden = calc_ener(self.r_hidden) # The energy
        self.H_in = calc_ener(self.r_in) # The energy
        self.H_out = calc_ener(self.r_out) # The energy
        self.H_hidden_traj_hyperfine[1] = self.H_hidden # H_traj[0] will be neglected

    def shift_bond_hidden(self,l,n2,n1,x):
        '''shift_bond_hidden() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming.
           In shift_bond_hidden(), no rescaling (normalization) is done before calculating the energy difference.
           But, rescale_bond_hidden() has to be called after accept.
        '''
        self.new_J_hidden[l][n2][n1] = (self.J_hidden[l][n2][n1] + x * self.RAT) * self.RESCALE_J
    def rescale_bond_hidden(self,l,n2):
        # rescaling 
        N = self.N
        t = self.new_J_hidden[l][n2] 
        N_prim = np.sum(t*t)
        SCALE = np.sqrt(N / N_prim)
        self.new_J_hidden[l][n2] = self.new_J_hidden[l][n2] * SCALE
    def shift_bond_hidden_v2(self,l,n2,n1,x):
        '''shift_bond_hidden() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming.
           shift_bond_hidden_v2() rescals the synaptic weights before calculating the energy differnce.'''
        self.new_J_hidden = copy.copy(self.J_hidden) # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_hidden[l][n2][n1] = (self.J_hidden[l][n2][n1] + x * self.RAT) * self.RESCALE_J
        # rescaling 
        self.rescale_bond_hidden(l,n2)
    def shift_bond_in(self,n2,n1,x):
        '''shift_bond_in() will shift the element of J_in with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           In shift_bond_in(), No normalization is done before calculating the energy difference.
           But, rescale_bond_in() has to be called after accept.
        '''
        self.new_J_in = copy.copy(self.J_in) # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_in[n2,n1] = (self.J_in[n2,n1] + x * self.RAT) * self.RESCALE_J
    def rescale_bond_in(self,n2):
        # rescaling 
        N_in = self.N_in
        t = self.new_J_in[n2]  
        N_prim = np.sum(t*t)
        SCALE = np.sqrt(N_in / N_prim)
        self.new_J_in[n2] = self.new_J_in[n2] * SCALE
    def shift_bond_in_v2(self,n2,n1,x):
        '''shift_bond_in() will shift the element of J_in with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           shift_bond_in_v2() rescals the synaptic weights before calculating the energy differnce.'''
        self.new_J_in = copy.copy(self.J_in) # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_in[n2,n1] = (self.J_in[n2,n1] + x * self.RAT) * self.RESCALE_J
        # step2:rescaling 
        self.rescale_bond_in(n2)

    def shift_bond_out(self,n2,n1,x):
        '''shift_bond_out() will shift the element of J_out with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           In shift_bond_out(), no normalization is done before calculating the energy difference.
           But, rescale_bond_out() has to be called after accept.
        '''
        self.new_J_out = copy.copy(self.J_out) # for storing temperay array when update 
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_out[n2,n1] = (self.J_out[n2,n1] + x * self.RAT) * self.RESCALE_J
    def rescale_bond_out(self,n2):
        # rescaling 
        N = self.N
        t = self.new_J_out[n2] 
        N_prim = np.sum(t*t) # Use sort, because we want to avoid the larger values 'eats' the smaller ones. But I do not need to use sort in np.sum(), I believe.
        SCALE = np.sqrt(N / N_prim)
        self.new_J_out[n2] = self.new_J_out[n2] * SCALE
    def shift_bond_out_v2(self,n2,n1,x):
        '''shift_bond_out() will shift the element of J_out with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           shift_bond_out_v2() rescals the synaptic weights before calculating the energy differnce.
        '''
        self.new_J_out = copy.copy(self.J_out) # for storing temperay array when update 
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_out[n2,n1] = (self.J_out[n2,n1] + x * self.RAT) * self.RESCALE_J
        # rescaling 
        self.rescale_bond_out(n2)

    # The following accept function is used if S is flipped.
    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
    # One of the following accept functions is used if J is shifted.
    def accept_bond_hidden_by_l_n2_n1(self,l,n2):
        self.J_hidden[l,n2] = self.new_J_hidden[l,n2]
        self.H = self.H + self.delta_H
    def accept_bond_in_by_n2_n1(self,n2):
        self.J_in[n2] = self.new_J_in[n2]
        self.H = self.H + self.delta_H
    def accept_bond_out_by_n2_n1(self,n2):
        self.J_out[n2] = self.new_J_out[n2]
        self.H = self.H + self.delta_H
    
    # One of the gap function is used if S is flipped.
    def part_gap_hidden_before_flip(self,mu,l_s,n):
        '''l_s: index for hidden layers of the NODE (hidden S), l_s =  1, 2,...,7 (Totally, (L-3)+2= 7 hidden node layers + 2 layers, if L = 10).
           Ref: Yoshino2019, eqn (31b)
           When S is fliped, only one machine changes its coordinates and it will affect the gap of the node in front of it and the gaps of the N nodes
           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff.
           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the
           Energy change coused by the flip of S^mu_node,n.
        '''
        part_gap = np.zeros(self.N + 1, dtype='float32')
        l_h = l_s - 1
        part_gap[0] = (np.sum(self.J_hidden[l_h,n,:] * self.S[mu,l_s-1,:])/self.SQRT_N) * self.S[mu,l_s,n]
        for n2 in range(self.N):
            part_gap[1+n2] = (np.sum(self.J_hidden[l_h+1,n2,:] * self.S[mu,l_s,:])/self.SQRT_N) * self.S[mu,l_s+1,n2] ##NOTE2023-4-29(6-1):the RIGHT VERSION.
            #part_gap[1+n2] = (np.sum(self.J_hidden[l_s,n2,:] * self.S[mu,l_s,:])/self.SQRT_N) * self.S[mu,l_s+1,n] #NOTE2023-4-29(6-1):WRONG: n should be n2.
        return part_gap 
    def part_gap_hidden_after_flip(self,mu,l_s,n):
        part_gap = np.zeros(self.N + 1,dtype='float32')
        l_h = l_s - 1
        part_gap[0] = (np.sum(self.J_hidden[l_h,n,:] * self.S[mu,l_s-1,:])/self.SQRT_N) * self.new_S[mu,l_s,n]
        for n2 in range(self.N):
            part_gap[1+n2] = (np.sum(self.J_hidden[l_h+1,n2,:] * self.new_S[mu,l_s,:])/self.SQRT_N) * self.S[mu,l_s+1,n2] 
        return part_gap 
    def part_gap_in_before_flip(self,mu,n):
        '''If a spin in the first layer flips, then r_in will change.
        '''
        part_gap = np.zeros(self.N + 1, dtype='float32')
        l_h = 0
        l_s = 0
        # effects on previous gap (only 1 gap is affected)
        part_gap[0] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/self.SQRT_N_IN) * self.S[mu, l_s, n]
        # effects on the N gaps in the next layer. Remember the assumption: in hidden layers, each layer has N nodes.
        for n2 in range(self.N):
            part_gap[1+n2] = ( np.sum(self.J_hidden[l_h, n2, :] * self.S[mu, l_s, :] )/self.SQRT_N ) * self.S[mu, l_s + 1, n2] 
        return part_gap  # Only the N+1 elements affect the Delta_H_eff.
    def part_gap_in_after_flip(self,mu,n):
        l_h = 0
        l_s = 0 
        part_gap = np.zeros(self.N + 1,dtype='float32')
        # effects on previous gap (only 1 gap is affected)
        part_gap[0] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/self.SQRT_N_IN ) * self.new_S[mu, l_s, n]
        # effects on the N gaps in the next layer. Remember the assumption: in hidden layers, each layer has N nodes.
        for n2 in range(self.N):
            part_gap[1 + n2] = ( np.sum(self.J_hidden[l_h, n2, :] * self.new_S[mu, l_s, :] )/self.SQRT_N ) * self.S[mu, l_s + 1, n2]
        return part_gap  # Only the N+1 elements affect the Delta_H_eff.
    def part_gap_out_before_flip(self,mu,n):
        ''' If a spin in the last hidden layer flips, then r_out will change.
        '''
        N = self.N
        N_out = self.N_out
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(N_out + 1, dtype='float32')

        part_gap[0] = (np.sum(self.J_hidden[-1,n,:] * self.S[mu,-2,:]/SQRT_N)) * self.S[mu,-1,n]
        for n2 in range(N_out):
            part_gap[1+n2] = (np.sum(self.J_out[n2,:] * self.S[mu,-1,:]/SQRT_N)) * self.S_out[mu,n2]
        return part_gap  # Only (N_out)+1 gaps affect the Delta_H_eff.
    def part_gap_out_after_flip(self,mu,n):
        ''' If a spin in the last hidden layer flips, then r_out will change.
        '''
        M = self.M
        N_out = self.N_out
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(N_out + 1,dtype='float32')

        part_gap[0] = (np.sum(self.J_hidden[-1,n,:] * self.S[mu,-2,:])/SQRT_N) * self.new_S[mu,-1,n]
        for n2 in range(N_out):
            part_gap[1+n2] = (np.sum(self.J_out[n2,:] * self.new_S[mu,-1,:])/SQRT_N) * self.S_out[mu,n2]
        return part_gap # Only (N_out)+1 gaps affect the Delta_H_eff.

    ## One of the gap function is used if J is shifted.
    def part_gap_in_before_shift(self,n):
        M = self.M
        SQRT_N_IN = self.SQRT_N_IN
        part_gap = np.zeros(M,dtype='float32')

        for mu in range(M):
            part_gap[mu] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
        return part_gap  # Only the M elements affect the Delta_H_eff.
    def part_gap_in_after_shift(self,n):
        M = self.M
        SQRT_N_IN = self.SQRT_N_IN
        part_gap = np.zeros(M,dtype='float32')

        for mu in range(M):
            part_gap[mu] = (np.sum(self.new_J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
        return part_gap 
    def part_gap_out_before_shift(self,n_out):
        M = self.M
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(M,dtype='float32')

        for mu in range(M):
            #self.S[mu,-1,:]  <---> self.S[mu,L-2,:]
            part_gap[mu] = (np.sum(self.J_out[n_out,:] * self.S[mu,-1,:])/SQRT_N) * self.S_out[mu,n_out]
        return part_gap 
    def part_gap_out_before_shift_2(self):
        M = self.M
        N_out = self.N_out
        SQRT_N = self.SQRT_N
        part_gap = np.zeros((M,N_out),dtype='float32')

        for mu in range(M):
            for n_out in range(N_out):
                #self.S[mu,-1,:]  <---> self.S[mu,L-2,:]
                part_gap[mu,n_out] = (np.sum(self.J_out[n_out,:] * self.S[mu,-1,:])/SQRT_N) * self.S_out[mu,n_out]
        return part_gap 
    def part_gap_out_after_shift(self,n_out):
        """The default version is this one."""
        M = self.M
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(M,dtype='float32')

        for mu in range(M):
            #self.S[mu,-1,:]  <---> self.S[mu,L-2,:]
            part_gap[mu] = (np.sum(self.new_J_out[n_out,:] * self.S[mu,-1,:])/SQRT_N) * self.S_out[mu,n_out]
        return part_gap 
    def part_gap_out_after_shift_2(self):
        M = self.M
        N_out = self.N_out
        SQRT_N = self.SQRT_N
        part_gap = np.zeros((M,N_out),dtype='float32')

        for mu in range(M):
            for n_out in range(N_out):
                #self.S[mu,-1,:]  <---> self.S[mu,L-2,:]
                part_gap[mu,n_out] = (np.sum(self.new_J_out[n_out,:] * self.S[mu,-1,:])/SQRT_N) * self.S_out[mu,n_out]
        return part_gap 
    #==========================================
    # Version 2 of part_gap_XX_shift functions. 
    # In this version, if update one bond, because of normaliztion, all bonds are changed, therefore, all bonds in that layer will contribute to the change of the energy.
    # Either of the gap functions is used if J is shifted.
    def part_gap_in_before_shift_2(self):
        M = self.M
        N = self.N
        SQRT_N_IN = self.SQRT_N_IN
        part_gap = np.zeros((M,N),dtype='float32')
        for mu in range(M):
            for n in range(N):
                part_gap[mu,n] = (np.sum(self.J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
        return part_gap  # Only the M elements affect the Delta_H_eff.
    def part_gap_in_after_shift_2(self):
        M = self.M
        N = self.N
        SQRT_N_IN = self.SQRT_N_IN
        part_gap = np.zeros((M,N),dtype='float32')
        for mu in range(M):
            for n in range(N):
                part_gap[mu,n] = (np.sum(self.new_J_in[n,:] * self.S_in[mu,:])/SQRT_N_IN) * self.S[mu,0,n]
        return part_gap 
    def part_gap_hidden_before_shift(self,l,n):
        M = self.M
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.M,dtype='float32')

        for mu in range(M):
            part_gap[mu] = (np.sum(self.J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
        return part_gap  # Only the M elements affect the Delta_H_eff.
    def part_gap_hidden_before_shift_2(self,l):
        M = self.M
        N = self.N
        SQRT_N = self.SQRT_N
        part_gap = np.zeros((M,N),dtype='float32')

        for mu in range(M):
            for n in range(N):
                part_gap[mu,n] = (np.sum(self.J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
        return part_gap  # Only the M elements affect the Delta_H_eff.
    def part_gap_hidden_after_shift(self,l,n):
        M = self.M
        SQRT_N = self.SQRT_N
        part_gap = np.zeros(self.M,dtype='float32')

        for mu in range(M):
            part_gap[mu] = (np.sum(self.new_J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
        return part_gap # Only the M elements affect the Delta_H_eff.
    def part_gap_hidden_after_shift_2(self,l):
        M = self.M
        N = self.N
        SQRT_N = self.SQRT_N
        part_gap = np.zeros((M,N),dtype='float32')
        for mu in range(M):
            for n in range(N):
                part_gap[mu,n] = (np.sum(self.new_J_hidden[l,n,:] * self.S[mu,l-1,:])/SQRT_N) * self.S[mu,l,n]
        return part_gap  # Only the M elements affect the Delta_H_eff.
    #Above are version 2 of part_gap_XXX_*shift functions.
    #====================================================  

    def rand_index_for_S(self):
        # For S: list_index_for_S = [(mu,l,n),...]
        list_index_for_S = []
        for _ in range(self.num_variables * (self.tot_steps-1)):
            list_index_for_S.append([randrange(self.M), randrange(1,self.L-1), randrange(self.N)])
        res_arr = np.array(list_index_for_S)
        return res_arr
    def rand_index_for_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_index_for_J = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            list_index_for_J.append([randrange(1,self.L), randrange(self.N), randrange(self.N)])
        res_arr = np.array(list_index_for_J)
        return res_arr
    def rand_series_for_x(self):
        """
        For generating J: list_for_x = [x1,x2,...]
        We separate rand_index_for_J() and rand_series_for_x(), instead of merginging them to one function and return a list of four-tuple (l,n2,n1,x).
        The reason is: x is float and l,n2,n1 are integers, it will induce trouble if one put them (x and l,n2,n1 ) together.
        """
        list_for_x = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            x = np.random.normal(loc=0,scale=1.0,size=None)
            x = round(x,10)
            list_for_x.append(x)
        res_arr = np.array(list_for_x)
        return res_arr

    def rand_series_for_decision_on_S(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_variables * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr
    def rand_series_for_decision_on_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_bonds * (self.tot_steps-1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr
    def check_and_save_seeds_v2(self,mc_index,list_k):
        """Use MC_step as index, instead of element in range(count_MC_step * self.num) as index
           v2 is a general version than v1.
        """
        ind = mc_index
        init = self.init
        self.list_k = list_k
        if ind==self.list_k[0]:
            self.list_k.pop(0) 
            np.save('../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.M,self.N,self.beta),self.S)   
            np.save('../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.N,self.beta),self.J_hidden)
            np.save('../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init,ind,self.N,self.N_in,self.beta),self.J_in)
            np.save('../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.N_out,self.N,self.beta),self.J_out)
            np.save('../data/host/seed_ener_new_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.M,self.N,self.beta),self.H_hidden)
        else:
            pass
    def check_and_save_seeds(self,mc_index):
        """Use MC_step as index, instead of element in range(count_MC_step * self.num) as index."""
        ind = mc_index
        init = self.init
        if ind==self.list_k[0]:
            self.list_k.pop(0) 
            np.save('../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.M,self.N,self.beta),self.S)   
            np.save('../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.N,self.beta),self.J_hidden)
            np.save('../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init,ind,self.N,self.N_in,self.beta),self.J_in)
            np.save('../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.N_out,self.N,self.beta),self.J_out)
            np.save('../data/host/seed_ener_new_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init,ind,self.L,self.M,self.N,self.beta),self.H_hidden)
        else:
            pass
    def check_and_save_v2(self,count_MC_step):
        ind = count_MC_step
        if int((ratio)**self.ind_save) == ind and self.save < self.S_traj.shape[0]:
            self.S_traj[self.ind_save] = self.S
            self.J_hidden_traj[self.ind_save] = self.J_hidden
            self.H_hidden_traj[self.ind_save] = self.H_hidden
            self.ind_save += 1
            self.check_and_save_v2(ind)
        else:
            pass 
    def check_and_save(self):
        ind = self.count_MC_step
        if 2**self.ind_save == ind and self.ind_save < self.S_traj.shape[0]:
            self.S_traj[self.ind_save] = self.S
            self.J_hidden_traj[self.ind_save] = self.J_hidden
            self.H_hidden_traj[self.ind_save] = self.H_hidden
            self.ind_save += 1
        else:
            pass
    def check_and_save_hyperfine_v2(self,update_index):
        ind = update_index
        if int((ratio)**self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
            self.S_traj_hyperfine[self.ind_save] = self.S
            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
            self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
            self.ind_save += 1
            self.check_and_save_hyperfine_v2(update_index)
        else:
            pass
    def check_and_save_hyperfine(self,update_index):
        ind = update_index
        if (2**self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
            self.S_traj_hyperfine[self.ind_save] = self.S
            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
            self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
            self.ind_save += 1
        else:
            pass

    # One of the follwing decision functions is used if S is flipped.
    def decision_by_mu_l_n_SIMPLE(self,mu,l,n):
        """If use this decision_by_mu_l_n_SIMPLE() function, the parameter EPS is not needed in the input."""
        # Const.s
        rand1 = np.random.random(1)
        a1 = self.part_gap_hidden_before_flip(mu,l,n)
        a2 = self.part_gap_hidden_after_flip(mu,l,n)
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e > EPS:
            if rand1 < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass
        else:
            self.accept_by_mu_l_n(mu,l,n) 
    def decision_by_mu_l_n(self,mu,l,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_hidden_after_flip(mu,l,n)) - calc_ener(self.part_gap_hidden_before_flip(mu,l,n))
        delta_e = self.delta_H
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,l,n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass
    def decision_node_in_by_mu_n(self,mu,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_in_after_flip(mu,n)) - calc_ener(self.part_gap_in_before_flip(mu,n))
        delta_e = self.delta_H
        temp_l = 0
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,temp_l,n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu,temp_l,n)
            else:
                pass
    def decision_node_out_by_mu_n(self,mu,n,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_out_after_flip(mu,n)) - calc_ener(self.part_gap_out_before_flip(mu,n))
        delta_e = self.delta_H
        l = -1
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu,l,n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu,l,n)
            else:
                pass

    # One of the follwing decision functions is used if J is shifted.
    def decision_bond_hidden_by_l_n2_n1(self,l,n2,n1,EPS,rand):
        self.delta_H = calc_ener(self.part_gap_hidden_after_shift(l,n2)) - calc_ener(self.part_gap_hidden_before_shift(l,n2))
        delta_e = self.delta_H
        #================
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            #self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
            self.accept_bond_hidden_by_l_n2_n1(l,n2)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_bond_hidden_by_l_n2_n1(l,n2)
            else:
                pass # We do not need a "remain" function
    def decision_bond_in_by_n2_n1(self,n2,n1,EPS,rand):
        a1 = self.part_gap_in_before_shift(n2)
        a2 = self.part_gap_in_after_shift(n2)
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            self.accept_bond_in_by_n2_n1(n2)
        else:
            #TEST (2 line)
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_bond_in_by_n2_n1(n2)
            else:
                pass # We do not need a "remain" function
    def decision_bond_out_by_n2_n1(self,n2,n1,EPS,rand):
        a1 = self.part_gap_out_before_shift(n2)
        a2 = self.part_gap_out_after_shift(n2)
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e < EPS:
            #self.accept_bond_out_by_n2_n1(n2,n1)
            self.accept_bond_out_by_n2_n1(n2)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                #self.accept_bond_out_by_n2_n1(n2,n1)
                self.accept_bond_out_by_n2_n1(n2)
            else:
                pass # We do not need a "remain" function
    @timethis
    def mc_main(self,replica_index):
        """MC for the host machine, i.e., it will save seeds for different waiting time."""
        str_replica_index = str(replica_index)
        rel_path='J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}_host.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
        src_dir = os.path.dirname(__file__) 
        abs_file_path=os.path.join(src_dir, rel_path)

        EPS = self.EPS
         
        ## MC siulation starts
        for MC_index in range(1,self.tot_steps):
            #print("Updating S:")
            for update_index in range(0, self.num_variables):
                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
                self.flip_spin(mu,l,n)
                rand1=np.random.random(1)
                if l == 0:
                    self.decision_node_in_by_mu_n(mu,n,EPS,rand1)
                elif l == self.num_hidden_node_layers - 1:

                    self.decision_node_out_by_mu_n(mu,n,EPS,rand1)
                else:
                    self.decision_by_mu_l_n(mu,l,n,EPS,rand1)
            #print("Updating J:")
            for update_index in range(0, self.num_bonds):
                self.random_update_J()
            self.count_MC_step += 1
            # Check and save the seeds 
            # IF MC_index EQUALS 2**k, WHERE k = 1,2,3,4,5,...,12, THEN SAVE THE CONFIGURATION OF THE NDOES AND BONDS. 
            # THIS OPERATION SHUOLD BE ONLY DONE IN HOST MACHINE, DO NOT DO IT IN A GUEST MACHINE.
            self.check_and_save_seeds(MC_inex)

            #Check and save the configureation of bonds 
            self.check_and_save() # save configuration

            #========================================================================================================
            # To check if the training dynamics have the aging effect.
            # The basic idea is: After we start training, we save the configurations of J_in, J_out, J_hidden and S 
            # (ie., S_hidden) at MC_index = 4, (8,) 16, (32,) 64, (128,) 256, (512,) 1024, (2048,) 4096, (8192.)
            # For each of these restarting configuration, we run N_replica = 10  dependent training (MC 'dynamics'). 
            # Each replica trajectory should have a label (a name, ie, 0, 1, 2, 3, ..., 9).
            # They can be paired to N_replica * (N_replica-1)/2 pairs. In each pair,
            # one trajectory is palyed the role of 'host_restart_MC_index' and the other is 'guest_restart_MC_index'.
            # After all these dynamics are obtained, we can calculate Q(t,l), q(t,l) for each trajectory pair.
            # Similiarly, we can obtain the tau(l) function for each trajectory.
            # Then we can know if there is aging effect.
            #========================================================================================================

    @timethis
    def mc_main_hyperfine(self,replica_index):
        """MC for the host machine, i.e., it will save seeds for different waiting time."""
        str_replica_index = str(replica_index)
        rel_path='J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}_host.dat'.format(str_replica_index,self.init,self.tw,self.L,self.N,self.beta,self.tot_steps)
        src_dir = os.path.dirname(__file__) 
        abs_file_path=os.path.join(src_dir, rel_path)
        EPS = self.EPS
        
        # MC siulation starts
        for MC_index in range(1,self.tot_steps):
            #print("MC step: {:d}".format(MC_index))
            #print("Updating S:")
            start_index_variables = (MC_index-1) * self.num
            end_index_variables = start_index_variables + self.num_variables 
            for update_index in range(start_index_variables, end_index_variables):
                mu,l,n = randrange(self.M),randrange(self.num_hidden_node_layers),randrange(self.N)
                self.flip_spin(mu,l,n)
                rand1 = np.random.random(1)
                if l == 0:
                    self.decision_node_in_by_mu_n(mu,n,EPS,rand1) 
                elif l == self.num_hidden_node_layers - 1:
                    self.decision_node_out_by_mu_n(mu,n,EPS,rand1)
                else:
                    self.decision_by_mu_l_n(mu,l,n,EPS,rand1) 
                #Check and save the configureation of variables
                self.check_and_save_hyperfine(update_index) # save configuration
            #print("Updating J:")
            start_index_bonds = end_index_variables
            end_index_bonds = MC_index * self.num
            for update_index in range(start_index_bonds, end_index_bonds):
                self.random_update_J()
                #Check and save the configureation of bonds 
                self.check_and_save_hyperfine(update_index) # save configuration
            #Check and save the seeds 
            self.check_and_save_seeds(MC_index)

    def mc_update_J_or_S(self):
        rand1 = np.random.random(1)
        if rand1 < self.PROB:
            self.random_update_S()
        else:
            self.random_update_J()

    @timethis
    def mc_main_random_update_hyperfine(self):
        """1. MC for the main machine. It will save seeds for different waiting time.
           2. Use update_index to denote waiting time (tw).
        """
        STEPS = self.tot_steps * self.num
        for update_index in range(STEPS):
            self.mc_update_J_or_S()
            #Check and save the configureation of variables
            self.check_and_save_hyperfine_v2(update_index) # save configuration
            #Check and save the seeds (ONLY IN mc_main) 
            self.check_and_save_seeds(update_index)
    @timethis
    def mc_random_update_hyperfine(self,str_timestamp):
        """MC for the guest machine. It will not save seeds for J or S. 
        """
        STEPS = self.tot_steps * self.num
        for update_index in range(STEPS):
            #print("Updating S and J randomely, with a fixed probability:")
            self.mc_update_J_or_S()
            #Check and save the configureation of bonds 
            self.check_and_save_hyperfine_v2(update_index) # save configuration
    @timethis
    def mc_main_random_update_hyperfine_2(self):
        """MC for the host machine, i.e., it will
           1. Save seeds for different waiting time.
           2. Use mc_index to denote waiting time (tw).
           3. Updating S and J randomly, with a fixed probability.
        """
        self.update_index = 0
        MC_STEPS = self.tot_steps
        for mc_index in range(MC_STEPS):
            for _ in range(self.num):
                self.mc_update_J_or_S()
                #Check and save the configureation of variables
                self.check_and_save_hyperfine_v2(self.update_index) # save configuration
                self.update_index = self.update_index + 1 
            #Check and save the seeds (ONLY IN mc_main) 
            self.check_and_save_seeds(mc_index)
    @timethis
    def mc_random_update_hyperfine_2(self,str_timestamp):
        """MC for the guest machine. It will 
           1. Use mc_index to denote waiting time (tw).
           2. Updating S and J randomly, with a fixed probability.
        """
        self.update_index = 0
        MC_STEPS = self.tot_steps 
        for mc_index in range(MC_STEPS):
            for _ in range(self.num):
                self.mc_update_J_or_S()
                #Check and save the configureation of variables
                self.check_and_save_hyperfine_v2(self.update_index) # save configuration
                self.update_index = self.update_index + 1 

