#===============
# main.py
# Date: 2021-8-9
#===============
# host.py 
# 1. Run a MC dynamics and save the trajectories for J_in, S, etc.
# 2. Save the seeds for J_in, J_out, S, etc at t = 2**N, where N = 2, 4, 8, 16, ... 
#========
# guest.py 
# 1. Run a MC dynamics and save the trajectories for J_in, S, etc. 
# 2. Do not save seeds.
#============
#Super Module
#============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '../..'
main_dir_local = '/home/gang/Github/Yoshino2G3'

def determine_module_filename(mf_index):
    """This is a meta-function. We use it to decide which utility-directory should use."""
    if mf_index == 0:
        module_filename = main_dir +'/py_functions/'
    elif mf_index == 1:
        module_filename = main_dir_local + '/py_functions_local/'
    return module_filename

module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)

from utilities import calc_ener, generate_S_in_and_out_2_spin_v3_resp, list_only_naked_dir
from HostNetwork import HostNetwork
from Network import Network, generate_tw_list, qq
from utilities_overlap import plot_test_delta_e_bond, plot_test_delta_e_node,plot_test_delta_e_bond_node_hidden

#import math
import datetime
import multiprocessing as mp

#========
#Module 2 
#========
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from random import choice
from random import randrange
import scipy as sp
from scipy.stats import norm
import tensorflow as tf
from time import sleep 
from time import time

#===========
# Parameters
#===========
beta = 0.0 
init = 0 
L = 0 
M = 0 
N = 0 
N_in = 0
N_out = 0
tot_steps = 0 
tw = 0
data_dir = '../data'

num_tw = 0
host_index=1
h = 1
#=========
#Functions
#=========
def guest(data_dir=data_dir,L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, sample_index=1, h=1):
    replica_index = int(time())
    str_replica_index = str(replica_index)

    # Obtain the timestamp list
    start_time_int = int(time())

    timestamp = generate_timestamp(data_dir)
    # Initilize an instance of network.
    o = Network(sample_index,tw,L,M,N,N_in,N_out,tot_steps,beta,timestamp,h,qq)

    # Load data
    str_timestamp = str(timestamp)
    o.load_S_and_J_2g()

    # Define variables
    o.set_vars()

    # Run MC simulations
    o.mc_random_update_hyperfine_2(str_timestamp) 

    # Save the state of S and J at this end of the epoch of training
    np.save('{}/{:s}/S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.S_traj_hyperfine)
    np.save('{}/{:s}/J_in_{:s}_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,str_timestamp,str_replica_index,o.init,o.tw,o.N,o.N_in,o.beta,tot_steps),o.J_in_traj_hyperfine)
    np.save('{}/{:s}/J_out_{:s}_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,str_timestamp,str_replica_index,o.init,o.tw,o.N_out,o.N,o.beta,tot_steps),o.J_out_traj_hyperfine)
    np.save('{}/{:s}/J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,str_timestamp,str_replica_index,o.init,o.tw,o.L,o.N,o.beta,tot_steps),o.J_hidden_traj_hyperfine)
    np.save('{}/{:s}/ener_new_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,str_timestamp,str_replica_index,o.init,o.tw,o.L,o.M,o.N,o.beta,tot_steps),o.H_hidden_traj_hyperfine)

    # Finished
    print("MC simulations (guest) done!")

def init_clean(data_dir=data_dir,L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, sample_index=1, h=1):
    """ """
    timestamp = int(time())
    print("starting time:{}".format(timestamp))
    # In general, the input layer and output layer is NOT of the same size of the hidden layers
    # Assumption: layer=0 denotes input layer; layer=L denotes the output layer; layer=1,..., L-1 denotes the hidden layers. 
    # Assumption: The output layer has N_out nodes. J_out has N * N_out bonds. 
    # We need to define two extra arrays for the input layer J_in and the output layer J_out
    J_in = np.zeros((N,N_in))
    J_out = np.zeros((N_out,N))
    file0 = '/public1/home/sch10322/huanggang/Yoshino2G3/augmented_mnist_3/M{}/dataset_zeros_m{}_sample{}.csv'.format(M,M,sample_index)
    file1 = '/public1/home/sch10322/huanggang/Yoshino2G3/augmented_mnist_3/M{}/dataset_ones_m{}_sample{}.csv'.format(M,M,sample_index)

    # Initilize an instance of network.
    o = HostNetwork(L,M,N,N_in,N_out,tot_steps,timestamp,h) 

    # Load data from tf.keras
    o.S_in,o.S_out = generate_S_in_and_out_2_spin_v3_resp(file0,file1,M,N_in,N_out)
    #o.S_in,o.S_out = generate_random_S_in_and_out(init,M,N_in,N_out)
    #=====================================
    # Make a new directory named timestamp
    #=====================================
    str_timestamp = str(timestamp)
    list_dir = ['../data/', str_timestamp]
    name_dir = "".join(list_dir)
    #==========================
    # Create directory name_dir
    #==========================
    os.makedirs(name_dir,exist_ok=True)

    src_dir = os.path.dirname(__file__) # <-- absolute dir where the script is in

    ##=========================================
    ## Save the arrays in the created directory
    ##=========================================
    # Save the initial configures:
    np.save('{}/{:s}/seed_S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,str_timestamp,sample_index,tw,L,M,N,beta),o.S)
    np.save('{}/{:s}/seed_J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,str_timestamp,sample_index,tw,L,N,beta),o.J_hidden)
    np.save('{}/{:s}/seed_J_in_{:s}_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,str_timestamp,sample_index,tw,N,N_in,beta),o.J_in)
    np.save('{}/{:s}/seed_J_out_{:s}_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,str_timestamp,sample_index,tw,N_out,N,beta),o.J_out)
    #ASSUME that S_in and S_out are fixed during the training. They are independent on the initial configurations (sample_index, h). (For simplicity)
    np.save('{}/{:s}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,M,N_in,beta),o.S_in)
    np.save('{}/{:s}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(data_dir,str_timestamp,M,N_out,beta),o.S_out)

    print('Initial config.s are generated.')

def generate_timestamp(data_dir=data_dir):    
    timestamp_list = list_only_naked_dir(data_dir)
    j = 0
    timestamp = timestamp_list[j] # j is a index, but this index should given by job.sh
    return timestamp
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=1, type=int, default=1, \
                        help="the index of initial configurations.  (the initial Conifiguration index)")
    parser.add_argument('-I', nargs='?', const=N_in, type=int, default=N_in, \
                        help="the number of bits for input.")
    parser.add_argument('-J', nargs='?', const=N_out, type=int, default=N_out, \
                        help="the number of classes for output.")
    parser.add_argument('-L', nargs='?', const=L, type=int, default=L, \
                        help="the number of layers.(Condition: L > 1)")
    parser.add_argument('-M', nargs='?', const=M, type=int, default=M, \
                        help="the number of samples.")
    parser.add_argument('-N', nargs='?', const=N, type=int, default=N, \
                        help="the number of nodes per layer.")
    parser.add_argument('-S', nargs='?', const=tot_steps, type=int, default=tot_steps, \
                        help="the number of total steps.")
    parser.add_argument('-W', nargs='?', const=tw, type=int, default=tw, \
                        help="the waiting time.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,sample_index = args.M,args.L,args.N,args.B,args.S,args.C
    N_in,N_out = args.I,args.J
    tw =  args.W
    NUM_CORES = 24
    n_replica = NUM_CORES
    SIZE_GROUP = 64 # BECAUSE for each value of h, we have 64 initial configurations prepared for the DNN.
    h = int(sample_index/SIZE_GROUP) + 1

    #=========================
    # Generate initiial S and J
    #=========================
    init_clean(data_dir,L,M,N,tot_steps, beta, N_in, N_out, tw, sample_index, h)
 
    timestamp = generate_timestamp(data_dir)
    str_timestamp = str(timestamp)
    #================================================ 
    #Use Multiprocessing to run MC on multiple cores
    #================================================ 
    start_t = datetime.datetime.now()
 
    param_tuple = [(data_dir,L,M,N,tot_steps, beta, N_in, N_out, tw, sample_index, h) for _ in range(NUM_CORES)]

    #MC simulations for other tw's as gustes
    for k in range(0, NUM_CORES):
        print("Start process ({}).".format(k))
        mp.Process(target=guest, args=param_tuple[k]).start() #start now
        sleep(2)

    # Testing
    print("# of replicas:{}.".format(n_replica))
    print("Used" + str(NUM_CORES) + " cores.")
    print("DONE!")
