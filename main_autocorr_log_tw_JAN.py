#============
#Super Module
#============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '../..'
def determine_module_filename(mf_index):
    if mf_index == 0:
        module_filename = main_dir +'/py_functions/'
    return module_filename
module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)

#=========================================
#Import the Module with self-defined class
#=========================================
from utilities import list_only_naked_dir
from utilities_autocorr import autocorr_J_hidden_v2, autocorr_J_in, autocorr_J_out, autocorr_S, plot_autocorr_J_hidden, plot_autocorr_S, \
     plot_test_delta_e_bond, plot_test_delta_e_node, plot_test_delta_e_bond_node_hidden 
from utilities_overlap import *

from Network import tw_list_for_guest, qq, step_list_v2, generate_traj_sampling_num

#print("step_list_v2")
#print(step_list_v2)
#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
from itertools import combinations
from time import sleep
#import math
import datetime
import multiprocessing as mp
#===========
# Parameters
#===========
n_replica = 0 # IMPORTANT: FOR  A NEW SYSTEM, PLEASE SET THE n_replica! 


beta = 66.70
sample_index = 2 
L = 10 
M = 10 
N = 3 
N_in = 784
N_out =2 
tot_steps = 1024 
tw = 2 

def autocorr_log_tw(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, sample_index=sample_index, n_replica=n_replica, autocorr_statistics=False, inner_representation=False):
    #n_pairs = int(n_replica * (n_replica-1) /2)
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------
    print("timestamp_list:")
    print(timestamp_list)
    num_hidden_node_layers = L - 1 
    num_hidden_bond_layers = L - 2
    #Determine the total number of degree of freedom 
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:    
        num = num_dof(L,M,N,N_in,N_out)

    BIAS = 1
    tot_steps_ = generate_traj_sampling_num(tot_steps * num, BIAS, qq) # Rescale 
    ave_traj_JJ0 = np.zeros((tot_steps_,num_hidden_bond_layers,N,N),dtype='float32')
    ave_traj_JJ0_in = np.zeros((tot_steps_,N,N_in),dtype='float32')
    ave_traj_JJ0_out = np.zeros((tot_steps_,N_out,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,num_hidden_node_layers,N),dtype='float32')
    #==============================================
    #aim: To find the indices for replicas
    #There are two ways.
    #method1: match pattern with glob
    #method2: match pattern with startwith, endwith
    #==============================================
    import glob
    i = 0
    stamp = timestamp_list[i]
    path = '/'.join([data_dir,stamp])
    # List the files including the indices for replicas, avoiding to obtain files that includs repeated replica indices.
    #prefixed = [filename for filename in os.listdir(path) if filename.startswith("J_hidden_")]
    prefixed = [filename for filename in glob.glob('/'.join([path,'J_hidden_*_*tw{:d}_*npy'.format(tw)]))]
    str_replica_index_list = []
    str_temp_list = []
    for term in prefixed:
        str_temp_list.append(term.split("/")[-1])
    for term in prefixed:
        #The file names have the format like "J_hidden_1629046894_sample2_*", therefore, we use the following command to extract the timestamp, for example, 1629046894. 
        str_replica_index_list.append(term.split("_",3)[2])

    res_autocorr_1 = np.zeros((n_replica,num_hidden_bond_layers,tot_steps_))
    res_autocorr_1_in = np.zeros((n_replica,tot_steps_))
    res_autocorr_1_out = np.zeros((n_replica,tot_steps_))
    res_autocorr_2 = np.zeros((n_replica,num_hidden_node_layers,tot_steps_))
    autocorr_index = 0

    for autocorr_index, str_replicas in enumerate(str_replica_index_list):
        J_in_traj = np.load('{}/{}/J_in_{:s}_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,N,N_in,beta,tot_steps))
        J_out_traj = np.load('{}/{}/J_out_{:s}_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,N_out,N,beta,tot_steps))
        J_traj = np.load('{}/{}/J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,L,N,beta,tot_steps))
        S_traj = np.load('{}/{}/S_{}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,L,M,N,beta,tot_steps))
        res_autocorr_1[autocorr_index] = autocorr_J_hidden_v2(J_traj)
        res_autocorr_1_in[autocorr_index] = autocorr_J_in(J_in_traj)
        res_autocorr_1_out[autocorr_index] = autocorr_J_out(J_out_traj)
        res_autocorr_2[autocorr_index] = autocorr_S(S_traj)
    ##Remember: Do not use a function'name as a name of variable
    mean_autocorr_1 = np.mean(res_autocorr_1,axis=0)
    mean_autocorr_1_in = np.mean(res_autocorr_1_in,axis=0)
    mean_autocorr_1_out = np.mean(res_autocorr_1_out,axis=0)
    mean_autocorr_2 = np.mean(res_autocorr_2,axis=0)
    np.save('{}/{}/autocorr_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps),mean_autocorr_1)
    np.save('{}/{}/autocorr_J_in_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps),mean_autocorr_1_in)
    np.save('{}/{}/autocorr_J_out_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps),mean_autocorr_1_out)
    np.save('{}/{}/autocorr_S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,timestamp_list[i],sample_index,tw,L,M,N,beta,tot_steps),mean_autocorr_2)
    
	# Statistics of autocorr
    if autocorr_statistics:
        autocorr_J_tstar = res_autocorr_1[:,:,-1]
        autocorr_J_in_tstar = res_autocorr_1_in[:,-1]
        autocorr_J_out_tstar = res_autocorr_1_out[:,-1]
        autocorr_S_tstar = res_autocorr_2[:,:,-1]
        np.save('{}/{:s}/autocorr_J_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps), autocorr_J_tstar)
        np.save('{}/{:s}/autocorr_J_in_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps), autocorr_J_in_tstar)
        np.save('{}/{:s}/autocorr_J_out_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,N,beta,tot_steps), autocorr_J_out_tstar)
        np.save('{}/{:s}/autocorr_S_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,sample_index,tw,L,M,N,beta,tot_steps), autocorr_S_tstar)

    #=====================================================
    # Do the average over different dynamic path from the same initial configuration, and the same waiting time.
    # np.mean() can average over the first axis, 
    # therefore, 'axis=0' is used.
    #=====================================================
    #plot_autocorr_J_hidden(mean_autocorr_1,timestamp_list[i],sample_index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2)
    #plot_autocorr_S(mean_autocorr_2,timestamp_list[i],sample_index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2) 

    #TESTING
    #print(str_temp_list)
    print("str_replica_index_list:") # WHY this list is empty?
    print(str_replica_index_list)
    #print("beta:")
    #print(beta)
    #print("shape of mean Q, and mean of q:")
    print(mean_autocorr_1.shape)
    print(mean_autocorr_2.shape)
    #print("mean_ol_1:")
    #print(mean_ol_1)
    #print("tuple_replicas:")
    #print(tuple_replicas)

def plot_delta_e():
    # To find the address of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    i = 0 # The first directory
    stamp = timestamp_list[i]
    #------------------------------------------------------------------------
    # Define the ener differences.
    Delta_E_bond_in = np.load('{}/{}/delta_e_bond_in.npy'.format(data_dir,stamp) )
    Delta_E_bond_hidden = np.load('{}/{}/delta_e_bond_hidden.npy'.format(data_dir,stamp) )
    Delta_E_bond_out = np.load('{}/{}/delta_e_bond_out.npy'.format(data_dir,stamp) )
    Delta_E_node_in = np.load('{}/{}/delta_e_node_in.npy'.format(data_dir,stamp) )
    Delta_E_node_hidden = np.load('{}/{}/delta_e_node_hidden.npy'.format(data_dir,stamp) )
    Delta_E_node_out = np.load('{}/{}/delta_e_node_out.npy'.format(data_dir,stamp) )
    # Plot
    plot_test_delta_e_bond(Delta_E_bond_in, Delta_E_bond_hidden, Delta_E_bond_out)
    plot_test_delta_e_node(Delta_E_node_in, Delta_E_node_hidden, Delta_E_node_out)
    plot_test_delta_e_bond_node_hidden(Delta_E_bond_hidden, Delta_E_node_hidden)
    plot_test_delta_e_bond_hidden(Delta_E_bond_hidden)
    plot_test_delta_e_node_hidden(Delta_E_node_hidden)
    plot_test_delta_e_bond_in(Delta_E_bond_in)
    plot_test_delta_e_node_in(Delta_E_node_in)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    #========================================================================================
    # To calclate the overlaps on parallel, we do not need the waiting time (tw) as an input. 
    # Instead, we write the tw's in a list and define a parameter list.
    #========================================================================================
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=sample_index, type=int, default=sample_index, \
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
    parser.add_argument('-R', nargs='?', const=n_replica, type=int, default=n_replica, \
                        help="the number of replicas.")
    parser.add_argument('-S', nargs='?', const=tot_steps, type=int, default=tot_steps, \
                        help="the number of total steps.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,sample_index,n_replica = args.M,args.L,args.N,args.B,args.S,args.C,args.R
    N_in,N_out = args.I,args.J

    #Generate the list of tw.
    tw_list = tw_list_for_guest(tot_steps)

    #================================================ 
    #Calculate overlaps
    #Use Multiprocessing to run MC on different cores
    #================================================ 
    start_t = datetime.datetime.now()
    NUM_CORES = 48  # THIS is A PARAMETER. It denotes how many cores you plan to use.
    print("The computer has " + str(NUM_CORES) + " cores.") 
    # Define parameters for each process, and store each set of input as a value of a dictionary dic_para.
    dic_para = {}
    for i in range(NUM_CORES):
        dic_para["para{}".format(i)] = (L,M,N,tot_steps, beta, N_in, N_out, tw_list[int(i/n_replica)], sample_index, n_replica, True, False)
    #The number of cores required is the number of tw's 
    for key, value in dic_para.items():
        print("Now start process ({}).".format(key))
        mp.Process(target=autocorr_log_tw, args=value).start() #start now
        sleep(1)
