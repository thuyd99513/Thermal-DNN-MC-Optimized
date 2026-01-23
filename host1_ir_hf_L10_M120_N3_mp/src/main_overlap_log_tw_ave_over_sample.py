#============
#Super Module
#============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '../..'
main_dir_local = '../..'
def determine_module_filename(mf_index):
    if mf_index == 0:
        module_filename = main_dir +'/py_functions'
    elif mf_index == 1:
        module_filename = main_dir_local + '/py_functions_local'
    return module_filename
module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)
#=========================================
#Import the Module with self-defined class
#=========================================
from Network import generate_tw_list,generate_traj_sampling_num
from Network import init_list,step_list_v2,qq

from utilities import *
from utilities_overlap import num_dof, num_dof_inner_representation, print_q, print_qq, print_qq_in, print_qq_out, plot_ave_overlap_S, plot_ave_overlap_J, errorbar_ave_overlap_S, errorbar_ave_overlap_J 

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl
import datetime
import multiprocessing as mp
from time import sleep

#===========
# Parameters
#===========
beta = 66.7 # float
init = 2 
L = 10 
M = 10 
N = 3 
N_in = 784
N_out = 10
D = 0
tot_steps = 0 
n_replica = 0 
tw = 0
X_index = 1

def overlap_log_ave_over_sample(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, n_replica=n_replica,init_list=init_list, inner_representation=False, overlap_statistics=False,X_index=1):
    """X_index: the index for calculating once depending on a certain sample_index, for example, sample_index = 2."""
    n_pairs = int(n_replica * (n_replica-1)/2)
    n_init = len(init_list)
    import argparse
    mpl.use('Agg')
    ext_index = 0
     
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.

    #Determine the total number of degree of freedom 
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:
        num = num_dof(L,M,N,N_in,N_out)

    BIAS = 1
    tot_steps_ = generate_traj_sampling_num(num*tot_steps,BIAS,qq)
    #tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 
     
    ave_traj_JJ0 = np.zeros((tot_steps_,L-2,N,N),dtype='float32')
    ave_traj_JJ0_in = np.zeros((tot_steps_,N,N_in),dtype='float32')
    ave_traj_JJ0_out = np.zeros((tot_steps_,N_out,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,L-1,N),dtype='float32')
    
    i = 0
    stamp = timestamp_list[i]
    #========================================================================
    # For getting the shape of res_J and res_S arrays, we load overlap_X.npy
    #========================================================================
    res_J=np.load('{}/{}/overlap_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,X_index,tw,L,N,beta,tot_steps))
    res_J_in=np.load('{}/{}/overlap_J_in_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,X_index,tw,L,N,beta,tot_steps))
    res_J_out=np.load('{}/{}/overlap_J_out_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,X_index,tw,L,N,beta,tot_steps))
    res_S=np.load('{}/{}/overlap_S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,X_index,tw,L,M,N,beta,tot_steps))
    print("shape of J, and of S:")
    print(res_J.shape)
    print(res_J_in.shape)
    print(res_J_out.shape)
    print(res_S.shape)

    res_J_ave_over_sample = np.zeros(res_J.shape)
    res_J_in_ave_over_sample = np.zeros(res_J_in.shape)
    res_J_out_ave_over_sample = np.zeros(res_J_out.shape)
    res_S_ave_over_sample = np.zeros(res_S.shape)
    std_J_ave_over_sample = np.zeros(res_J.shape)
    std_J_in_ave_over_sample = np.zeros(res_J_in.shape)
    std_J_out_ave_over_sample = np.zeros(res_J_out.shape)
    std_S_ave_over_sample = np.zeros(res_S.shape)
    temp_res_J = np.zeros((n_init,res_J.shape[0],res_J.shape[1]))
    temp_res_J_in = np.zeros((n_init,res_J_in.shape[0]))
    temp_res_J_out = np.zeros((n_init,res_J_out.shape[0]))
    temp_res_S = np.zeros((n_init,res_S.shape[0],res_J.shape[1]))
    for j,init in enumerate(init_list):     
        data_dir_ = '../../ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp_tw{:d}/data1'.format(L,M,N,init,tw)
        timestamp_list2 = list_only_naked_dir(data_dir_) # There is only one directory.
        stamp2 = timestamp_list2[0]
        temp_res_J[j] = np.load('{}/{}/overlap_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,N,beta,tot_steps))
        temp_res_J_in[j] = np.load('{}/{}/overlap_J_in_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,N,beta,tot_steps))
        temp_res_J_out[j] = np.load('{}/{}/overlap_J_out_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,N,beta,tot_steps))
        temp_res_S[j] = np.load('{}/{}/overlap_S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,M,N,beta,tot_steps))

    #=====================================================
    # Do the average over different dynamic path from the same initial configuration, and the same waiting time.
    # np.mean() can average over the first axis, 
    # therefore, 'axis=0' is used.
    #=====================================================
    scale = len(init_list)
    res_J_ave_over_sample = np.mean(temp_res_J,axis=0) # "axis=0" is required
    res_J_in_ave_over_sample = np.mean(temp_res_J_in,axis=0) # "axis=0" is required
    res_J_out_ave_over_sample = np.mean(temp_res_J_out,axis=0) # "axis=0" is required
    res_S_ave_over_sample = np.mean(temp_res_S,axis=0)
    std_J_ave_over_sample = np.std(temp_res_J,axis=0)/np.sqrt(scale) # "axis=0" is required
    std_J_in_ave_over_sample = np.std(temp_res_J_in,axis=0)/np.sqrt(scale) # "axis=0" is required
    std_J_out_ave_over_sample = np.std(temp_res_J_out,axis=0)/np.sqrt(scale) # "axis=0" is required
    std_S_ave_over_sample = np.std(temp_res_S,axis=0)/np.sqrt(scale)

    # Go back to the current location: to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.

    #Save the average overlaps of J and S: 
    np.save('{}/{}/overlap_J_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),res_J_ave_over_sample)
    np.save('{}/{}/overlap_J_in_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),res_J_in_ave_over_sample)
    np.save('{}/{}/overlap_J_out_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),res_J_out_ave_over_sample)
    np.save('{}/{}/overlap_S_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,M,N,beta,tot_steps),res_S_ave_over_sample)
    np.save('{}/{}/overlap_std_J_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),std_J_ave_over_sample)
    np.save('{}/{}/overlap_std_J_in_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),std_J_in_ave_over_sample)
    np.save('{}/{}/overlap_std_J_out_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps),std_J_out_ave_over_sample)
    np.save('{}/{}/overlap_std_S_ave_over_weight_sample_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,M,N,beta,tot_steps),std_S_ave_over_sample)
    #print("shape of J, and of S:")
    #print(res_J.shape)
    #print(res_S.shape)
    #print("shape of average J, and of S:")
    #print(res_J_ave_over_sample.shape)
    #print(res_S_ave_over_sample.shape)
    #print("tot_steps_") 
    print(tot_steps_) 
    #Plot or not plot 
    print_q(res_S_ave_over_sample,std_S_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2,inner_representation=False) 
    print_qq(res_J_ave_over_sample,std_J_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2,inner_representation=False) 
    print_qq_in(res_J_in_ave_over_sample,std_J_in_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2,inner_representation=False) 
    print_qq_out(res_J_out_ave_over_sample,std_J_out_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2,inner_representation=False) 
    if overlap_statistics:
        overlap_tstar_S = np.zeros((n_init, n_pairs, L-1))
        overlap_tstar_J = np.zeros((n_init, n_pairs, L-2)) # Maybe for J_in and J_out can do the same thing
        for j,init in enumerate(init_list):     
            data_dir_ = '../../ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp_tw{:d}/data1'.format(L,M,N,init,tw)
            timestamp_list2 = list_only_naked_dir(data_dir_) # There is only one directory.
            stamp2 = timestamp_list2[0]
            print(overlap_tstar_S.shape)
            print(overlap_tstar_J.shape)
            overlap_tstar_S[j] = np.load('{}/{}/overlap_S_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,M,N,beta,tot_steps))
            overlap_tstar_J[j] = np.load('{}/{}/overlap_J_tstar_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,N,beta,tot_steps))
        n_init_pairs = n_init * n_pairs
        overlap_tstar_S = np.reshape(overlap_tstar_S, (n_init_pairs, L-1))      
        overlap_tstar_J = np.reshape(overlap_tstar_J, (n_init_pairs, L-2)) 
        overlap_tstar_S_T = overlap_tstar_S.T        
        overlap_tstar_J_T = overlap_tstar_J.T
        #Save the J_tstar and S_tstar in file for layer l, respectively, in a specific directory.
        data_dir_ = '../../ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp_tw{:d}/data1'.format(L,M,N,init,2,tw)
        for l in range(L-1):
            file_overlap_tstar = open("{}/overlap_S_tstart_tw{:d}_L{:d}_M{:d}_N{:d}_step{:d}_l{:d}.dat".format(data_dir,tw,L,M,N,tot_steps,l),"w")
            file_overlap_tstar.write("# overlap_S\n")
            for j in range(n_init_pairs): 
                #file_acc.write("{:7.4f} {:7.6f}\n".format(time_arr[i], acc_arr[i])) 
                file_overlap_tstar.write("{}\n".format( overlap_tstar_S_T[l][j]) )
            file_overlap_tstar.close()
        
if __name__ == '__main__':
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
    parser.add_argument('-R', nargs='?', const=n_replica, type=int, default=n_replica, \
                        help="the number of replicas.")
    parser.add_argument('-S', nargs='?', const=tot_steps, type=int, default=tot_steps, \
                        help="the number of total steps.")
    args = parser.parse_args()
    M,L,N,beta,tot_steps,n_replica = args.M,args.L,args.N,args.B,args.S,args.R
    N_in,N_out = args.I,args.J
    X_index = args.C
    tw = 1024
    #================================
    #To calculate the average overlap
    #================================
    start_t = datetime.datetime.now()
    NUM_CORES = 1 # THIS A PARAMETER. IT DEPENDS ON THE SETTING IN PREVIOUS STEPS. 
    para_tuple = []
    for i in range(NUM_CORES):
        para_tuple.append((L, M, N, tot_steps, beta, N_in, N_out, tw, n_replica, init_list, False,True, X_index)) 
    #The number of cores required is the number of tw's 
    for i,item in enumerate(para_tuple):
        print("Now start process ({}).".format(i))
        mp.Process(target=overlap_log_ave_over_sample, args=item).start() #start now
        sleep(1)
