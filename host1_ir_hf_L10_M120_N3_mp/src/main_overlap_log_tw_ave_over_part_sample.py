#============
#Super Module
#============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '/public1/home/sc91981/Yoshino2G3'
main_dir_local = '/home/gang/Github/Yoshino2G3'
def determine_module_filename(mf_index):
    if mf_index == 0:
        module_filename = main_dir +'/py_functions/'
    elif mf_index == 1:
        module_filename = main_dir_local + '/py_functions_local/'
    return module_filename
module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)
#=========================================
#Import the Module with self-defined class
#=========================================
from Network import generate_tw_list,generate_traj_sampling_num
from Network import step_list_v2,qq

from utilities import *
from utilities_overlap import num_dof, print_q, plot_ave_overlap_S, plot_part_ave_overlap_S_loglog, plot_ave_overlap_J, errorbar_ave_overlap_S, errorbar_ave_overlap_J, nsample_list 

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
n_replica = 4 
tw = 0

def overlap_log_ave_over_part_sample(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, n_replica=n_replica):
    '''This function is for seeing the convergence of the overlaps over the total number of samples.'''
    n_pairs = n_replica * (n_replica-1)/2
    import argparse
    mpl.use('Agg')
    ext_index = 0
     
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory. (Mthod 1, sometimes does not work.)
    #timestamp_list = list_only_first_dir(data_dir) # There is only one directory.

    #Determine the total number of degree of freedom 
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
    #res_J=np.load('{}/{}/overlap_J_{:s}_sample2_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps))
    res_S=np.load('{}/{}/overlap_S_{:s}_sample2_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,M,N,beta,tot_steps))
    print("shape of J, and of S:")
    #print(res_J.shape)
    print(res_S.shape)
    n_scale = len(nsample_list)

    #res_J_ave_over_sample = np.zeros(res_J.shape)
    res_S_ave_over_sample = np.zeros(res_S.shape)
    #std_J_ave_over_sample = np.zeros(res_J.shape)
    #std_S_ave_over_sample = np.zeros(res_S.shape)
    res_S_ave_over_sample_high = np.zeros((n_scale, res_S.shape[0], res_S.shape[1]))
    for k, nsample in enumerate(nsample_list):
        init_list = list(range(nsample))
        print("number of samples we are considering:{}.".format(len(init_list)) )
        #temp_res_J = np.zeros((len(init_list),res_J.shape[0],res_J.shape[1]))
        temp_res_S = np.zeros((len(init_list),res_S.shape[0],res_S.shape[1]))
        for j,init in enumerate(init_list):     
            data_dir_ = '../../ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp/data1'.format(L,M,N,init)
            timestamp_list2 = list_only_naked_dir_v2(data_dir_) # There is only one directory.
            stamp2 = timestamp_list2
            print("stamp2:{}".format(stamp2))
            print("number of samples:{}".format(nsample))
            #temp_res_J[j] = np.load('{}/{}/overlap_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,N,beta,tot_steps))
            temp_res_S[j] = np.load('{}/{}/overlap_S_{:s}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir_,stamp2,stamp2,init,tw,L,M,N,beta,tot_steps))
        #=====================================================
        # Do the average over different dynamic path from the same initial configuration, and the same waiting time.
        # np.mean() can average over the first axis, 
        # therefore, 'axis=0' is used.
        #=====================================================
        #scale = nsample
        #res_J_ave_over_sample = np.mean(temp_res_J,axis=0) # "axis=0" is required
        res_S_ave_over_sample = np.mean(temp_res_S,axis=0)
        #std_J_ave_over_sample = np.std(temp_res_J,axis=0)/np.sqrt(scale) # "axis=0" is required
        #std_S_ave_over_sample = np.std(temp_res_S,axis=0)/np.sqrt(scale)
        res_S_ave_over_sample_high[k] = res_S_ave_over_sample
        
        #Maybe this lines are not required.
        # Go back to the current location: to find the locations of configurations.
        data_dir = '../data1'
        timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.

        #Save the average overlaps of J and S: 
        #np.save('{}/{}/overlap_J_ave_over_{:d}_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,nsample,stamp,tw,L,N,beta,tot_steps),res_J_ave_over_sample)
        np.save('{}/{}/overlap_high_S_ave_over_{:d}_sample_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,nsample,stamp,tw,L,M,N,beta,tot_steps),res_S_ave_over_sample_high)
        #np.save('{}/{}/overlap_std_J_ave_over_{:d}_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,nsample,stamp,tw,L,N,beta,tot_steps),std_J_ave_over_sample)
        #np.save('{}/{}/overlap_std_S_ave_over_{:d}_sample_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,nsample,stamp,tw,L,M,N,beta,tot_steps),std_S_ave_over_sample)
        print("shape of J, and of S:")
        #print(res_J.shape)
        print(res_S.shape)
        print("shape of average J, and of S:")
        #print(res_J_ave_over_sample.shape)
        print(res_S_ave_over_sample.shape)
        print("tot_steps_") 
        print(tot_steps_) 
        #Plot
        #plot_ave_overlap_J(res_J_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2)
        #plot_ave_overlap_S(res_S_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2) 
        #print_q(res_S_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2) 
        #errorbar_ave_overlap_J(res_J_ave_over_sample,std_J_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2)
        #errorbar_ave_overlap_S(res_S_ave_over_sample,std_S_ave_over_sample,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2) 
    print("LOOP DONE, FINE!")
    for l in range(L-2):
        plot_part_ave_overlap_S_loglog(res_S_ave_over_sample_high,stamp,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list_v2,nsample_list,l) 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
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

    tw_list = generate_tw_list(tot_steps) 

    #================================
    #To calculate the average overlap
    #================================
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count()) 
    print("The computer has " + str(num_cores) + " cores.")
    para_tuple = []
    for i, item in enumerate(tw_list):
        para_tuple.append((L,M,N,tot_steps, beta, N_in, N_out, tw_list[i], n_replica)) 
    #The number of cores required is the number of tw's 
    for i,item in enumerate(para_tuple):
        print("Now start process ({}).".format(i))
        mp.Process(target=overlap_log_ave_over_part_sample, args=item).start() #start now
        sleep(1)
