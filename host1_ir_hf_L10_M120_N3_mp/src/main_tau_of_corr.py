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
from utilities import *
from utilities_overlap import *

from Network import l_list, qq, step_list_v2
from Network import generate_traj_sampling_num, generate_tw_list

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

num_tw = 0 

beta = 66.7
L = 10 
N = 3 
N_in = 784
N_out = 2
D = 0
init = 2
M = 30 
tot_steps = 128 
n_replica = 64

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

    import argparse
    parser = argparse.ArgumentParser()

    #PARAMETERS
    n_pairs = n_replica * (n_replica-1)/2
    tot_steps_list = [tot_steps] * num_tw
    tw_list = generate_tw_list(tot_steps)

    step_array_v2 = np.array(step_list_v2)

    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------
    # M,L,N,tot_steps, we can obtain these parameter from the first direcotry
    # The following 5 lines do this thing. 
 
    #num_hidden_bond_layers = L - 2
    #num_variables = N * M * num_hidden_node_layers 
    #num_bonds = N * N_in + SQ_N * num_hidden_bond_layers + N_out * N
    #num_variables = int(num_variables) 
    #num_bonds = int(num_bonds)
    #num = num_variables + num_bonds

    #Determine the total number of degree of freedom 
    num = num_dof(L,M,N,N_in,N_out)
    n_sampling_start = 71
    n_sampling_stop = 72
    BIAS = 1
    #tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 
    tot_steps_ = generate_traj_sampling_num(tot_steps * num, BIAS, qq) # Rescale 
     
    ave_traj_JJ0 = np.zeros((tot_steps_,L-2,N,N),dtype='float32')
    ave_traj_SS0 = np.zeros((tot_steps_,M,L-1,N),dtype='float32')
    
    #aim: To find the indices for replicas
    #method1: match pattern with glob
    #method2: match pattern with startwith, endwith
    import glob
    i = 0
    stamp = timestamp_list[i]
    path = '/'.join([data_dir,stamp])
    #prefixed = [filename for filename in os.listdir(path) if filename.startswith("J_hidden_")]
    prefixed = [filename for filename in glob.glob('/'.join([path,'overlap_*npy']))]
    grand_J = np.load('{}/{:s}/grand_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,L,M,N,beta,tot_steps))
    grand_S = np.load('{}/{:s}/grand_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,L,M,N,beta,tot_steps))

    #PART 1
    shape = grand_J.shape
    shape2 = grand_S.shape
    print("shape:{} !".format(shape))
    grand_Q_J = np.zeros((shape[0],shape[1]))
    grand_q_S = np.zeros((shape2[0],shape2[1]))
    for tw_index in range(shape[0]):
        print("tw index:")
        print(tw_index)
        for l_index in range(shape[1]):
            print("l index:")
            print(l_index)
            QJ = grand_J[tw_index,l_index]
            #for Q
            x1 = np.mean(QJ[n_sampling_start:n_sampling_stop])
            grand_Q_J[tw_index,l_index] = x1 
            print(grand_Q_J[tw_index,l_index])
    for tw_index in range(shape2[0]):
        print("tw index:")
        print(tw_index)
        for l_index in range(shape2[1]):
            print("l index:")
            print(l_index)
            QS = grand_S[tw_index,l_index]
            #for q
            x2 = np.mean(QS[n_sampling_start:n_sampling_stop])
            grand_q_S[tw_index,l_index] = x2
            #for Q
            print("grand_tau_S:")
            print(grand_q_S)
            print(grand_q_S[tw_index,l_index])

    print("grand_tau_J:")
    print(grand_Q_J)
    print("grand_tau_S:")
    print(grand_q_S)
    np.save('{}/grand_Q_J_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_Q_J)
    np.save('{}/grand_q_S_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_q_S)
    #Plot tau_J and tau_S
    plot_tau_J_tw_X(grand_Q_J,stamp,L,M,N,beta,tot_steps,tw_list)
    plot_tau_S_tw_X(grand_q_S,stamp,L,M,N,beta,tot_steps,tw_list)
    
    #PART 2
    shape = grand_J.shape
    shape2 = grand_S.shape
    print("shape:{} !".format(shape))
    grand_J_mean = np.mean(grand_J,axis=0)
    grand_S_mean = np.mean(grand_S,axis=0)
    grand_Q_J = np.zeros(shape[1])
    grand_q_S = np.zeros(shape2[1])
    for l_index in range(shape[1]):
        print("l index:")
        print(l_index)
        QJ = grand_J_mean[l_index]
        #for Q
        x1 = np.mean(QJ[n_sampling_start:n_sampling_stop])
        grand_Q_J[l_index] = x1 
        print(grand_Q_J[l_index])
    for l_index in range(shape2[1]):
        QS = grand_S_mean[l_index]
        #for q
        x2 = np.mean(QS[n_sampling_start:n_sampling_stop])
        grand_q_S[l_index] = x2
        #for Q
        print("grand_tau_S:")
        print(grand_q_S)
        print(grand_q_S[l_index])

    print("grand_tau_J:")
    print(grand_Q_J)
    print("grand_tau_S:")
    print(grand_q_S)
    np.save('{}/grand_Q_J_mean_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_Q_J)
    np.save('{}/grand_q_S_mean_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,L,M,N,beta,tot_steps), grand_q_S)
    #Plot tau_J and tau_S
    plot_tau_J_tw_X_mean(grand_Q_J,stamp,L,M,N,beta,tot_steps)
    plot_tau_S_tw_X_mean(grand_q_S,stamp,L,M,N,beta,tot_steps)
