#================================================================================================
#Code name: overlaps_more.py
#Author: Gang Huang
#Date: 2021-4-23
#Version : 0
#Before running this code, one have to calculate all the basic overlaps of J (or J_hidden) and S.
#================================================================================================
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

#=======
#Modules
#=======
from utilities import *
from utilities_overlap import *

#===================================
#LOAD ALL BASIC RESULTS FOR OVERLAPS
#===================================
from Network import l_list, l_S_list, qq, step_list_v2
from Network import generate_traj_sampling_num, generate_tw_list

#=======
# Module
#=======
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

#================
# Parameter set 1
#================
num_tw = 0  # integer
init = 2 # CONSTANT, convension: we store averaged data in *_init2      

#================
# Parameter set 2
#================
beta = 0.0 # float
L = 0 
M = 0 
N = 0 
N_in = 0
N_out = 0
D = 0
tot_steps = 0 
n_replica = 0


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

    #========== 
    # Variables
    #========== 
    
    #_tw = 0 #CONSTANT
    tw_list = generate_tw_list(tot_steps)
    num_tw = len(tw_list)
    tot_steps_list = [tot_steps] * num_tw 
    #BASIC PARAMETERS
    import argparse
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory now. So set the index i=0.
    i = 0
    stamp = timestamp_list[i]

    #-------------------------------------
    # Generate a parameter list para_list.
    #-------------------------------------
    list_of_step_and_tw_tuple = [] # an Auxiliary list
    for i in range(num_tw):
        list_of_step_and_tw_tuple.append((tot_steps_list[i],tw_list[i]))
    para_list = []
    for _, item in enumerate(list_of_step_and_tw_tuple):
        para_list.append([L, M, N, N_in, N_out, item[0], item[1], init]) 

    #Determine the total number of degree of freedom 
    num = num_dof(L,M,N,N_in,N_out)

    BIAS = 1
    tot_steps_max = max(tot_steps_list) # Only one value is used.
    tot_steps_ = generate_traj_sampling_num(tot_steps_max*num,BIAS,qq) # Rescale 
    print("tot_steps_")
    print(tot_steps_)
    #==========================================================================
    # WE WILL LOAD ALL THE CALCULATED Overlaps, AND COMBINE THEM INTO NEW ARRAYS, NAMMED grand_J AND grand_S. THEREFORE, WE
    # create two arrays: the shape of J or S, ref averaged res_J and res_S in overlaps_twX.py.
    grand_J = np.zeros((num_tw, L-2, tot_steps_))
    grand_S = np.zeros((num_tw, L-1, tot_steps_))
    print("grand_J.shape") 
    print(grand_J.shape) 
    #load with loops
    print("init{}:",init)
    
    for index_tw, tw in enumerate(tw_list):
        grand_J[index_tw] = np.load('{}/{}/overlap_J_ave_over_sample_{:s}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,N,beta,tot_steps_list[index_tw]))
        grand_S[index_tw] = np.load('{}/{}/overlap_S_ave_over_sample_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,tw,L,M,N,beta,tot_steps_list[index_tw]))

    # Save the averaged overlaps
    # The grand_S and grand_J are the averaged overlaps of J and S over different initial configurations.
    np.save('{}/{:s}/grand_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,L,M,N,beta,tot_steps_max),grand_J)
    np.save('{}/{:s}/grand_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,stamp,L,M,N,beta,tot_steps_max),grand_S)
    #=====================================================
    # plot the overlaps for a fixed layer (eg, 2), to see the waiting time-dependence of the overlaps Q(t,l) and q(t,l).
    #=====================================================
    for l_index in l_list: 
        #plot_overlap_J_tw_X_ave_over_init_16(grand_J[0],grand_J[1],grand_J[2],grand_J[3],grand_J[4],grand_J[5],grand_J[6],grand_J[7],grand_J[8],grand_J[9],grand_J[10],grand_J[11],grand_J[12],grand_J[13],grand_J[14],grand_J[15],stamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps_max,tw_list,step_list_v2)
        plot_overlap_J_tw_X_ave_over_init_4(grand_J[0],grand_J[1],grand_J[2],grand_J[3],stamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps_max,tw_list,step_list_v2)
    for l_index in l_S_list: 
        #plot_overlap_S_tw_X_ave_over_init_16(grand_S[0],grand_S[1],grand_S[2],grand_S[3],grand_S[4],grand_S[5],grand_S[6],grand_S[7],grand_S[8],grand_S[9],grand_S[10],grand_S[11],grand_S[12],grand_S[13],grand_S[14],grand_S[15],stamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps_max,tw_list,step_list_v2)
        plot_overlap_S_tw_X_ave_over_init_4(grand_S[0],grand_S[1],grand_S[2],grand_S[3],stamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps_max,tw_list,step_list_v2)
