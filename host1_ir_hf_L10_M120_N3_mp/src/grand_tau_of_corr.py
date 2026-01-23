#=================================================================================================
#Code name: overlaps_more.py
#Author: Gang Huang
#Date: 2021-4-23
#Version : 0
#Before running this code, one have to calculate all the basic overlaps vof J (or J_hidden) and S.
#=================================================================================================
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

from utilities import *
from utilities_overlap import *

#===================================
#LOAD ALL BASIC RESULTS FOR OVERLAPS
from Network import l_list,l_S_list, M_list, generate_tw_list

#=======
# Module
#=======
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

#===========
# Parameters
#===========
beta = 66.7
init = 2
L = 0 
M = 0 
N = 3 
N_in = 784
N_out = 2
tot_steps = 0 
tw = 0

if __name__ == '__main__':
    #BASIC PARAMETERS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', nargs='?', const=beta, type=float, default=beta, \
                        help="the inverse temperautre.")
    parser.add_argument('-C', nargs='?', const=init, type=int, default=init, \
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
    args = parser.parse_args()
    M,L,N,beta,tot_steps,init = args.M,args.L,args.N,args.B,args.S,args.C
    N_in,N_out = args.I,args.J
    mpl.use('Agg')
    ext_index = 0
 
    alpha_list = np.array(M_list)/N

    #Generate the list of tw.
    tw_list = generate_tw_list(tot_steps)
    
    tot_steps_list = [tot_steps] * len(M_list)

    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory now. So set the index i=0.
    i = 0

    SQ_N = N ** 2
    num_hidden_node_layers = L - 1 
    #num_hidden_bond_layers = L - 2
    num_variables = int(N * M * num_hidden_node_layers) 
    num_bonds = int(SQ_N * L)
    num = num_variables + num_bonds

    BIAS = 1
    #tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 

    ggrand_tau_J = np.zeros((len(alpha_list),len(tw_list),len(l_list)))
    ggrand_tau_S = np.zeros((len(alpha_list),len(tw_list),len(l_S_list)))
    ggrand_tau_J_LOGIC = np.zeros((len(alpha_list),len(tw_list),len(l_list)))
    ggrand_tau_S_LOGIC = np.zeros((len(alpha_list),len(tw_list),len(l_S_list)))
    #==========================================================================
    # WE WILL LOAD ALL THE CALCULATED Overlaps, AND COMBINE THEM INTO NEW ARRAYS, NAMMED grand_J AND grand_S. THEREFORE, WE
    # create two arrays: the shape of J or S, ref averaged res_J and res_S in overlaps_twX.py.
   
    #load with loops
    print("init{}:",init)
    for index_M,term in enumerate(M_list):
        ggrand_tau_J[index_M] = np.load(main_dir+'/ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp/data1/grand_tau_J_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(L,term,N,init,L,term,N,beta,tot_steps_list[index_M]))
        ggrand_tau_S[index_M] = np.load(main_dir +'/ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp/data1/grand_tau_S_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(L,term,N,init,L,term,N,beta,tot_steps_list[index_M]))
        ggrand_tau_J_LOGIC[index_M] = np.load(main_dir+'/ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp/data1/grand_tau_J_LOGIC_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(L,term,N,init,L,term,N,beta,tot_steps_list[index_M]))
        ggrand_tau_S_LOGIC[index_M] = np.load(main_dir +'/ir_hf_L{:d}_M{:d}_N{:d}_sample{:d}_mp/data1/grand_tau_S_LOGIC_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(L,term,N,init,L,term,N,beta,tot_steps_list[index_M]))

    # Save the averaged overlaps
    # The grand_S and grand_J are the averaged overlaps of J and S over different initial configurations.
    np.save('{}/ggrand_tau_J_{:s}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(data_dir,timestamp_list[i],L,N,beta),ggrand_tau_J)
    np.save('{}/ggrand_tau_S_{:s}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(data_dir,timestamp_list[i],L,N,beta),ggrand_tau_S)
   
    #=====================================================
    # plot the overlaps for a fixed layer (eg, 2), to see the waiting time-dependence of the overlaps Q(t,l) and q(t,l).
    #=====================================================
    ggrand_tau_J = ggrand_tau_J_LOGIC.transpose((1,2,0))
    ggrand_tau_S = ggrand_tau_S_LOGIC.transpose((1,2,0))
    for l_index,term in enumerate(l_list): 
        print("shape of ggrand_tau_J:")
        print(ggrand_tau_J.shape)
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[:],l_index,L,N,beta,alpha_list,tw_list[:])
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[0::2],l_index,L,N,beta,alpha_list,tw_list[0::2])
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[1::2],l_index,L,N,beta,alpha_list,tw_list[1::2])
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[0::3],l_index,L,N,beta,alpha_list,tw_list[0::3])
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[-2:-1],l_index,L,N,beta,alpha_list,tw_list[-2:-1])
        plot_ggrand_tau_J_tw_ave_over_init(ggrand_tau_J[-1:],l_index,L,N,beta,alpha_list,tw_list[-1:])
    for l_index,term in enumerate(l_S_list): 
        print("for l={}, shape of ggrand_tau_S:".format(l_index))
        print(ggrand_tau_S.shape)
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[:],l_index,L,N,beta,alpha_list,tw_list[:])
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[0::2],l_index,L,N,beta,alpha_list,tw_list[0::2])
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[1::2],l_index,L,N,beta,alpha_list,tw_list[1::2])
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[0::3],l_index,L,N,beta,alpha_list,tw_list[0::3])
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[-2:-1],l_index,L,N,beta,alpha_list,tw_list[-2:-1])
        plot_ggrand_tau_S_tw_ave_over_init(ggrand_tau_S[-1:],l_index,L,N,beta,alpha_list,tw_list[-1:])
