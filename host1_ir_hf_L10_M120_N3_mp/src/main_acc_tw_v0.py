# The function main_acc_tw.py is for calculating the average accuracy for classification over 64 replicas.
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
from utilities import generate_S_in_and_out_2_spin_v3_resp, list_only_naked_dir
from utilities_overlap import overlap_J_hidden, overlap_S, plot_overlap_J_hidden, plot_overlap_S, \
     plot_test_delta_e_bond, plot_test_delta_e_node, plot_test_delta_e_bond_node_hidden 
from utilities_overlap import *

from Network import generate_tw_list, qq, step_list_v2, generate_traj_sampling_num

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
n_replica = 64 # IMPORTANT: FOR  A NEW SYSTEM, PLEASE SET THE n_replica! 


beta = 66.70
sample_index = 2 
L = 10 
M = 10 
N = 3 
N_in = 784
N_out =2 
tot_steps = 1024 
tw = 2 
step = 0

def acc_tw(L=L, M=M, N=N, tot_steps=tot_steps, beta=beta, N_in=N_in, N_out=N_out, tw=tw, sample_index=sample_index, n_replica=n_replica):
    mpl.use('Agg')
    ext_index = 0 
    
    # to find the locations of configurations.
    data_dir = '../data1'
    timestamp_list = list_only_naked_dir(data_dir) # There is only one directory.
    #------------------------------------------------------------------------

    num_hidden_node_layers = L - 1 
    num_hidden_bond_layers = L - 2
    #Determine the total number of degree of freedom 
    num = num_dof(L,M,N,N_in,N_out)

    BIAS = 1
    #tot_steps_ = int(np.log2(tot_steps * num + BIAS)) # Rescale 
    tot_steps_ = generate_traj_sampling_num(tot_steps * num, BIAS, qq) # Rescale 
    print("tot_steps_") 
    print(tot_steps_) 
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

    score = np.zeros((n_replica,tot_steps_))
    acc = np.zeros((n_replica,tot_steps_))
    mean_acc = np.zeros(tot_steps_)
    r_index = 0

    file0 = '/public1/home/sc91981/Yoshino2G3/augmented_mnist_3/M{}/dataset_zeros_m{}_sample{}.csv'.format(M,M,sample_index)
    file1 = '/public1/home/sc91981/Yoshino2G3/augmented_mnist_3/M{}/dataset_ones_m{}_sample{}.csv'.format(M,M,sample_index)
    # Load training data from lib. Note that S_in and S_out are spins (their elments are 1, or -1)
    S_in,S_out = generate_S_in_and_out_2_spin_v3_resp(file0,file1,M,N_in,N_out)
    np.save('S_in_sample{}.npy'.format(sample_index), S_in)
    np.save('S_out_sample{}.npy'.format(sample_index), S_out)

    #print("S_in' s shape:{}".format(S_in.shape))
    #print("S_out' s shape:{}".format(S_out.shape))
    S_out_predict_traj = np.zeros((tot_steps_,M,N_out))

    replicas = str_replica_index_list
    for r_index, str_replicas in enumerate(replicas):
        J_in_traj = np.load('{}/{}/J_in_{:s}_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,N,N_in,beta,tot_steps))
        J_out_traj = np.load('{}/{}/J_out_{:s}_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,N_out,N,beta,tot_steps))
        J_traj = np.load('{}/{}/J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,L,N,beta,tot_steps))
        S_traj = np.zeros((tot_steps_,M,num_hidden_node_layers,N))
        #S_traj = np.load('{}/{}/S_{}_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}_step{:d}.npy'.format(data_dir,stamp,str_replicas,sample_index,tw,L,M,N,beta,tot_steps))
        print("J_in_traj's shape:{}".format(J_in_traj.shape))
        print("J_out_traj's shape:{}".format(J_out_traj.shape))
        print("J_traj's shape:{}".format(J_traj.shape))
        print("S_traj's shape:{}".format(S_traj.shape))
        for step in range(tot_steps_):
            for i in range(N):
                l = 0
                #From S_in to obtain S1
                S_traj[step,:,l,i] = np.sign(np.sum(J_in_traj[step,i,:] * S_in[:,:], axis = 1))
                print("S_TRAJ[{},:,{},{}]= {}".format(step,l,i,S_traj[step,:,l,i]))
            for i in range(N):
                #From S1 to obtain S2, ...,  S_9
                for l in range(1,L-1):
                    S_traj[step,:,l,i] = np.sign(np.sum(J_traj[step,l-1,i,:] * S_traj[step,:,l-1,:], axis = 1))
                    print("S_TRAj[{},:,{},{}]= {}".format(step,l,i,S_traj[step,:,l,i]))
            for i in range(N_out):
                #From S_n to obtain S_out
                l = L-1
                S_out_predict_traj[step,:,i] = np.sign(np.sum(J_out_traj[step,i,:] * S_traj[step,:,l-1,:], axis =1))
                print("S_out_predict_traj[{},:,{}]= {}".format(step,i,S_out_predict_traj[step,:,i]))

            #Calculate accuracy for this kernel.
            print("Predicted all samples: {}\n, {}".format(S_out_predict_traj[step,:], S_out[:]))
            for mu in range(M):
                print("Predicted: {}".format((S_out_predict_traj[step,mu] == S_out[mu]).all()))
                score[r_index,step] += (S_out_predict_traj[step,mu] == S_out[mu]).all()

            acc[r_index,step] = score[r_index,step]/M
            print("ACC for a replica: {}".format(acc[r_index,step]))
            ##Remember: Do not use a function'name as a name of variable
    for step in range(tot_steps_):
        print("acc[{}]= {}".format(step, acc[:,step]))
    mean_acc = np.mean(acc,axis=0)
    print("All ACC over replica for all steps: {}".format(mean_acc))
    

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
    tw_list = generate_tw_list(tot_steps)

    #================================================ 
    #Calculate overlaps
    #Use Multiprocessing to run MC on different cores
    #================================================ 
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count()) 
    print("The computer has " + str(num_cores) + " cores.")
    
    # Define parameters for each process, and store each set of input as a value of a dictionary dic_para.
    dic_para = {}
    for i, tw_term in enumerate(tw_list):
        dic_para["para{}".format(i)] = (L,M,N, tot_steps, beta, N_in, N_out, tw_term, sample_index, n_replica)
    #The number of cores required is the number of tw's 
    for key, value in dic_para.items():
        print("Now start process ({}).".format(key))
        mp.Process(target=acc_tw, args=value).start() #start now
        sleep(1)
