#============
#Super Module
#============
import sys
mf_index = 0
#===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
#===============================================================================
main_dir = '..'
main_dir_local = '/home/gang/github/Yoshino2G3'
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
from utilities import *

from random import choice
import copy
import math
import numpy as np
import tensorflow as tf # Import dataset from tf.keras, instead of from keras directly.
from scipy.stats import norm
import os
import matplotlib.pyplot as plt
from random import randrange
import scipy as sp
from time import time

class HostNetwork:
    def __init__(self,L,M,N,N_in,N_out,tot_steps,timestamp,h):
        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only the part affected by the flip of a SPIN (S) or a shift of
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected."""
        self.L = int(L)
        self.M = int(M)
        self.N = int(N)
        self.tot_steps = int(tot_steps)
        self.timestamp = timestamp
        self.N_in = int(N_in)
        self.N_out = int(N_out)
        self.num_hidden_node_layers = self.L - 1 # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
        self.num_hidden_bond_layers = self.L - 2
        self.BIAS = 1
        self.H = 0 # for storing energy
        self.new_H = 0 # for storing temperay energy when update
        # Energy difference caused by update of sample mu
        self.delta_H= 0

        # Intialize S and J by the array S and J.
        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
        self.S = init_S(self.M,self.num_hidden_node_layers,self.N)
        self.J_hidden = init_J(self.num_hidden_bond_layers,self.N)
        # Define J_in and J_out, etc.
        self.S_in = np.zeros((self.M,self.N_in))  
        self.S_out = np.zeros((self.M,self.N_out))  
        self.J_in = init_J_in(self.N,self.N_in)  
        self.J_out = init_J_out(self.N_out,self.N)
      
        self.r = 0 # the initial gap
        self.r_in = 0 # the initial gap
        self.r_out = 0 # the initial gap

        # Initialize the inner parameters: num_bonds, num_variables
        self.num_variables = 0
        self.num_bonds = 0
        self.num = 0

        self.count_MC_step = 0
        self.h = h # index of host group
        #====
        #TEST
        #====
        print("S:")
        print(self.S)
        print("J_hidden:")
        print(self.J_hidden)
        print("J_in:")
        print(self.J_in)
        print("J_out:")
        print(self.J_out)

    def accept_by_mu_l_n(self,mu,l,n):
        """This accept function is used if S is flipped."""
        self.S[mu,l,n] = self.new_S[mu,l,n]
        self.H = self.H + self.delta_H
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
