# ===============
# Super Module
# ===============
import sys
import torch.profiler

mf_index = 0
# ===============================================================================
# Main dirctories (CHANGE THE DEFINITION OF THE TWO STRING FOR YOUR SIMULATIONS)
# ===============================================================================
main_dir = 'D:/Yoshino2G3--MC/L10M120N3'
main_dir_local = 'D:/Yoshino2G3--MC/L10M120N3'


def determine_module_filename(mf_index):
    if mf_index == 0:

        module_filename = main_dir + '/py_functions/'
    elif mf_index == 1:
        module_filename = main_dir_local + '/py_functions_local/'
    return module_filename


module_filename = determine_module_filename(mf_index)
sys.path.append(module_filename)
# ======
# Module
# ======
from random import choice
import copy
from functools import wraps
import math
import numpy as np
from scipy.stats import norm
import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from random import randrange
import scipy as sp
from time import time
from utilities import (calc_ener, calc_ener_torch, calc_ener_torch_batch, calc_ener_batch, calc_ener_batch_torch,
                       draw_history, random_numbers_with_gap)
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

SAMPLE_START = 1  # SET THIS PARAMETER
SAMPLE_END = 100  # SET THIS PARAMETER
num_cores = 64

num_cores = 64
Bound_tw = 8  # All tw which is smaller than Bound_tw will save seeds for.
l_list = [1, 2, 3, 4, 5, 6, 7, 8]
l_index_list = [0, 1, 2, 3, 4, 5, 6, 7]
l_S_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
l_S_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
init_list = [i for i in range(SAMPLE_START, SAMPLE_END + 1)]  # For M480 tw32, 64, 128, 256, 512, 1024
init_list_all = np.array(range(20 * num_cores))
M_list = [30, 60, 120, 240, 480, 960, 1920]
QS_cutoff = 0.02
QJ_cutoff = 0.02

# The parameter ratio is used for generating the sampling time point.
qq = 5.0
pp = qq + 1
ratio = pp / qq

# general ratio 
step_list_v2 = [int((ratio) ** n) for n in range(200)]
tw_list_test = [0, 1, 2, 3]  # MC steps


def generate_tw_list(tot_steps):
    if tot_steps < 4 + 1:
        tw_list = tw_list_test
    else:
        # tw_list = [1,1,1,1]
        # tw_list = [64,64,64,64]
        # tw_list = [128,128,128,128]
        # tw_list = [256,256,256,256]
        # tw_list = [512,512,512,512]
        tw_list = [1024, 1024, 1024, 1024]
    return tw_list


time_dict = {}


def timethis(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time()
        result = fun(*args, **kwargs)
        end = time()
        # print(fun.__name__, end - start)
        if fun.__name__ in time_dict.keys():
            time_dict[fun.__name__] += end - start
        else:
            time_dict[fun.__name__] = end - start
        return result

    return wrapper


def generate_traj_sampling_num(num_steps, BIAS, qq):
    try:
        if qq == 1:
            sampled_steps = int(np.log2(num_steps + BIAS))
        elif qq > 0 and qq != 1:
            sampled_steps = int(np.log(num_steps + BIAS) / np.log((qq + 1) / qq))
    except:
        print("The parameter qq should be positive.")
    print("sampled_steps")
    print(sampled_steps)
    return sampled_steps


class Network:
    def __init__(self, sample_index, tw, L, M, N, N_in, N_out, tot_steps, beta, timestamp, h, qq=2):
        """Since Yoshino_3.0, when update the energy, we do not calculate all the gaps, but only calculate the part affected by the flip of a SPIN (S)  or a shift of
           a BOND (J). This will accelerate the calculation by hundreds of times. (2) Besides, we note that we do NOT need to define a functon: remain(), which records
           the new MC steps' S, J and H, even though one MC move is rejected. 
           h : index of the host machine.
        """
        # Parameters used in the host machine (No. 0)
        self.r_out = None
        self.r_in = None
        self.r_hidden = None
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
        self.num_hidden_node_layers = self.L - 1  # After distingush num_hidden_node_layers and num_hidden_bond_layers, then I found the clue is much clear!
        self.num_hidden_bond_layers = self.L - 2
        # Define new parameters; T (technically required, to save memory)
        self.BIAS = 1  # For avoiding x IS NOT EQUAL TO ZERO in log2(x)
        self.BIAS2 = 5  # For obtaing long enough list list_k.
        # self.T = int(np.log2(self.tot_steps+self.BIAS)) # we keep the initial state in the first step
        # self.T = generate_traj_sampling_num(self.tot_steps,self.BIAS,qq)
        self.H = 0  # for storing energy
        self.new_H = 0  # for storing temperay energy when update

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'

        # Energy difference caused by update of sample mu
        self.delta_H = 0
        self.H_history = []

        # Intialize S,J_hidden,J_in and J_out by the saved arrays, which are saved from the host machine. 
        # We use S, instead of S_hidden, because S[0], S[L-2] still have interaction to J_in and J_out. Ref. Yoshino eq (8).
        data_dir = 'D:/Yoshino2G3--MC/L10M120N3/data'
        str_timestamp = str(timestamp)
        self.S = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, L, M, N, beta))
        self.J_hidden = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, L, N, beta))
        self.J_in = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, N, N_in, beta))
        self.J_out = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, N_out, N, beta))

        # =========================================================================================================
        # The following copy operations must be done in this __init__() function.
        # NOTE that one can NOT delay the COPY operations, for example, just putting it in the function flip_spin(), etc.
        # However, we ALSO NEED the COPY operation in functions flip_spin() and shift_bond_in(), etc.
        # =========================================================================================================
        self.new_S = copy.copy(self.S)  # for storing temperay array when update
        self.new_J_hidden = copy.copy(self.J_hidden)  # for storing temperay array when update
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
        # self.T_2 = int(np.log2(self.tot_steps * self.num + self.BIAS)) # we keep the initial state in the first step
        # ========================================================================================================================================
        # The arrays for storing MC trajectories of S, J and H (hyperfine)
        # Note that we DO NOT nedd to define arrays self.S_in_traj_hyperfine, or S_out_traj_hyperfine, because self.S_in and self.S_out are fixed.
        self.J_hidden_traj_hyperfine = np.zeros((self.T_2, self.num_hidden_bond_layers, self.N, self.N),
                                                dtype='float32')
        self.S_traj_hyperfine = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
        self.H_hidden_traj_hyperfine = np.zeros(self.T_2, dtype='float32')
        self.J_in_traj_hyperfine = np.zeros((self.T_2, self.N, self.N_in), dtype='float32')
        self.J_out_traj_hyperfine = np.zeros((self.T_2, self.N_out, self.N), dtype='float32')
        # ========================================================================================================================================

        # DEFINE SOME PARAMETERS
        self.EPS = 0.000001
        self.RAT = 0.1  # r: Yoshino2019 Eq(35), RAT=0.1
        self.RESCALE_J = 1.0 / np.sqrt(1 + self.RAT ** 2)
        self.SQRT_N = np.sqrt(self.N)
        self.SQRT_N_IN = np.sqrt(self.N_in)
        self.RAT_in = self.RAT * self.N_in / self.N  # r: Yoshino2019 Eq(35), RAT=0.1, we rescale it for input bond layer.
        self.RAT_out = self.RAT * self.N_out / self.N  # r: Yoshino2019 Eq(35), RAT=0.1, we rescale it for output bond layer.
        self.PROB = self.num_variables / self.num
        self.cutoff1 = self.N * self.N_in
        self.cutoff2 = self.cutoff1 + self.num_hidden_bond_layers * self.SQ_N

        # EXAMPLE: list_k = [0,1,2,3,4,8,16,64,256,1024,4096,16384,65536] # list_k is used for waiting time.
        temp_set1 = set([2 ** (i) for i in range(int(np.log2(self.tot_steps + self.BIAS) + self.BIAS2))])
        temp_set2 = set([i for i in range(1,
                                          Bound_tw)])  # For small time step, we save seeds for each step (IMPORTANT: tw=0 should be excluded).
        temp_list = list(temp_set1.union(temp_set2))
        temp_list.sort()  # Sort the list. This is important.
        # HARD version
        self.list_k_4_hyperfine = [2 ** (i) for i in
                                   range(int(np.log2(self.tot_steps * self.num + self.BIAS) + self.BIAS2))]
        # SIMPLE version
        self.list_k = [0, 8, 16, 32, 64, 128, 256, 512, 1024]

        # For saving S and J accurately, I need know the exact update steps.
        self.update_index = 0

        # Define J_in and J_out
        self.S_in = np.zeros((self.M, self.N_in))
        self.S_out = np.zeros((self.M, self.N_out))

        # ================================
        self.S = torch.tensor(self.S, device=self.device)
        self.J_hidden = torch.tensor(self.J_hidden, device=self.device)
        self.J_in = torch.tensor(self.J_in, device=self.device)
        self.J_out = torch.tensor(self.J_out, device=self.device)
        self.S_in = torch.tensor(self.S_in, device=self.device)
        self.S_out = torch.tensor(self.S_out, device=self.device)
        # ==============================
        # For testing
        self.Delta_E_node_hidden = []
        self.Delta_E_node_in = []
        self.Delta_E_node_out = []
        self.Delta_E_bond_hidden = []
        self.Delta_E_bond_in = []
        self.Delta_E_bond_out = []

    def gap_in_init(self):
        '''Ref: Yoshino2019, eqn (31b):'''
        r_in = np.zeros((self.M, self.N), dtype='float32')

        # Precompute SQRT_N_IN to avoid repeated calculations
        sqrt_n_in_inv = 1.0 / self.SQRT_N_IN

        for mu in range(self.M):
            # Compute the dot product for the current mu once, saving it for reuse
            J_in_S_in_product = np.dot(self.J_in, self.S_in[mu, :]) * sqrt_n_in_inv

            # Use the precomputed result in the calculation for r_in
            r_in[mu, :] = J_in_S_in_product * self.S[mu, 0, :]

        return r_in

    def gap_in_init_torch(self):
        # Preallocate r_in as a PyTorch tensor
        self.r_in = torch.zeros((self.M, self.N), dtype=torch.float32, device=self.device)

        # Precompute SQRT_N_IN to avoid repeated calculations
        sqrt_n_in_inv = 1.0 / self.SQRT_N_IN

        for mu in range(self.M):
            # Compute the dot product for the current mu once, saving it for reuse
            J_in_S_in_product = torch.matmul(self.J_in, self.S_in[mu, :]) * sqrt_n_in_inv

            # Use the precomputed result in the calculation for r_in
            self.r_in[mu, :] = J_in_S_in_product * self.S[mu, 0, :]

        return self.r_in

    def gap_hidden_init(self):
        '''Ref: Yoshino2019, eqn (31b)'''
        L = self.L
        M = self.M
        N = self.N
        r_hidden = np.zeros((M, L, N), dtype='float32')
        for mu in range(M):
            for l in range(self.num_hidden_bond_layers):  # l = 2,...,L-1
                S_current = self.S[mu, l, :]  # The current layer's spin configuration
                S_next = self.S[mu, l + 1, :]  # The next layer's spin configuration

                # Use dot product for vectorized computation
                J_dot_S_current = np.dot(self.J_hidden[l, :, :], S_current)  # Vectorized summation

                # Normalize and compute r_hidden values
                r_hidden[mu, l, :] = (J_dot_S_current / self.SQRT_N) * S_next

        return r_hidden

    def gap_hidden_init_torch(self):
        L = self.L
        M = self.M
        N = self.N
        # Initialize r_hidden as a tensor of zeros with the correct dtype and device
        self.r_hidden = torch.zeros((M, L, N), dtype=torch.float32,
                                    device=self.device)  # Ensure the correct device (e.g., GPU)

        for mu in range(M):
            for l in range(self.num_hidden_bond_layers):  # l = 2,...,L-1
                # Ensure that S_current and S_next are float32 tensors (if not already)
                S_current = self.S[mu, l, :]  # Current layer's spin configuration
                S_next = self.S[mu, l + 1, :]  # Next layer's spin configuration

                # Ensure that self.J_hidden[l, :, :] is also float32
                J_dot_S_current = torch.matmul(self.J_hidden[l, :, :], S_current)  # Matrix multiplication

                # Normalize and compute r_hidden values
                self.r_hidden[mu, l, :] = (J_dot_S_current / self.SQRT_N) * S_next

        return self.r_hidden

    def gap_out_init(self):
        M = self.M
        N_out = self.N_out
        r_out = np.zeros((M, N_out), dtype='float32')

        # Precompute SQRT_N to avoid repeated calculation
        sqrt_n_inv = 1.0 / self.SQRT_N

        for mu in range(M):
            # Compute the dot product once per mu
            J_out_S_last_layer_product = np.dot(self.J_out, self.S[mu, -1, :]) * sqrt_n_inv

            # Update r_out for all n2 values using the precomputed value
            r_out[mu, :] = J_out_S_last_layer_product * self.S_out[mu, :]

        return r_out

    def gap_out_init_torch(self):
        # Preallocate r_out as a PyTorch tensor
        self.r_out = torch.zeros((self.M, self.N_out), dtype=torch.float32, device=self.device)

        # Precompute SQRT_N to avoid repeated calculation
        sqrt_n_inv = 1.0 / self.SQRT_N

        for mu in range(self.M):
            # Compute the dot product once per mu
            J_out_S_last_layer_product = torch.matmul(self.J_out, self.S[mu, -1, :]) * sqrt_n_inv

            # Update r_out for all n2 values using the precomputed value
            self.r_out[mu, :] = J_out_S_last_layer_product * self.S_out[mu, :]

        return self.r_out

    def flip_spin(self, mu, l, n):
        '''flip_spin() will flip S at a given index tuple (l,mu,n). We add l,mu,n as parameters, for parallel programming. Note: any spin can be update except the input/output.'''
        # Update self.new_S
        self.new_S = copy.copy(self.S)  # for storing temperay array when update
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
        self.S_in = np.load('../data/host/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(M, N_in, beta))
        self.S_out = np.load('../data/host/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(M, N_out, beta))
        # =========================================================================================================================================
        # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of the host machine is the same as the guest machines.
        self.J_in = np.load(
            '../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init, tw, N, N_in, beta))
        self.J_out = np.load(
            '../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init, tw, N_out, N,
                                                                                               beta))
        self.J_hidden = np.load(
            '../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init, tw, L, N, beta))
        self.S = np.load(
            '../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init, tw, L, M, N, beta))

    def load_S_and_J_2g(self):
        beta = self.beta
        init = self.init
        L = self.L
        M = self.M
        N = self.N
        N_in = self.N_in
        N_out = self.N_out
        tw = self.tw
        # h = int(init/num_cores) + 1
        self.S_in = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(self.timestamp,
                                                                                                  M,
                                                                                                  N_in,
                                                                                                  beta))
        self.S_out = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(self.timestamp,
                                                                                                    M,
                                                                                                    N_out,
                                                                                                    beta))
        # =========================================================================================================================================
        # Import the nodes and bonds at t=tw. Motivation: make sure the initial configuraton of the host machine is the same as the guest machines.
        self.J_in = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(
                self.timestamp, init, tw, N, N_in, beta))
        self.J_out = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(
                self.timestamp, init, tw, N_out, N, beta))
        self.J_hidden = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(
                self.timestamp, init, tw, L, N, beta))
        self.S = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(
                self.timestamp, init, tw, L, M, N, beta))

        # ================================
        self.S = torch.tensor(self.S, dtype=torch.float32, device=self.device)
        self.J_hidden = torch.tensor(self.J_hidden, dtype=torch.float32, device=self.device)
        self.J_in = torch.tensor(self.J_in, dtype=torch.float32, device=self.device)
        self.J_out = torch.tensor(self.J_out, dtype=torch.float32, device=self.device)
        self.S_in = torch.tensor(self.S_in, dtype=torch.float32, device=self.device)
        self.S_out = torch.tensor(self.S_out, dtype=torch.float32, device=self.device)

    def random_update_S(self):
        # Const.s
        EPS = self.EPS
        rand2 = np.random.random(1)

        mu, l, n = randrange(self.M), randrange(self.num_hidden_node_layers), randrange(self.N)
        # self.flip_spin(mu, l, n)
        if l == 0:
            self.decision_node_in_by_mu_n(mu, n, EPS, rand2)
        elif l == self.num_hidden_node_layers - 1:
            self.decision_node_out_by_mu_n(mu, n, EPS, rand2)
        else:
            self.decision_by_mu_l_n(mu, l, n, EPS, rand2)

    def random_update_S_batch(self, num_updates):
        # 批量随机生成 mu, l, n
        mu_vals = np.random.randint(0, self.M, num_updates)
        n_vals = np.random.randint(0, self.N, num_updates)

        l_vals = random_numbers_with_gap(0, self.num_hidden_node_layers, num_updates, 1)
        rand_vals = np.random.random(num_updates)

        # 逐层判断并批量调用对应的更新逻辑
        mask_in = l_vals == 0
        mask_out = l_vals == self.num_hidden_node_layers - 1

        # 分别处理输入层、中间层、输出层
        self.decision_node_in_by_mu_n_batch(mu_vals[mask_in], n_vals[mask_in], self.EPS, rand_vals[mask_in])
        self.decision_node_out_by_mu_n_batch(mu_vals[mask_out], n_vals[mask_out], self.EPS, rand_vals[mask_out])
        self.decision_by_mu_l_n_batch(mu_vals[~(mask_in | mask_out)], l_vals[~(mask_in | mask_out)],
                                      n_vals[~(mask_in | mask_out)], self.EPS, rand_vals[~(mask_in | mask_out)])

    def random_update_S_batch_torch(self, num_updates):
        # 批量随机生成 mu, l, n
        mu_vals = torch.randint(0, self.M, (num_updates,), dtype=torch.long, device=self.device)  # 使用 torch.randint 生成整数
        n_vals = torch.randint(0, self.N, (num_updates,), dtype=torch.long, device=self.device)

        # 使用自定义的随机数生成带间隔的 l_vals，这部分需要改成适配 PyTorch 的代码
        l_vals = random_numbers_with_gap(0, self.num_hidden_node_layers, num_updates, 1)
        l_vals = torch.from_numpy(l_vals).to(self.device)

        rand_vals = torch.rand(num_updates, device=self.device)  # 使用 torch.rand 生成随机数

        # 逐层判断并批量调用对应的更新逻辑
        mask_in = l_vals == 0  # 输入层
        mask_out = l_vals == self.num_hidden_node_layers - 1  # 输出层

        # 分别处理输入层、中间层、输出层
        self.decision_node_in_by_mu_n_batch_torch(mu_vals[mask_in], n_vals[mask_in], self.EPS, rand_vals[mask_in])
        self.decision_node_out_by_mu_n_batch_torch(mu_vals[mask_out], n_vals[mask_out], self.EPS, rand_vals[mask_out])
        self.decision_by_mu_l_n_batch_torch(mu_vals[~(mask_in | mask_out)], l_vals[~(mask_in | mask_out)],
                                            n_vals[~(mask_in | mask_out)], self.EPS, rand_vals[~(mask_in | mask_out)])
        # with ThreadPoolExecutor() as executor:
        #     futures = []
        #     futures.append(executor.submit(self.decision_node_in_by_mu_n_batch_torch, mu_vals[mask_in], n_vals[mask_in],
        #                                    self.EPS, rand_vals[mask_in]))
        #     futures.append(executor.submit(self.decision_node_out_by_mu_n_batch_torch, mu_vals[mask_out], n_vals[mask_out],
        #                                    self.EPS, rand_vals[mask_out]))
        #     futures.append(executor.submit(self.decision_by_mu_l_n_batch_torch, mu_vals[~(mask_in | mask_out)],
        #                                    l_vals[~(mask_in | mask_out)],
        #                                    n_vals[~(mask_in | mask_out)], self.EPS, rand_vals[~(mask_in | mask_out)]))
        #     for future in futures:
        #         future.result()


    def random_update_J(self):
        """Method 1"""
        # Const.s
        EPS = self.EPS
        x = np.random.normal(loc=0, scale=1.0, size=None)
        x = round(x, 10)
        cutoff1 = self.cutoff1
        cutoff2 = self.cutoff2
        P1 = cutoff1 / self.num_bonds
        P2 = cutoff2 / self.num_bonds
        rand1, rand2 = np.random.random(1), np.random.random(1)

        if rand1 < P1:
            n2, n1 = randrange(self.N), randrange(self.N_in)
            # Update J_in[n2,n1]
            self.shift_bond_in_v2(n2, n1, x)  # Note1/2: shift_bond_in_v2() will give normalized result.
            self.decision_bond_in_by_n2_n1(n2, n1, EPS, rand2)
        elif rand1 > P2:
            n2, n1 = randrange(self.N_out), randrange(self.N)
            self.shift_bond_out_v2(n2, n1, x)
            self.decision_bond_out_by_n2_n1(n2, n1, EPS, rand2)
        else:
            l, n2, n1 = randrange(self.num_hidden_bond_layers), randrange(self.N), randrange(self.N)
            self.shift_bond_hidden_v2(l, n2, n1, x)
            self.decision_bond_hidden_by_l_n2_n1(l, n2, n1, EPS, rand2)

    def update_J(self, J, n2, n1, x, EPS, rand, sqrt_num, pre_layer, next_layer):
        """通用更新函数，适用于 `J_in`, `J_out`, `J_hidden` 更新"""
        new_J = np.copy(J[n2])  # 为每个更新创建临时副本
        # 更新元素值
        new_J[n1] = (J[n2, n1] + x * self.RAT) * self.RESCALE_J

        # 重缩放
        SCALE = np.sqrt(self.N / np.sum(new_J * new_J))
        new_J *= SCALE

        a1 = self.part_gap_common_shift(sqrt_num, J[n2], pre_layer, next_layer)
        a2 = self.part_gap_common_shift(sqrt_num, new_J, pre_layer, next_layer)

        delta_e = calc_ener(a2) - calc_ener(a1)

        # 接受/拒绝更新
        if delta_e < EPS or rand < np.exp(-delta_e * self.beta):
            J[n2] = new_J
            self.H = self.H + delta_e

    def update_J_torch(self, J, n2, n1, x, EPS, rand, sqrt_num, pre_layer, next_layer):
        """Torch版本通用更新函数，适用于 `J_in`, `J_out`, `J_hidden` 更新"""
        # 创建 J[n2] 的副本（保持梯度追踪）
        new_J = J[n2].clone()

        # 更新元素值
        new_J[n1] = (J[n2, n1] + x * self.RAT) * self.RESCALE_J

        # 重缩放
        scale = torch.sqrt(self.N / torch.sum(new_J * new_J))
        new_J *= scale

        # 计算更新前后的能量
        a1 = self.part_gap_common_shift_torch(sqrt_num, J[n2], pre_layer, next_layer)
        a2 = self.part_gap_common_shift_torch(sqrt_num, new_J, pre_layer, next_layer)

        delta_e = calc_ener_torch(a2, self.device) - calc_ener_torch(a1, self.device)

        # 接受/拒绝更新
        accept_condition = (delta_e < EPS) | (rand < torch.exp(-delta_e * self.beta))
        if accept_condition:
            J[n2] = new_J
            self.H = self.H + delta_e

    def update_J_batch_torch(self, J, n2, n1, x, EPS, rand, sqrt_num, pre_layer, next_layer):
        batch_size = n2.shape[0]
        new_J = J[n2].clone()  # 使用 clone() 来避免直接修改原始 tensor

        # 更新指定列的值
        new_J[torch.arange(batch_size), n1] = (J[n2, n1] + x * self.RAT) * self.RESCALE_J

        # 重缩放每一行
        norm = torch.sqrt(torch.sum(new_J ** 2, dim=1, keepdim=True))  # shape = (batch_size, 1)
        SCALE = torch.sqrt(self.N / norm)  # shape = (batch_size, 1)
        new_J = new_J * SCALE  # 按行缩放

        # 计算 a1 和 a2
        a1 = self.part_gap_common_shift_torch_batch(sqrt_num, J[n2], pre_layer, next_layer)  # shape = (batch_size, ...)
        a2 = self.part_gap_common_shift_torch_batch(sqrt_num, new_J, pre_layer, next_layer)  # shape = (batch_size, ...)

        # 计算能量变化 delta_e
        delta_e = calc_ener_torch_batch(a2, self.device) - calc_ener_torch_batch(a1, self.device)  # shape = (batch_size,)

        # 判断接受/拒绝更新
        accept_mask = (delta_e < EPS) | (rand < torch.exp(-delta_e * self.beta))  # shape = (batch_size,)
        accept_mask = accept_mask.to(self.device)

        # 更新 J 和能量
        J[n2[accept_mask]] = new_J[accept_mask]  # 只更新接受的行
        self.H += delta_e[accept_mask].sum()  # 更新总能量
        # for i in range(batch_size):
        #     self.update_J_torch(J, n2[i], n1[i], x[i], EPS, rand[i], sqrt_num, pre_layer, next_layer[:, i])

    def update_tasks(self, n2, update_list, J, EPS, sqrt_num, pre_layer, next_layer):
        for n1, x, rand in update_list:
            self.update_J_torch(J, n2, n1, x, EPS, rand, sqrt_num, pre_layer, next_layer)

    @timethis
    def random_update_J_batch(self, batch_size):
        """Batch处理版本的更新函数"""
        EPS = self.EPS
        cutoff1 = self.cutoff1
        cutoff2 = self.cutoff2
        P1 = cutoff1 / self.num_bonds
        P2 = cutoff2 / self.num_bonds

        # 生成批量随机数
        x_batch = np.random.normal(loc=0, scale=1.0, size=batch_size)
        rand_batch = np.random.random(batch_size)

        # 分组存储任务
        case1_tasks = defaultdict(list)  # Case 1: 按 n2 分组
        case2_tasks = defaultdict(list)  # Case 2: 按 n2 分组
        case3_tasks = defaultdict(lambda: defaultdict(list))  # Case 3: 按 (l, n2) 分组

        # 为每个 batch 生成 J 更新
        for i in range(batch_size):
            x = x_batch[i]
            rand = rand_batch[i]

            if rand < P1:
                # Case 1: 更新 J_in
                n2, n1 = randrange(self.N), randrange(self.N_in)
                case1_tasks[n2].append((n1, x, rand))
                self.update_J(self.J_in, n2, n1, x, EPS, rand, self.SQRT_N_IN, self.S_in, self.S[:, 0, n2])
            elif rand > P2:
                # Case 2: 更新 J_out
                n2, n1 = randrange(self.N_out), randrange(self.N)
                case2_tasks[n2].append((n1, x, rand))
                self.update_J(self.J_out, n2, n1, x, EPS, rand, self.SQRT_N, self.S[:, -1, :], self.S_out[:, n2])
            else:
                # Case 3: 更新 J_hidden
                l, n2, n1 = randrange(self.num_hidden_bond_layers), randrange(self.N), randrange(self.N)
                case3_tasks[l][n2].append((n1, x, rand))
                self.update_J(self.J_hidden[l], n2, n1, x, EPS, rand, self.SQRT_N, self.S[:, l - 1, :],
                              self.S[:, l, n2])

        # with ThreadPoolExecutor() as executor:
        #     # 提交每个 n2 的任务组到线程池
        #     futures = []
        #     for n2, updates in case1_tasks.items():
        #         futures.append(executor.submit(self.update_tasks, n2, updates, self.J_in, EPS, self.SQRT_N_IN,
        #                                        self.S_in, self.S[:, 0, n2]))
        #
        #     for n2, updates in case2_tasks.items():
        #         futures.append(executor.submit(self.update_tasks, n2, updates, self.J_out, EPS, self.SQRT_N,
        #                                        self.S[:, -1, :], self.S_out[:, n2]))
        #
        #     for l, n2_tasks in case3_tasks.items():
        #         for n2, updates in n2_tasks.items():
        #             futures.append(executor.submit(self.update_tasks, n2, updates, self.J_hidden[l], EPS, self.SQRT_N,
        #                                            self.S[:, l, :], self.S[:, l+1, n2]))
        #
        #     # 等待所有任务完成
        #     for future in futures:
        #         future.result()

    @timethis
    def random_update_J_batch_torch(self, batch_size):
        EPS = self.EPS
        cutoff1 = self.cutoff1
        cutoff2 = self.cutoff2
        P1 = cutoff1 / self.num_bonds
        P2 = cutoff2 / self.num_bonds

        device = self.device  # 假设 device 是类属性，表示 'cuda' 或 'cpu'

        # 生成批量随机数
        x_batch = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=device)
        rand_batch = torch.rand(batch_size, device=device)
        # 创建 mask 确定更新条件
        mask_case1 = rand_batch < P1
        mask_case2 = rand_batch > P2
        mask_case3 = ~(mask_case1 | mask_case2)  # 剩下的情况是 Case 3

        # Case 1: 更新 J_in
        if mask_case1.any():
            true_count = mask_case1.sum().item()
            n1_case1 = torch.randint(0, self.N_in, (true_count,), device=device)
            x_selected = x_batch[mask_case1]
            rand_selected = rand_batch[mask_case1]
            epoch = true_count // self.N
            n2 = torch.arange(self.N, device=device)
            for i in range(epoch):
                n1_for_n2 = n1_case1[i * self.N: i * self.N + self.N]
                x_for_n2 = x_selected[i * self.N: i * self.N + self.N]
                rand_for_n2 = rand_selected[i * self.N: i * self.N + self.N]
                self.update_J_batch_torch(self.J_in, n2, n1_for_n2, x_for_n2, EPS, rand_for_n2,
                                    self.SQRT_N_IN, self.S_in, self.S[:, 0, n2])
            remain = true_count % self.N
            if remain > 0:
                n2 = torch.randperm(self.N)[:remain].to(device)
                n1_for_n2 = n1_case1[-remain:]
                x_for_n2 = x_selected[-remain:]
                rand_for_n2 = rand_selected[-remain:]
                self.update_J_batch_torch(self.J_in, n2, n1_for_n2, x_for_n2, EPS, rand_for_n2,
                                    self.SQRT_N_IN, self.S_in, self.S[:, 0, n2])

        # Case 2: 更新 J_out
        if mask_case2.any():
            true_count = mask_case2.sum().item()
            n1_case2 = torch.randint(0, self.N, (true_count,), device=device)
            x_selected = x_batch[mask_case2]
            rand_selected = rand_batch[mask_case2]
            epoch = true_count // self.N_out
            n2 = torch.arange(self.N_out, device=device)
            for i in range(epoch):
                n1_for_n2 = n1_case2[i * self.N_out: i * self.N_out + self.N_out]
                x_for_n2 = x_selected[i * self.N_out: i * self.N_out + self.N_out]
                rand_for_n2 = rand_selected[i * self.N_out: i * self.N_out + self.N_out]
                self.update_J_batch_torch(self.J_out, n2, n1_for_n2, x_for_n2, EPS, rand_for_n2,
                                    self.SQRT_N, self.S[:, -1, :], self.S_out[:, n2])
            remain = true_count % self.N_out
            if remain > 0:
                n2 = torch.randperm(self.N_out)[:remain].to(device)
                n1_for_n2 = n1_case2[-remain:]
                x_for_n2 = x_selected[-remain:]
                rand_for_n2 = rand_selected[-remain:]
                self.update_J_batch_torch(self.J_out, n2, n1_for_n2, x_for_n2, EPS, rand_for_n2,
                                    self.SQRT_N, self.S[:, -1, :], self.S_out[:, n2])

        # Case 3: 更新 J_hidden
        if mask_case3.any():
            true_count = mask_case3.sum().item()
            n1_case3 = torch.randint(0, self.N, (true_count,), device=device)
            x_selected = x_batch[mask_case3]
            rand_selected = rand_batch[mask_case3]
            n2 = torch.arange(self.N, device=device)
            batch_size = self.num_hidden_bond_layers * self.N
            epoch = true_count // batch_size
            for i in range(epoch):
                for l_index in range(self.num_hidden_bond_layers):
                    begin_index = i * batch_size + l_index * self.N
                    n1_for_l_n2 = n1_case3[begin_index: begin_index + self.N]
                    x_for_l_n2 = x_selected[begin_index: begin_index + self.N]
                    rand_for_l_n2 = rand_selected[begin_index: begin_index + self.N]
                    self.update_J_batch_torch(self.J_hidden[l_index], n2, n1_for_l_n2, x_for_l_n2, EPS, rand_for_l_n2,
                                        self.SQRT_N, self.S[:, l_index, :], self.S[:, l_index+1, n2])

            remain = true_count % batch_size
            if remain > 0:
                epoch = remain // self.N
                l_range = torch.randperm(self.num_hidden_bond_layers)[:epoch]
                for index, l_index in enumerate(l_range):
                    begin_index = true_count - remain + index * self.N
                    n1_for_l_n2 = n1_case3[begin_index: begin_index + self.N]
                    x_for_l_n2 = x_selected[begin_index: begin_index + self.N]
                    rand_for_l_n2 = rand_selected[begin_index: begin_index + self.N]
                    self.update_J_batch_torch(self.J_hidden[l_index], n2, n1_for_l_n2, x_for_l_n2, EPS, rand_for_l_n2,
                                        self.SQRT_N, self.S[:, l_index, :], self.S[:, l_index+1, n2])

                remain = remain % self.N
                if remain > 0:
                    l_index = torch.randint(0, self.num_hidden_bond_layers, (1,), device=device).item()
                    n2 = torch.randperm(self.N)[:remain].to(device)
                    n1_for_l_n2 = n1_case3[-remain:]
                    x_for_l_n2 = x_selected[-remain:]
                    rand_for_l_n2 = rand_selected[-remain:]
                    self.update_J_batch_torch(self.J_hidden[l_index], n2, n1_for_l_n2, x_for_l_n2, EPS, rand_for_l_n2,
                                        self.SQRT_N, self.S[:, l_index, :], self.S[:, l_index+1, n2])



    @timethis
    def random_update_J_batch_torch1(self, batch_size):
        EPS = self.EPS
        cutoff1 = self.cutoff1
        cutoff2 = self.cutoff2
        P1 = cutoff1 / self.num_bonds
        P2 = cutoff2 / self.num_bonds

        # 生成批量随机数
        x_batch = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device=self.device)
        rand_batch = torch.rand(batch_size, device=self.device)

        # 分组存储任务
        case1_tasks = defaultdict(list)  # Case 1: 按 n2 分组
        case2_tasks = defaultdict(list)  # Case 2: 按 n2 分组
        case3_tasks = defaultdict(lambda: defaultdict(list))  # Case 3: 按 (l, n2) 分组

       # 为每个 batch 生成 J 更新
        for i in range(batch_size):
            x = x_batch[i]
            rand = rand_batch[i]

            if rand < P1:
                # Case 1: 更新 J_in
                n2, n1 = randrange(self.N), randrange(self.N_in)
                case1_tasks[n2].append((n1, x, rand))
                # self.update_J(self.J_in, n2, n1, x, EPS, rand, self.SQRT_N_IN, self.S_in, self.S[:, 0, n2])
            elif rand > P2:
                # Case 2: 更新 J_out
                n2, n1 = randrange(self.N_out), randrange(self.N)
                case2_tasks[n2].append((n1, x, rand))
                # self.update_J(self.J_out, n2, n1, x, EPS, rand, self.SQRT_N, self.S[:, -1, :], self.S_out[:, n2])
            else:
                # Case 3: 更新 J_hidden
                l, n2, n1 = randrange(self.num_hidden_bond_layers), randrange(self.N), randrange(self.N)
                case3_tasks[l][n2].append((n1, x, rand))
                # self.update_J(self.J_hidden[l], n2, n1, x, EPS, rand, self.SQRT_N, self.S[:, l - 1, :],
                #               self.S[:, l, n2])

        with ThreadPoolExecutor() as executor:
            # 提交每个 n2 的任务组到线程池
            futures = []
            for n2, updates in case1_tasks.items():
                futures.append(executor.submit(self.update_tasks, n2, updates, self.J_in, EPS, self.SQRT_N_IN,
                                               self.S_in, self.S[:, 0, n2]))

            for n2, updates in case2_tasks.items():
                futures.append(executor.submit(self.update_tasks, n2, updates, self.J_out, EPS, self.SQRT_N,
                                               self.S[:, -1, :], self.S_out[:, n2]))

            for l, n2_tasks in case3_tasks.items():
                for n2, updates in n2_tasks.items():
                    futures.append(executor.submit(self.update_tasks, n2, updates, self.J_hidden[l], EPS, self.SQRT_N,
                                                   self.S[:, l, :], self.S[:, l + 1, n2]))

            # 等待所有任务完成
            for future in futures:
                future.result()

                
    def set_vars(self):
        """ Define some variables. 
        Ref: He Yujian's book, Fig. 3.2, m-layer network; Yoshino2020, Fig.1. L-layer network.
        We ASSUME that each hidden layer has N neurons.
        """
        self.gap_hidden_init_torch()  # The initial gap
        self.gap_in_init_torch()  # The initial gap
        self.gap_out_init_torch()  # The initial gap

        self.S_traj_hyperfine[0, :, :, :] = self.S.cpu().numpy()
        self.J_hidden_traj_hyperfine[0, :, :, :] = self.J_hidden.cpu().numpy()
        self.J_in_traj_hyperfine[0, :, :] = self.J_in.cpu().numpy()  # The shape of J_in : (N, N_in)
        self.J_out_traj_hyperfine[0, :, :] = self.J_out.cpu().numpy()  # The shape of o.J_out:  (N_out, N)

        self.H_hidden = calc_ener_torch(self.r_hidden, self.device)  # The energy
        self.H_in = calc_ener_torch(self.r_in, self.device)  # The energy
        self.H_out = calc_ener_torch(self.r_out, self.device)  # The energy
        # self.H_hidden_traj_hyperfine[1] = self.H_hidden  # H_traj[0] will be neglected

    def shift_bond_hidden(self, l, n2, n1, x):
        '''shift_bond_hidden() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming.
           In shift_bond_hidden(), no rescaling (normalization) is done before calculating the energy difference.
           But, rescale_bond_hidden() has to be called after accept.
        '''
        self.new_J_hidden[l][n2][n1] = (self.J_hidden[l][n2][n1] + x * self.RAT) * self.RESCALE_J

    def rescale_bond_hidden(self, l, n2):
        # rescaling 
        N = self.N
        t = self.new_J_hidden[l][n2]
        N_prim = np.sum(t * t)
        SCALE = np.sqrt(N / N_prim)
        self.new_J_hidden[l][n2] = self.new_J_hidden[l][n2] * SCALE

    def shift_bond_hidden_v2(self, l, n2, n1, x):
        '''shift_bond_hidden() will shift the element of J with a given index to another value. We add l,n2,n1 as parameters, for parallel programming.
           shift_bond_hidden_v2() rescals the synaptic weights before calculating the energy differnce.'''
        self.new_J_hidden = copy.copy(self.J_hidden)  # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_hidden[l][n2][n1] = (self.J_hidden[l][n2][n1] + x * self.RAT) * self.RESCALE_J
        # rescaling 
        self.rescale_bond_hidden(l, n2)

    def shift_bond_in(self, n2, n1, x):
        '''shift_bond_in() will shift the element of J_in with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           In shift_bond_in(), No normalization is done before calculating the energy difference.
           But, rescale_bond_in() has to be called after accept.
        '''
        self.new_J_in = copy.copy(self.J_in)  # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_in[n2, n1] = (self.J_in[n2, n1] + x * self.RAT) * self.RESCALE_J

    def rescale_bond_in(self, n2):
        # Access the relevant data once
        J_in_n2 = self.new_J_in[n2]

        # Calculate the scale factor
        SCALE = np.sqrt(self.N_in / np.sum(J_in_n2 ** 2))

        # Rescale in-place
        J_in_n2 *= SCALE
        self.new_J_in[n2] = J_in_n2

    def shift_bond_in_v2(self, n2, n1, x):
        '''shift_bond_in() will shift the element of J_in with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           shift_bond_in_v2() rescals the synaptic weights before calculating the energy differnce.'''
        self.new_J_in = copy.copy(self.J_in)  # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_in[n2, n1] = (self.J_in[n2, n1] + x * self.RAT) * self.RESCALE_J
        # step2:rescaling 
        self.rescale_bond_in(n2)

    def shift_bond_out(self, n2, n1, x):
        '''shift_bond_out() will shift the element of J_out with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           In shift_bond_out(), no normalization is done before calculating the energy difference.
           But, rescale_bond_out() has to be called after accept.
        '''
        self.new_J_out = copy.copy(self.J_out)  # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_out[n2, n1] = (self.J_out[n2, n1] + x * self.RAT) * self.RESCALE_J

    def rescale_bond_out(self, n2):
        # rescaling 
        N = self.N
        t = self.new_J_out[n2]
        N_prim = np.sum(
            t * t)  # Use sort, because we want to avoid the larger values 'eats' the smaller ones. But I do not need to use sort in np.sum(), I believe.
        SCALE = np.sqrt(N / N_prim)
        self.new_J_out[n2] = self.new_J_out[n2] * SCALE

    def shift_bond_out_v2(self, n2, n1, x):
        '''shift_bond_out() will shift the element of J_out with a given index to another value. We add n2,n1 as parameters, for parallel programming.
           shift_bond_out_v2() rescals the synaptic weights before calculating the energy differnce.
        '''
        self.new_J_out = copy.copy(self.J_out)  # for storing temperay array when update
        # scale denotes standard deviation
        # x is a random number following the Gaussian distribution with 0 mean and variance 1. Ref: Yoshino2019
        self.new_J_out[n2, n1] = (self.J_out[n2, n1] + x * self.RAT) * self.RESCALE_J
        # rescaling 
        self.rescale_bond_out(n2)

    # The following accept function is used if S is flipped.
    def accept_by_mu_l_n(self, mu, l, n):
        """This accept function is used if S is flipped."""
        self.S[mu, l, n] = -self.S[mu, l, n]
        self.H = self.H + self.delta_H

    def accept_by_mu_l_n_batch(self, mu_array, l_array, n_array, delta_H_array):
        """
        Batch version of accept_by_mu_l_n.
        Parameters:
            mu_array: np.ndarray
                Array of indices for `mu`.
            l_array: np.ndarray
                Array of indices for `l`.
            n_array: np.ndarray
                Array of indices for `n`.
            delta_H_array: np.ndarray
                Array of delta_H values corresponding to the batch.
        """
        # Batch flip spins in S
        self.S[mu_array, l_array, n_array] *= -1  # Flip all spins at once

        # Batch update H
        self.H += np.sum(delta_H_array)  # Sum the delta_H of accepted updates

    def accept_by_mu_l_n_batch_torch(self, mu_array, l_array, n_array, delta_H_array):
        # Batch flip spins in S
        self.S[mu_array, l_array, n_array] *= -1  # Flip all spins at once

        # Batch update H
        self.H += torch.sum(delta_H_array)  # Sum the delta_H of accepted updates

    # One of the following accept functions is used if J is shifted.
    def accept_bond_hidden_by_l_n2_n1(self, l, n2):
        self.J_hidden[l, n2] = self.new_J_hidden[l, n2]
        self.H = self.H + self.delta_H

    def accept_bond_hidden_by_l_n2_n1_batch(self, l, n2, new_J_hidden, delta_h):
        self.J_hidden[l, n2] = new_J_hidden
        self.H = self.H + delta_h

    def accept_bond_in_by_n2_n1(self, n2):
        self.J_in[n2] = self.new_J_in[n2]
        self.H = self.H + self.delta_H

    def accept_bond_in_by_n2_n1_batch(self, n2, new_J_in, delta_h):
        self.J_in[n2] = new_J_in
        self.H = self.H + delta_h

    def accept_bond_out_by_n2_n1(self, n2):
        self.J_out[n2] = self.new_J_out[n2]
        self.H = self.H + self.delta_H

    def accept_bond_out_by_n2_n1_batch(self, n2, new_J_out, delta_h):
        self.J_out[n2] = new_J_out
        self.H = self.H + delta_h

    # One of the gap function is used if S is flipped.
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        '''l_s: index for hidden layers of the NODE (hidden S), l_s =  1, 2,...,7 (Totally, (L-3)+2= 7 hidden node layers + 2 layers, if L = 10).
           Ref: Yoshino2019, eqn (31b)
           When S is fliped, only one machine changes its coordinates and it will affect the gap of the node in front of it and the gaps of the N nodes
           behind it. Therefore, N+1 gaps contributes to the Delta_H_eff.
           We define a small array, part_gap, which has N+1 elements. Each elements of part_gap is a r^mu_node. Use part_gap, we can calculate the
           Energy change coused by the flip of S^mu_node,n.
        '''
        l_h = l_s - 1

        # Preallocate the result array
        part_gap = np.zeros(self.N + 1, dtype=np.float32)

        # Effect on the previous gap (index 0)
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]  # Dot product
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]

        # Effect on the N gaps in the next layer (indices 1 to N)
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :].reshape(-1)  # Matrix-vector multiplication
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]  # Element-wise scaling

        return part_gap

    def part_gap_hidden_before_flip_batch(self, mu_batch, l_s_batch, n_batch):
        return np.array([self.part_gap_hidden_before_flip(mu, l, n) for mu, l, n in zip(mu_batch, l_s_batch, n_batch)])

    def part_gap_hidden_after_flip(self, mu, l_s, n):
        part_gap = np.zeros(self.N + 1, dtype='float32')
        l_h = l_s - 1

        self.S[mu, l_s, n] = -self.S[mu, l_s, n]

        part_gap[0] = np.dot(self.J_hidden[l_h, n, :], self.S[mu, l_h, :]) / self.SQRT_N * self.S[mu, l_s, n]

        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :].reshape(-1)  # 矩阵-向量乘法
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]  # 元素级缩放

        self.S[mu, l_s, n] = -self.S[mu, l_s, n]

        return part_gap

    def part_gap_hidden_after_flip_torch(self, mu, l_s, n):
        part_gap = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
        l_h = l_s - 1

        # 翻转自旋
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]

        # 计算对前一个间隙（索引 0）的影响
        part_gap[0] = ( torch.dot(self.J_hidden[l_h, n, :], self.S[mu, l_h, :]) / self.SQRT_N) * self.S[mu, l_s, n]

        # 计算对下一层间隙（索引 1 到 N）的影响
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :].view(-1)  # 矩阵-向量乘法
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]  # 元素级缩放

        # 恢复自旋
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]

        return part_gap

    def part_gap_hidden_before_flip_torch(self, mu, l_s, n):
        part_gap = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
        l_h = l_s - 1

        # 计算对前一个间隙（索引 0）的影响
        part_gap[0] = ( torch.dot(self.J_hidden[l_h, n, :], self.S[mu, l_h, :]) / self.SQRT_N) * self.S[mu, l_s, n]

        # 计算对下一层间隙（索引 1 到 N）的影响
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :].view(-1)  # 矩阵-向量乘法
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]  # 元素级缩放

        return part_gap

    def part_gap_hidden_after_flip_batch(self, mu_batch, l_s_batch, n_batch):
        return np.array([self.part_gap_hidden_after_flip(mu, l, n) for mu, l, n in zip(mu_batch, l_s_batch, n_batch)])

    def part_gap_hidden_after_flip_batch_torch1(self, mu_batch, l_s_batch, n_batch):
        return torch.stack([self.part_gap_hidden_after_flip_torch(mu, l, n) for mu, l, n in zip(mu_batch, l_s_batch, n_batch)])

    def part_gap_hidden_before_flip_batch_torch1(self, mu_batch, l_s_batch, n_batch):
        return torch.stack([self.part_gap_hidden_before_flip_torch(mu, l, n) for mu, l, n in zip(mu_batch, l_s_batch, n_batch)])

    def part_gap_hidden_after_flip_batch_torch(self, mu_batch, l_s_batch, n_batch):
        # 创建一个张量来存储所有批次的结果
        part_gap_batch = torch.zeros(len(mu_batch), self.N + 1, dtype=torch.float32, device=self.device)

        # 更新每个 mu, l_s, n 对应的部分能量差
        l_h_batch = l_s_batch - 1  # 计算上一层的索引

        # 翻转自旋，使用 advanced indexing 来同时更新所有 batch 的对应自旋
        self.S[mu_batch, l_s_batch, n_batch] = -self.S[mu_batch, l_s_batch, n_batch]

        # S_lh_batch = self.S[mu_batch, l_h_batch, :]  # (batch_size, N)
        S_l_batch = self.S[mu_batch, l_s_batch, :]  # (batch_size, N)
        S_next_batch = self.S[mu_batch, l_s_batch + 1, :]  # (batch_size, N)

        # 计算 part_gap[0]，这里是矩阵-向量乘法，进行批量操作
        part_gap_batch[:, 0] = torch.sum(self.J_hidden[l_h_batch, n_batch, :] * self.S[mu_batch, l_h_batch, :],
                                         dim=1) / self.SQRT_N * self.S[mu_batch, l_s_batch, n_batch]

        # 将 S_l_batch 调整为 (batch_size, N, 1) 或者直接调整 J_hidden 的维度以兼容
        J_hidden_next_batch = torch.bmm(self.J_hidden[l_s_batch], self.S[mu_batch, l_s_batch].unsqueeze(-1)).squeeze(
            -1)  # (batch_size, N) 与 (N, batch_size) 转置后相乘
        part_gap_batch[:, 1:] = (J_hidden_next_batch / self.SQRT_N) * S_next_batch  # 计算 part_gap 后面的部分

        self.S[mu_batch, l_s_batch, n_batch] = -self.S[mu_batch, l_s_batch, n_batch]
        return part_gap_batch

    def part_gap_hidden_before_flip_batch_torch(self, mu_batch, l_s_batch, n_batch):
        # 创建一个张量来存储所有批次的结果
        part_gap_batch = torch.zeros(len(mu_batch), self.N + 1, dtype=torch.float32, device=self.device)

        # 更新每个 mu, l_s, n 对应的部分能量差
        l_h_batch = l_s_batch - 1  # 计算上一层的索引

        S_lh_batch = self.S[mu_batch, l_h_batch, :]  # (batch_size, N)
        S_l_batch = self.S[mu_batch, l_s_batch, :]  # (batch_size, N)
        S_next_batch = self.S[mu_batch, l_s_batch + 1, :]  # (batch_size, N)

        # 计算 part_gap[0]，这里是矩阵-向量乘法，进行批量操作
        part_gap_batch[:, 0] = torch.sum(self.J_hidden[l_h_batch, n_batch, :] * S_lh_batch,
                                         dim=1) / self.SQRT_N * self.S[mu_batch, l_s_batch, n_batch]

        J_hidden_next_batch = torch.bmm(self.J_hidden[l_s_batch], self.S[mu_batch, l_s_batch].unsqueeze(-1)).squeeze(
            -1)  # (batch_size, N) 与 (N, batch_size) 转置后相乘
        part_gap_batch[:, 1:] = (J_hidden_next_batch / self.SQRT_N) * S_next_batch  # 计算 part_gap 后面的部分

        return part_gap_batch

    def part_gap_in_before_flip(self, mu, n):
        """
        If a spin in the first layer flips, then r_in will change.
        """
        # Allocate the result array with dtype float32
        part_gap = np.zeros(self.N + 1, dtype=np.float32)

        # Compute the effect on the previous gap
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :]) / self.SQRT_N_IN * self.S[mu, 0, n]

        # Compute the effects on the N gaps in the next layer using vectorized operations
        part_gap[1:] = ((self.J_hidden[0, :, :] @ self.S[mu, 0, :]) / self.SQRT_N) * self.S[mu, 1, :]

        return part_gap  # Return the result array with N+1 elements

    def part_gap_in_before_flip_batch(self, mu_vals, n_vals):
        # 返回长度与 mu_vals 一致的数组
        return np.array([self.part_gap_in_before_flip(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_in_after_flip(self, mu, n):
        # Preallocate the result array with dtype float32
        part_gap = np.zeros(self.N + 1, dtype=np.float32)

        self.S[mu, 0, n] = -self.S[mu, 0, n]
        # Compute the effect on the previous gap (index 0)
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :] / self.SQRT_N_IN) * self.S[mu, 0, n]

        # Compute the effects on the N gaps in the next layer (indices 1 to N)
        part_gap[1:] = (self.J_hidden[0, :, :] @ self.S[mu, 0, :] / self.SQRT_N) * self.S[mu, 1, :]

        self.S[mu, 0, n] = -self.S[mu, 0, n]

        return part_gap  # Only the N+1 elements affect the Delta_H_eff.

    import torch

    def part_gap_in_after_flip_batch(self, mu_vals, n_vals):
        # 批量计算翻转后的部分能量差
        # 返回长度与 mu_vals 一致的数组
        return np.array([self.part_gap_in_after_flip(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_in_after_flip_torch(self, mu, n):
        # 预分配结果张量，数据类型为 float32
        part_gap = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
        # 修改 S 张量的值
        self.S[mu, 0, n] = -self.S[mu, 0, n]

        # 计算对前一个间隙（索引 0）的影响
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :] / self.SQRT_N_IN) * self.S[mu, 0, n]

        # 计算对下一层 N 个间隙（索引 1 到 N）的影响
        part_gap[1:] = (self.J_hidden[0, :, :] @ self.S[mu, 0, :] / self.SQRT_N) * self.S[mu, 1, :]

        # 恢复 S 张量的值
        self.S[mu, 0, n] = -self.S[mu, 0, n]

        return part_gap  # 只有 N+1 个元素影响 Delta_H_eff


    @timethis
    def part_gap_in_after_flip_batch_torch(self, mu_vals, n_vals):
        # Preallocate the result tensor (part_gap) on the specified device
        part_gap = torch.zeros((mu_vals.size(0), self.N + 1), dtype=torch.float32, device=self.device)

        # Flip the sign of S[mu, 0, n] for the computation
        self.S[mu_vals, 0, n_vals] = -self.S[mu_vals, 0, n_vals]

        # Compute the part_gap for index 0 (previous layer)
        part_gap[:, 0] = torch.sum(
            self.J_in[n_vals, :] @ self.S_in[mu_vals, :].T / self.SQRT_N_IN, dim=1
        ) * self.S[mu_vals, 0, n_vals]

        # Compute the part_gap for indices 1 to N (next layer)
        part_gap[:, 1:] = torch.sum(
            self.J_hidden[0, :, :] @ self.S[mu_vals, 0, :].T / self.SQRT_N, dim=1
        ) * self.S[mu_vals, 1, :]

        # Restore the original sign of S[mu, 0, n]
        self.S[mu_vals, 0, n_vals] = -self.S[mu_vals, 0, n_vals]

        # Return the part_gap tensor (batch-wise result)
        return part_gap

    def part_gap_in_after_flip_batch_torch1(self, mu_vals, n_vals):
        # 批量计算翻转后的部分能量差
        return torch.stack([self.part_gap_in_after_flip_torch(mu, n) for mu, n in zip(mu_vals, n_vals)])


    def part_gap_in_before_flip_torch(self, mu, n):
        # 预分配结果张量，数据类型为 float32
        part_gap = torch.zeros(self.N + 1, dtype=torch.float32, device=self.device)
        # 计算对前一个间隙（索引 0）的影响
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :] / self.SQRT_N_IN) * self.S[mu, 0, n]

        # 计算对下一层 N 个间隙（索引 1 到 N）的影响
        part_gap[1:] = (self.J_hidden[0, :, :] @ self.S[mu, 0, :] / self.SQRT_N) * self.S[mu, 1, :]

        return part_gap  # 只有 N+1 个元素影响 Delta_H_eff


    def part_gap_in_before_flip_batch_torch1(self, mu_vals, n_vals):
        # 批量计算翻转后的部分能量差
        return torch.stack([self.part_gap_in_before_flip_torch(mu, n) for mu, n in zip(mu_vals, n_vals)])


    def part_gap_in_before_flip_batch_torch(self, mu_vals, n_vals):
        # Preallocate the result tensor (part_gap) on the specified device
        part_gap = torch.zeros((mu_vals.size(0), self.N + 1), dtype=torch.float32, device=self.device)

        # Compute the part_gap for index 0 (previous layer)
        part_gap[:, 0] = torch.sum(
            self.J_in[n_vals, :] @ self.S_in[mu_vals, :].T / self.SQRT_N_IN, dim=1
        ) * self.S[mu_vals, 0, n_vals]

        # Compute the part_gap for indices 1 to N (next layer)
        part_gap[:, 1:] = torch.sum(
            self.J_hidden[0, :, :] @ self.S[mu_vals, 0, :].T / self.SQRT_N, dim=1
        ) * self.S[mu_vals, 1, :]

        # Return the part_gap tensor (batch-wise result)
        return part_gap

    def part_gap_out_before_flip(self, mu, n):
        N_out = self.N_out
        SQRT_N = self.SQRT_N

        # Preallocate the result array
        part_gap = np.zeros(N_out + 1, dtype=np.float32)

        # Effect on the previous gap (index 0)
        J_hidden_S = self.J_hidden[-1, n, :] @ self.S[mu, -2, :]  # Dot product
        part_gap[0] = (J_hidden_S / SQRT_N) * self.S[mu, -1, n]

        # Effects on the N_out gaps in the output layer (indices 1 to N_out)
        J_out_S = self.J_out @ self.S[mu, -1, :]  # Matrix-vector multiplication
        part_gap[1:] = (J_out_S / SQRT_N) * self.S_out[mu, :]  # Element-wise scaling

        return part_gap

    def part_gap_out_before_flip_torch(self, mu, n):
        N_out = self.N_out
        SQRT_N = self.SQRT_N

        # 预分配结果张量
        part_gap = torch.zeros(N_out + 1, dtype=torch.float32, device=self.device)

        # 对前一个间隙（索引 0）的影响
        J_hidden_S = torch.dot(self.J_hidden[-1, n, :], self.S[mu, -2, :])  # 点积
        part_gap[0] = (J_hidden_S / SQRT_N) * self.S[mu, -1, n]

        # 对输出层 N_out 个间隙（索引 1 到 N_out）的影响
        J_out_S = self.J_out @ self.S[mu, -1, :]  # 矩阵-向量乘法
        part_gap[1:] = (J_out_S / SQRT_N) * self.S_out[mu, :]  # 元素级缩放

        return part_gap

    def part_gap_out_before_flip_batch(self, mu_vals, n_vals):
        # 返回长度与 mu_vals 一致的数组
        return np.array([self.part_gap_out_before_flip(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_out_before_flip_batch_torch1(self, mu_vals, n_vals):
        return torch.stack([self.part_gap_out_before_flip_torch(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_out_after_flip_batch_torch1(self, mu_vals, n_vals):
        return torch.stack([self.part_gap_out_after_flip_torch(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_out_after_flip_batch_torch(self, mu_vals, n_vals):
        # Preallocate the result tensor for batch results
        part_gap_batch = torch.zeros(len(mu_vals), self.N_out + 1, dtype=torch.float32, device=self.device)

        # Flip the corresponding elements in S for the batch computation
        self.S[mu_vals, -1, n_vals] = -self.S[mu_vals, -1, n_vals]

        # Compute part_gap for the first gap (index 0) using matrix multiplication
        J_hidden_S = torch.matmul(self.J_hidden[-1, n_vals, :], self.S[mu_vals, -2, :].T)  # Dot product
        part_gap_batch[:, 0] = (J_hidden_S / self.SQRT_N) * self.S[mu_vals, -1, n_vals]

        # Compute part_gap for the output layer gaps (indices 1 to N_out)
        J_out_S_new = torch.matmul(self.J_out, self.S[mu_vals, -1, :].T)  # Matrix-vector multiplication
        J_out_S_new_expanded = J_out_S_new.squeeze(-1).unsqueeze(0)  # (2, 1) -> (1, 2)
        part_gap_batch[:, 1:] = (J_out_S_new_expanded / self.SQRT_N) * (self.S_out[mu_vals, :])

        # Restore the flipped values in S
        self.S[mu_vals, -1, n_vals] = -self.S[mu_vals, -1, n_vals]

        return part_gap_batch

    def part_gap_out_before_flip_batch_torch(self, mu_vals, n_vals):
        # Preallocate the result tensor for batch results
        part_gap_batch = torch.zeros(len(mu_vals), self.N_out + 1, dtype=torch.float32, device=self.device)

        # Compute part_gap for the first gap (index 0) using matrix multiplication
        J_hidden_S = torch.matmul(self.J_hidden[-1, n_vals, :], self.S[mu_vals, -2, :].T)  # Dot product
        part_gap_batch[:, 0] = (J_hidden_S / self.SQRT_N) * self.S[mu_vals, -1, n_vals]

        # Compute part_gap for the output layer gaps (indices 1 to N_out)
        J_out_S_new = torch.matmul(self.J_out, self.S[mu_vals, -1, :].T)  # Matrix-vector multiplication
        J_out_S_new_expanded = J_out_S_new.squeeze(-1).unsqueeze(0)  # (2, 1) -> (1, 2)
        part_gap_batch[:, 1:] = (J_out_S_new_expanded / self.SQRT_N) * self.S_out[mu_vals, :]

        return part_gap_batch

    def part_gap_out_after_flip(self, mu, n):
        """Compute the change in r_out after a spin flip in the last hidden layer."""
        N_out = self.N_out
        SQRT_N = self.SQRT_N

        # Preallocate the result array
        part_gap = np.zeros(N_out + 1, dtype=np.float32)

        self.S[mu, -1, n] = -self.S[mu, -1, n]
        # Effect on the previous gap (index 0)
        J_hidden_S = self.J_hidden[-1, n, :] @ self.S[mu, -2, :]  # Dot product
        part_gap[0] = (J_hidden_S / SQRT_N) * self.S[mu, -1, n]

        # Effects on the N_out gaps in the output layer (indices 1 to N_out)
        J_out_S_new = self.J_out @ self.S[mu, -1, :]  # Matrix-vector multiplication
        part_gap[1:] = (J_out_S_new / SQRT_N) * self.S_out[mu, :]  # Element-wise scaling

        self.S[mu, -1, n] = -self.S[mu, -1, n]

        return part_gap

    def part_gap_out_after_flip_batch(self, mu_vals, n_vals):
        return np.array([self.part_gap_out_after_flip(mu, n) for mu, n in zip(mu_vals, n_vals)])

    def part_gap_out_after_flip_torch(self, mu, n):
        """
        计算在最后一层隐藏层中自旋翻转后 r_out 的变化。
        """
        N_out = self.N_out
        SQRT_N = self.SQRT_N

        # 预分配结果张量
        part_gap = torch.zeros(N_out + 1, dtype=torch.float32, device=self.device)

        # 翻转自旋
        self.S[mu, -1, n] = -self.S[mu, -1, n]

        # 对前一个间隙（索引 0）的影响
        J_hidden_S = torch.dot(self.J_hidden[-1, n, :], self.S[mu, -2, :])  # 点积
        part_gap[0] = (J_hidden_S / SQRT_N) * self.S[mu, -1, n]

        # 对输出层 N_out 个间隙（索引 1 到 N_out）的影响
        J_out_S_new = self.J_out @ self.S[mu, -1, :]  # 矩阵-向量乘法
        part_gap[1:] = (J_out_S_new / SQRT_N) * self.S_out[mu, :]  # 元素级缩放

        # 恢复自旋
        self.S[mu, -1, n] = -self.S[mu, -1, n]

        return part_gap


    def part_gap_common_shift(self, in_num, J_n, S_before, S_after):
        part_gap = np.zeros(self.M, dtype='float32')
        part_gap[:] = (np.sum(J_n * S_before, axis=1) / in_num) * S_after
        return part_gap

    def part_gap_common_shift_torch(self, in_num, J_n, S_before, S_after):
        """Torch版本的part_gap_common_shift"""
        # 计算 part_gap 并使用张量操作替代 NumPy
        part_gap = torch.sum(J_n * S_before, dim=1) / in_num
        part_gap = part_gap * S_after
        return part_gap

    def part_gap_common_shift_torch_batch(self, in_num, J_n, S_before, S_after):
        part_gap = torch.zeros(J_n.size(0), self.M, dtype=torch.float32, device=self.device)
        part_gap[:] = (torch.sum(J_n.unsqueeze(1) * S_before, dim=2) / in_num) * S_after.T
        return part_gap

    def part_gap_in_shift(self, n, J_in_n):
        return self.part_gap_common_shift(self.SQRT_N_IN, J_in_n, self.S_in, self.S[:, 0, n])

    def part_gap_out_shift(self, n_out, J_out_n_out):
        return self.part_gap_common_shift(self.SQRT_N, J_out_n_out, self.S[:, -1, :], self.S_out[:, n_out])

    def part_gap_hidden_shift(self, l, n, J_hidden_l_n):
        return self.part_gap_common_shift(self.SQRT_N, J_hidden_l_n, self.S[:, l, :], self.S[:, l+1, n])

    # Above are version 2 of part_gap_XXX_*shift functions.
    # ====================================================

    def rand_series_for_x(self):
        """
        For generating J: list_for_x = [x1,x2,...]
        We separate rand_index_for_J() and rand_series_for_x(), instead of merginging them to one function and return a list of four-tuple (l,n2,n1,x).
        The reason is: x is float and l,n2,n1 are integers, it will induce trouble if one put them (x and l,n2,n1 ) together.
        """
        list_for_x = []
        for _ in range(self.num_bonds * (self.tot_steps - 1)):
            x = np.random.normal(loc=0, scale=1.0, size=None)
            x = round(x, 10)
            list_for_x.append(x)
        res_arr = np.array(list_for_x)
        return res_arr

    def rand_series_for_decision_on_S(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_variables * (self.tot_steps - 1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr

    def rand_series_for_decision_on_J(self):
        # For generating J: list_index_for_J = [(l,n2,n1),...]
        list_for_decision = []
        for _ in range(self.num_bonds * (self.tot_steps - 1)):
            list_for_decision.append(np.random.random(1))
        res_arr = np.array(list_for_decision)
        return res_arr

    def check_and_save_seeds_v2(self, mc_index, list_k):
        """Use MC_step as index, instead of element in range(count_MC_step * self.num) as index
           v2 is a general version than v1.
        """
        ind = mc_index
        init = self.init
        self.list_k = list_k
        if ind == self.list_k[0]:
            self.list_k.pop(0)
            np.save('../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind, self.L,
                                                                                                     self.M, self.N,
                                                                                                     self.beta), self.S)
            np.save('../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind, self.L,
                                                                                                      self.N,
                                                                                                      self.beta),
                    self.J_hidden)
            np.save('../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init, ind, self.N,
                                                                                                     self.N_in,
                                                                                                     self.beta),
                    self.J_in)
            np.save('../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind,
                                                                                                       self.N_out,
                                                                                                       self.N,
                                                                                                       self.beta),
                    self.J_out)
            np.save('../data/host/seed_ener_new_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind,
                                                                                                            self.L,
                                                                                                            self.M,
                                                                                                            self.N,
                                                                                                            self.beta),
                    self.H_hidden)
        else:
            pass

    def check_and_save_seeds(self, mc_index):
        """Use MC_step as index, instead of element in range(count_MC_step * self.num) as index."""
        ind = mc_index
        init = self.init
        if ind == self.list_k[0]:
            self.list_k.pop(0)
            np.save('../data/host/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind, self.L,
                                                                                                     self.M, self.N,
                                                                                                     self.beta), self.S)
            np.save('../data/host/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind, self.L,
                                                                                                      self.N,
                                                                                                      self.beta),
                    self.J_hidden)
            np.save('../data/host/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(init, ind, self.N,
                                                                                                     self.N_in,
                                                                                                     self.beta),
                    self.J_in)
            np.save('../data/host/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind,
                                                                                                       self.N_out,
                                                                                                       self.N,
                                                                                                       self.beta),
                    self.J_out)
            np.save('../data/host/seed_ener_new_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(init, ind,
                                                                                                            self.L,
                                                                                                            self.M,
                                                                                                            self.N,
                                                                                                            self.beta),
                    self.H_hidden)
        else:
            pass

    def check_and_save_v2(self, count_MC_step):
        ind = count_MC_step
        if int((ratio) ** self.ind_save) == ind and self.save < self.S_traj.shape[0]:
            self.S_traj[self.ind_save] = self.S
            self.J_hidden_traj[self.ind_save] = self.J_hidden
            self.H_hidden_traj[self.ind_save] = self.H_hidden
            self.ind_save += 1
            self.check_and_save_v2(ind)
        else:
            pass

    def check_and_save(self):
        ind = self.count_MC_step
        if 2 ** self.ind_save == ind and self.ind_save < self.S_traj.shape[0]:
            self.S_traj[self.ind_save] = self.S
            self.J_hidden_traj[self.ind_save] = self.J_hidden
            self.H_hidden_traj[self.ind_save] = self.H_hidden
            self.ind_save += 1
        else:
            pass

    def check_and_save_hyperfine_v2(self, update_index):
        ind = update_index
        if int((ratio) ** self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
            self.S_traj_hyperfine[self.ind_save] = self.S
            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
            self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
            self.ind_save += 1
            self.check_and_save_hyperfine_v2(update_index)
        else:
            pass

    def check_and_save_hyperfine(self, update_index):
        ind = update_index
        if (2 ** self.ind_save) == ind and self.ind_save < self.S_traj_hyperfine.shape[0]:
            self.S_traj_hyperfine[self.ind_save] = self.S
            self.J_in_traj_hyperfine[self.ind_save] = self.J_in
            self.J_out_traj_hyperfine[self.ind_save] = self.J_out
            self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
            self.ind_save += 1
        else:
            pass

    # One of the follwing decision functions is used if S is flipped.
    def decision_by_mu_l_n_SIMPLE(self, mu, l, n):
        """If use this decision_by_mu_l_n_SIMPLE() function, the parameter EPS is not needed in the input."""
        # Const.s
        rand1 = np.random.random(1)
        a1 = self.part_gap_hidden_before_flip(mu, l, n)
        a2 = self.part_gap_hidden_after_flip(mu, l, n)
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e > EPS:
            if rand1 < np.exp(-delta_e * self.beta):
                self.accept_by_mu_l_n(mu, l, n)
            else:
                pass
        else:
            self.accept_by_mu_l_n(mu, l, n)

    def decision_by_mu_l_n(self, mu, l, n, EPS, rand):
        self.delta_H = calc_ener(self.part_gap_hidden_after_flip(mu, l, n)) - calc_ener(
            self.part_gap_hidden_before_flip(mu, l, n))
        delta_e = self.delta_H
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu, l, n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu, l, n)
            else:
                pass

    def decision_by_mu_l_n_batch(self, mu_array, l_array, n_array, EPS, rand_array):
        """
        Batch version of decision_by_mu_l_n.
        Parameters:
            mu_array: np.ndarray
                Array of indices for `mu`.
            l_array: np.ndarray
                Array of indices for `l`.
            n_array: np.ndarray
                Array of indices for `n`.
            EPS: float
                Threshold value for energy difference.
            rand_array: np.ndarray
                Array of random numbers for probability decision.
        """
        if len(mu_array) == 0:
            return
        # Batch compute delta_H
        delta_H_array = (calc_ener_batch(self.part_gap_hidden_after_flip_batch(mu_array, l_array, n_array))
                         - calc_ener_batch(self.part_gap_hidden_before_flip_batch(mu_array, l_array, n_array)))

        # Compute delta_e
        delta_e_array = delta_H_array

        # Determine accept conditions
        accept_condition = delta_e_array < EPS  # Direct acceptance
        prob_ref_array = np.exp(-delta_e_array * self.beta)  # Probability threshold
        prob_accept_condition = rand_array < prob_ref_array  # Probabilistic acceptance

        # Combine acceptance conditions
        accept_indices = accept_condition | prob_accept_condition

        # Apply batch acceptance
        self.accept_by_mu_l_n_batch(mu_array[accept_indices],
                                    l_array[accept_indices],
                                    n_array[accept_indices],
                                    delta_H_array[accept_indices])

    def decision_by_mu_l_n_batch_torch(self, mu_array, l_array, n_array, EPS, rand_array):
        if len(mu_array) == 0:
            return
        # Batch compute delta_H
        delta_H_array = (calc_ener_batch_torch(self.part_gap_hidden_after_flip_batch_torch1(mu_array, l_array, n_array),
                                               self.device) -
                         calc_ener_batch_torch(self.part_gap_hidden_before_flip_batch_torch1(mu_array, l_array, n_array),
                                               self.device))

        # Compute delta_e
        delta_e_array = delta_H_array

        # Determine accept conditions
        accept_condition = delta_e_array < EPS  # Direct acceptance
        prob_ref_array = torch.exp(-delta_e_array * self.beta)  # Probability threshold
        prob_accept_condition = rand_array < prob_ref_array  # Probabilistic acceptance

        # Combine acceptance conditions
        accept_indices = accept_condition | prob_accept_condition

        # Apply batch acceptance
        self.accept_by_mu_l_n_batch_torch(mu_array[accept_indices],
                                          l_array[accept_indices],
                                          n_array[accept_indices],
                                          delta_H_array[accept_indices])

    def decision_node_in_by_mu_n(self, mu, n, EPS, rand):
        self.delta_H = calc_ener(self.part_gap_in_after_flip(mu, n)) - calc_ener(self.part_gap_in_before_flip(mu, n))
        delta_e = self.delta_H
        temp_l = 0
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu, temp_l, n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu, temp_l, n)
            else:
                pass

    def decision_node_in_by_mu_n_batch(self, mu_vals, n_vals, EPS, rand_vals):
        """
        批量处理 decision_node_in_by_mu_n。
        :param mu_vals: 一维数组，表示 mu 的索引。
        :param n_vals: 一维数组，表示 n 的索引。
        :param EPS: 标量，能量变化的阈值。
        :param rand_vals: 一维数组，长度与 mu_vals 一致，表示随机数。
        """
        if len(mu_vals) == 0:
            return

        temp_l = 0  # 对应的固定层
        # 批量计算能量差
        delta_H_vals = (
                calc_ener_batch(self.part_gap_in_after_flip_batch(mu_vals, n_vals)) -
                calc_ener_batch(self.part_gap_in_before_flip_batch(mu_vals, n_vals))
        )

        # 更新 self.delta_H 为批量值（可选，仅供后续使用或调试）
        self.delta_H = delta_H_vals

        # 判断条件 delta_H < EPS 的布尔数组
        accept_condition = delta_H_vals < EPS

        # 对符合条件的直接接受
        self.accept_by_mu_l_n_batch(mu_vals[accept_condition],
                                    np.full(np.sum(accept_condition), temp_l),
                                    n_vals[accept_condition],
                                    delta_H_vals[accept_condition])

        # 对不符合条件的计算概率接受
        not_accepted = ~accept_condition
        prob_ref_vals = np.exp(-delta_H_vals[not_accepted] * self.beta)
        prob_condition = rand_vals[not_accepted] < prob_ref_vals

        # 处理符合概率条件的接受
        self.accept_by_mu_l_n_batch(mu_vals[not_accepted][prob_condition],
                                    np.full(np.sum(prob_condition), temp_l),
                                    n_vals[not_accepted][prob_condition],
                                    delta_H_vals[not_accepted][prob_condition])

    import torch

    def decision_node_in_by_mu_n_batch_torch(self, mu_vals, n_vals, EPS, rand_vals):
        if len(mu_vals) == 0:
            return

        temp_l = 0  # 对应的固定层
        # 批量计算能量差
        delta_H_vals = (
                calc_ener_batch_torch(self.part_gap_in_after_flip_batch_torch1(mu_vals, n_vals), self.device) -
                calc_ener_batch_torch(self.part_gap_in_before_flip_batch_torch1(mu_vals, n_vals), self.device)
        )

        # 更新 self.delta_H 为批量值（可选，仅供后续使用或调试）
        self.delta_H = delta_H_vals

        # 判断条件 delta_H < EPS 的布尔张量
        accept_condition = delta_H_vals < EPS

        not_accepted = ~accept_condition
        prob_ref_vals = torch.exp(-delta_H_vals[not_accepted] * self.beta)
        prob_condition = rand_vals[not_accepted] < prob_ref_vals

        # 合并直接接受和概率接受的条件
        final_accept_condition = accept_condition.clone()
        final_accept_condition[not_accepted] = prob_condition

        # 处理所有符合条件的样本
        self.accept_by_mu_l_n_batch_torch(
            mu_vals[final_accept_condition],
            torch.full((final_accept_condition.sum().item(),), temp_l, device=self.device),
            n_vals[final_accept_condition],
            delta_H_vals[final_accept_condition]
        )

    def decision_node_out_by_mu_n(self, mu, n, EPS, rand):
        self.delta_H = calc_ener(self.part_gap_out_after_flip(mu, n)) - calc_ener(self.part_gap_out_before_flip(mu, n))
        delta_e = self.delta_H
        l = -1
        if delta_e < EPS:
            self.accept_by_mu_l_n(mu, l, n)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_by_mu_l_n(mu, l, n)
            else:
                pass

    def decision_node_out_by_mu_n_batch(self, mu_array, n_array, EPS, rand_array):
        """
        Batch version of decision_node_out_by_mu_n.
        Parameters:
            mu_array: np.ndarray
                Array of indices for `mu`.
            n_array: np.ndarray
                Array of indices for `n`.
            EPS: float
                Threshold value for energy difference.
            rand_array: np.ndarray
                Array of random numbers for probability decision.
        """
        if len(mu_array) == 0:
            return
        # Batch compute delta_H
        delta_H_array = calc_ener_batch(self.part_gap_out_after_flip_batch(mu_array, n_array)) - \
                        calc_ener_batch(self.part_gap_out_before_flip_batch(mu_array, n_array))

        # Compute delta_e
        delta_e_array = delta_H_array

        # Determine which updates to accept
        l = -1  # l is fixed for all cases
        accept_condition = delta_e_array < EPS  # Direct accept if delta_e < EPS
        prob_ref_array = np.exp(-delta_e_array * self.beta)  # Probability threshold for acceptance
        prob_accept_condition = rand_array < prob_ref_array  # Probabilistic acceptance

        # Indices to accept
        accept_indices = accept_condition | prob_accept_condition

        # Apply acceptance
        self.accept_by_mu_l_n_batch(mu_array[accept_indices],
                                    np.full(np.sum(accept_indices), l),  # l is constant, replicated for batch
                                    n_array[accept_indices],
                                    delta_H_array[accept_indices])

    import torch

    def decision_node_out_by_mu_n_batch_torch(self, mu_array, n_array, EPS, rand_array):
        if len(mu_array) == 0:
            return

        # Batch compute delta_H
        delta_H_array = (calc_ener_batch_torch(
            self.part_gap_out_after_flip_batch_torch1(mu_array, n_array), self.device)
                         - calc_ener_batch_torch(
                    self.part_gap_out_before_flip_batch_torch1(mu_array, n_array), self.device))

        # Compute delta_e (same as delta_H for this case)
        delta_e_array = delta_H_array

        # Determine which updates to accept
        l = -1  # l is fixed for all cases
        accept_condition = delta_e_array < EPS  # Direct accept if delta_e < EPS
        prob_ref_array = torch.exp(-delta_e_array * self.beta)  # Probability threshold for acceptance
        prob_accept_condition = rand_array < prob_ref_array  # Probabilistic acceptance

        # Indices to accept
        accept_indices = accept_condition | prob_accept_condition

        # Apply acceptance
        self.accept_by_mu_l_n_batch_torch(mu_array[accept_indices],
                                          torch.full((torch.sum(accept_indices),), l, device=self.device),
                                          # l is constant, replicated for batch
                                          n_array[accept_indices],
                                          delta_H_array[accept_indices])

    # One of the follwing decision functions is used if J is shifted.
    def decision_bond_hidden_by_l_n2_n1(self, l, n2, n1, EPS, rand):
        self.delta_H = calc_ener(self.part_gap_hidden_shift(l, n2, self.new_J_hidden[l, n2, :])) - calc_ener(
            self.part_gap_hidden_shift(l, n2, self.J_hidden[l, n2, :]))
        delta_e = self.delta_H
        # ================
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            # self.accept_bond_hidden_by_l_n2_n1(l,n2,n1)
            self.accept_bond_hidden_by_l_n2_n1(l, n2)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_bond_hidden_by_l_n2_n1(l, n2)
            else:
                pass  # We do not need a "remain" function

    def decision_bond_in_by_n2_n1(self, n2, n1, EPS, rand):
        a1 = self.part_gap_in_shift(n2, self.J_in[n2])
        a2 = self.part_gap_in_shift(n2, self.new_J_in[n2])
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e < EPS:
            # Replace o.S by o.new_S:
            self.accept_bond_in_by_n2_n1(n2)
        else:
            # TEST (2 line)
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                self.accept_bond_in_by_n2_n1(n2)
            else:
                pass  # We do not need a "remain" function

    def decision_bond_out_by_n2_n1(self, n2, n1, EPS, rand):
        a1 = self.part_gap_out_shift(n2, self.J_out[n2])
        a2 = self.part_gap_out_shift(n2, self.new_J_out[n2])
        self.delta_H = calc_ener(a2) - calc_ener(a1)
        delta_e = self.delta_H
        if delta_e < EPS:
            # self.accept_bond_out_by_n2_n1(n2,n1)
            self.accept_bond_out_by_n2_n1(n2)
        else:
            prob_ref = np.exp(-delta_e * self.beta)
            if rand < prob_ref:
                # self.accept_bond_out_by_n2_n1(n2,n1)
                self.accept_bond_out_by_n2_n1(n2)
            else:
                pass  # We do not need a "remain" function

    @timethis
    def mc_main(self, replica_index):
        """MC for the host machine, i.e., it will save seeds for different waiting time."""
        str_replica_index = str(replica_index)
        rel_path = 'J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}_host.dat'.format(str_replica_index,
                                                                                                      self.init,
                                                                                                      self.tw, self.L,
                                                                                                      self.N, self.beta,
                                                                                                      self.tot_steps)
        src_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(src_dir, rel_path)

        EPS = self.EPS

        ## MC siulation starts
        for MC_index in range(1, self.tot_steps):
            # print("Updating S:")
            for update_index in range(0, self.num_variables):
                mu, l, n = randrange(self.M), randrange(self.num_hidden_node_layers), randrange(self.N)
                self.flip_spin(mu, l, n)
                rand1 = np.random.random(1)
                if l == 0:
                    self.decision_node_in_by_mu_n(mu, n, EPS, rand1)
                elif l == self.num_hidden_node_layers - 1:

                    self.decision_node_out_by_mu_n(mu, n, EPS, rand1)
                else:
                    self.decision_by_mu_l_n(mu, l, n, EPS, rand1)
            # print("Updating J:")
            for update_index in range(0, self.num_bonds):
                self.random_update_J()
            self.count_MC_step += 1
            # Check and save the seeds 
            # IF MC_index EQUALS 2**k, WHERE k = 1,2,3,4,5,...,12, THEN SAVE THE CONFIGURATION OF THE NDOES AND BONDS. 
            # THIS OPERATION SHUOLD BE ONLY DONE IN HOST MACHINE, DO NOT DO IT IN A GUEST MACHINE.
            self.check_and_save_seeds(MC_inex)

            # Check and save the configureation of bonds
            self.check_and_save()  # save configuration

            # ========================================================================================================
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
            # ========================================================================================================

    @timethis
    def mc_main_hyperfine(self, replica_index):
        """MC for the host machine, i.e., it will save seeds for different waiting time."""
        str_replica_index = str(replica_index)
        rel_path = 'J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}_step{:d}_host.dat'.format(str_replica_index,
                                                                                                      self.init,
                                                                                                      self.tw, self.L,
                                                                                                      self.N, self.beta,
                                                                                                      self.tot_steps)
        src_dir = os.path.dirname(__file__)
        abs_file_path = os.path.join(src_dir, rel_path)
        EPS = self.EPS

        # MC siulation starts
        for MC_index in range(1, self.tot_steps):
            # print("MC step: {:d}".format(MC_index))
            # print("Updating S:")
            start_index_variables = (MC_index - 1) * self.num
            end_index_variables = start_index_variables + self.num_variables
            for update_index in range(start_index_variables, end_index_variables):
                mu, l, n = randrange(self.M), randrange(self.num_hidden_node_layers), randrange(self.N)
                self.flip_spin(mu, l, n)
                rand1 = np.random.random(1)
                if l == 0:
                    self.decision_node_in_by_mu_n(mu, n, EPS, rand1)
                elif l == self.num_hidden_node_layers - 1:
                    self.decision_node_out_by_mu_n(mu, n, EPS, rand1)
                else:
                    self.decision_by_mu_l_n(mu, l, n, EPS, rand1)
                    # Check and save the configureation of variables
                self.check_and_save_hyperfine(update_index)  # save configuration
            # print("Updating J:")
            start_index_bonds = end_index_variables
            end_index_bonds = MC_index * self.num
            for update_index in range(start_index_bonds, end_index_bonds):
                self.random_update_J()
                # Check and save the configureation of bonds
                self.check_and_save_hyperfine(update_index)  # save configuration
            # Check and save the seeds
            self.check_and_save_seeds(MC_index)

    def mc_update_J_or_S(self):
        rand1 = np.random.random(1)
        if rand1 < self.PROB:
            self.random_update_S()
        else:
            self.random_update_J()

    def mc_update_J_or_S_batch(self, num_updates):
        # 生成随机数以决定更新 S 或 J
        rand_vals = np.random.random(num_updates)
        update_S_mask = rand_vals < self.PROB

        S_num_updates = np.sum(update_S_mask)
        J_num_updates = np.sum(~update_S_mask)

        self.random_update_S_batch_total(S_num_updates)
        self.random_update_J_batch(J_num_updates)  # 更新 J

    def mc_update_J_or_S_torch(self, num_updates):
        # 生成随机数以决定更新 S 或 J
        rand_vals = torch.rand(num_updates)  # 使用 torch.rand 替代 np.random.random
        update_S_mask = rand_vals < self.PROB  # 生成更新 S 的 mask

        # 计算更新 S 和 J 的次数
        S_num_updates = update_S_mask.sum().item()  # .sum() 会返回一个 Tensor，使用 .item() 获取数值
        J_num_updates = (~update_S_mask).sum().item()

        # 调用对应的更新函数
        self.random_update_S_batch_total_torch(S_num_updates)
        self.random_update_J_batch_torch(J_num_updates)  # 更新 J

    @timethis
    def random_update_S_batch_total(self, S_num_updates):
        S_batch_size = self.num_hidden_node_layers // 2
        # 分别调用批量更新函数
        while S_num_updates > S_batch_size:
            self.random_update_S_batch(S_batch_size)  # 更新 S
            S_num_updates -= S_batch_size
        if S_num_updates > 0:
            self.random_update_S_batch(S_num_updates)

    @timethis
    def random_update_S_batch_total_torch(self, S_num_updates):
        S_batch_size = self.num_hidden_node_layers // 2
        # 分别调用批量更新函数
        while S_num_updates > S_batch_size:
            self.random_update_S_batch_torch(S_batch_size)  # 更新 S
            S_num_updates -= S_batch_size
        if S_num_updates > 0:
            self.random_update_S_batch_torch(S_num_updates)


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
                # Check and save the configureation of variables
                self.check_and_save_hyperfine_v2(self.update_index)  # save configuration
                self.update_index = self.update_index + 1
                # Check and save the seeds (ONLY IN mc_main)
            self.check_and_save_seeds(mc_index)

    @timethis
    def mc_random_update_hyperfine_2(self, str_timestamp):
        """MC for the guest machine. It will 
           1. Use mc_index to denote waiting time (tw).
           2. Updating S and J randomly, with a fixed probability.
        """
        self.update_index = 0
        MC_STEPS = self.tot_steps

        self.H_history.append(self.H)
        # 使用 Profiler
        # with torch.profiler.profile(
        #         activities=[
        #             torch.profiler.ProfilerActivity.CPU,
        #             torch.profiler.ProfilerActivity.CUDA],
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
        #         record_shapes=True,
        #         with_stack=True
        # ) as prof:

        for mc_index in range(self.num // 189):
            self.mc_update_J_or_S_torch(MC_STEPS * 189)
            # Check and save the configureation of variables
            self.check_and_save_hyperfine_v2(MC_STEPS)  # save configuration
            self.update_index = self.update_index + 1
            self.H_history.append(self.H.cpu().item())

        draw_history(self.H_history)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
