# ======
# Module
# ======
import copy
# import dask.dataframe as dd
from datetime import datetime
import math
import numpy as np
#import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange
import random
import scipy as sp
import torch
from numba import njit
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras.utils import \
    to_categorical  # to_categorical() returns a binary matrix representation of the input. The class axis is placed last.
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========
# Functions
# =========
def num_dof(L, M, N, N_in, N_out):
    """This function is for the DNN system in our DNN and PT_DNN project."""
    SQ_N = N ** 2
    num_hidden_node_layers = L - 1
    num_hidden_bond_layers = L - 2
    num_variables = int(N * M * num_hidden_node_layers)
    num_bonds = int(N * N_in + SQ_N * num_hidden_bond_layers + N_out * N)
    num = num_variables + num_bonds
    return num


def num_dof_inner_representation(L, M, N, N_in, N_out):
    """This function is for the DNN system in our DNN and PT_DNN project."""
    num_hidden_node_layers = L - 1
    num_variables = int(N * M * num_hidden_node_layers)
    return num_variables


def sharpen(arr):
    arr[arr > 0] = 1  # An important UPDATE!!


def spinization(arr):
    arr[arr == 0] = -1  # We DID this step in the last AT ONCE, for the largest training set!


def generate_coord(N):
    """Randomly set the initial coordinates,whose components are 1 or -1."""
    v = np.zeros(N, dtype='int8')
    list_spin = [-1, 1]
    for i in range(N):
        v[i] = np.random.choice(list_spin)
    return v


def generate_J(N):
    J = generate_rand_normal_numbers(N)
    return J


def generate_rand_normal_numbers(N=4, mu=0, sigma=1):
    """生成N个满足正态分布的随机数,平均值mu,方差sigma."""
    samp = np.random.normal(loc=mu, scale=sigma, size=N)
    return samp


def generate_S_in_and_out_2_spin(sample_index, M, N_in):
    """Generate S_in and S_out from tf.keras. S_in and S_out are fixed for all the machines.
       1. Nout = 2 (version 1, DATE).
    """
    N_out = 2
    # (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    file = np.load("/public1/home/sc91981/Yoshino2G3/augmented_mnist/augmented_mnist.npz")
    train_X = file["arr_0"]
    train_y = file["arr_1"]
    # print("Train_y's shpae:{}".format(train_y.shape))
    # ================================
    # Generate new and CLEAN datasets 
    new_train_X = generate_new_train_X2(train_X, train_y, M, sample_index)
    new_train_y = generate_new_train_y2(train_y, M, sample_index)
    # =========================================================
    # Generate S_in. Since each figure is a 2D array, new_train_M[:M] is a 3D array. We reshpae the 3D array into 2D array.
    S_in = new_train_X[:M].reshape(-1, N_in)
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    S_out = to_categorical(new_train_y[:M])

    # In generate_S_in_and_out_2_spin(), we need {1,-1}, instead of {1,0} values.
    sharpen(S_in)  # Convert all positive value to 1
    spinization(S_in)  # Convert 0 to -1.
    spinization(S_out)
    return (S_in, S_out)


def generate_S_in_and_out_2(sample_index, M, N_in):
    """Generate S_in and S_out from tf.keras. S_in and S_out are fixed for all the machines.
       1. Nout = 10 (version 1, DATE).
    """
    N_out = 2
    file = np.load("~/Yoshino2G3/augmented_mnist/augmented_mnist.npz")
    train_X = file["arr_0"]
    train_y = file["arr_1"]
    # ================================
    # Generate new and CLEAN datasets 
    new_train_X = generate_new_train_X2(train_X, train_y, M, sample_index)
    new_train_y = generate_new_train_y2(train_y, M, sample_index)
    # =========================================================
    # Generate S_in. Since each figure is a 2D array, new_train_M[:M] is a 3D array. We reshpae the 3D array into 2D array.
    S_in = new_train_X[:M].reshape(-1, N_in)
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    S_out = to_categorical(new_train_y[:M])
    return (S_in, S_out)


def generate_S_in_and_out_2_spin_v2(sample_index, M, N_in):
    """Generate S_in and S_out from tf.keras. S_in and S_out are fixed for all the machines.
       1. Nout = 2
       2. Diff. between generate_S_in_and_out_2_spin_v2 and generate_S_in_and_out_2_spin() is: the former generate larger
          dataset.
    """
    N_out = 2
    file = np.load("/public1/home/sc91981/Yoshino2G3/augmented_mnist_2/augmented_mnist_2.npz")
    train_X = file["arr_0"]
    train_y = file["arr_1"]
    # ================================
    # Generate new and CLEAN datasets 
    new_train_X = generate_new_train_X2(train_X, train_y, M, sample_index)
    new_train_y = generate_new_train_y2(train_y, M, sample_index)
    # =========================================================
    # Generate S_in. Since each figure is a 2D array, new_train_M[:M] is a 3D array. We reshpae the 3D array into 2D array.
    S_in = new_train_X[:M].reshape(-1, N_in)
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    S_out = to_categorical(new_train_y[:M])

    # In generate_S_in_and_out_2_spin(), we need {1,-1}, instead of {1,0} values.
    sharpen(S_in)  # Convert all positive value to 1
    spinization(S_in)  # Convert 0 to -1.
    spinization(S_out)
    return (S_in, S_out)


def generate_S_in_and_out_2_spin_v3(file0, file1, sample_index, M, N_in, N_out):
    a = int(M / N_out)
    # Read csv files
    df0 = pd.read_csv(file0, header=None, skiprows=(sample_index - 1) * a, nrows=a)
    df1 = pd.read_csv(file1, header=None, skiprows=(sample_index - 1) * a, nrows=a)
    # Drop the index
    df0 = df0.drop(0, axis=1)
    df1 = df1.drop(0, axis=1)
    # 0
    train_X0 = df0.iloc[:, :-1]  # select columns
    train_X0 = train_X0.values  # Convert dataframe to array
    train_y0 = df0.iloc[:, -1]
    train_y0 = train_y0.values  # Convert dataframe to array
    # 1
    train_X1 = df1.iloc[:, :-1]
    train_X1 = train_X1.values
    train_y1 = df1.iloc[:, -1]
    train_y1 = train_y1.values

    new_X = np.concatenate((train_X0, train_X1), axis=0)
    new_y = np.concatenate((train_y0, train_y1), axis=0)
    # ================================
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    new_y = to_categorical(new_y)

    # In generate_S_in_and_out_2_spin(), we need {1,-1}, instead of {1,0} values.
    sharpen(new_X)  # Convert all positive value to 1
    spinization(new_X)  # Convert 0 to -1.
    spinization(new_y)


def generate_S_in_and_out_2_spin_v3_resp(file0, file1, M, N_in, N_out):
    # old: generate_S_in_and_out_2_spin_v3_resp(file0,file1,sample_index,M,N_in,N_out):
    "The output only has M samples."
    a = int(M / N_out)
    # Read csv files
    # file0 ='/public1/home/sc91981/Yoshino2G3/augmented_mnist_3/M{}/augmented_mnist_3_zeros_m{}_sample{}.csv'.format(M,M,sample_index)
    # file1 ='/public1/home/sc91981/Yoshino2G3/augmented_mnist_3/M{}/augmented_mnist_3_ones_m{}_sample{}.csv'.format(M,M,sample_index)
    df0 = pd.read_csv(file0, header=None, skiprows=0, nrows=a)
    df1 = pd.read_csv(file1, header=None, skiprows=0, nrows=a)
    # Drop the index
    df0 = df0.drop(0, axis=1)
    df1 = df1.drop(0, axis=1)
    # 0
    train_X0 = df0.iloc[:, :-1]  # select columns
    train_X0 = train_X0.values  # Convert dataframe to array
    train_y0 = df0.iloc[:, -1]
    train_y0 = train_y0.values  # Convert dataframe to array
    # 1
    train_X1 = df1.iloc[:, :-1]
    train_X1 = train_X1.values
    train_y1 = df1.iloc[:, -1]
    train_y1 = train_y1.values

    new_X = np.concatenate((train_X0, train_X1), axis=0)
    new_y = np.concatenate((train_y0, train_y1), axis=0)
    # ================================
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    new_y = to_categorical(new_y)

    # In generate_S_in_and_out_2_spin(), we need {1,-1}, instead of {1,0} values.
    sharpen(new_X)  # Convert all positive value to 1
    spinization(new_X)  # Convert 0 to -1.
    spinization(new_y)

    return (new_X, new_y)


def generate_S_in_and_out_2_v2(sample_index, M, N_in):
    """Generate S_in and S_out from tf.keras. S_in and S_out are fixed for all the machines.
       1. Nout = 10 
       2. Diff. between generate_S_in_and_out_2_v2 and generate_S_in_and_out_2() is: the former generate larger
    """
    N_out = 2
    file = np.load("~/Yoshino2G3/augmented_mnist_2/augmented_mnist_2.npz")
    train_X = file["arr_0"]
    train_y = file["arr_1"]
    # ================================
    # Generate new and CLEAN datasets 
    new_train_X = generate_new_train_X2(train_X, train_y, M, sample_index)
    new_train_y = generate_new_train_y2(train_y, M, sample_index)
    # =========================================================
    # Generate S_in. Since each figure is a 2D array, new_train_M[:M] is a 3D array. We reshpae the 3D array into 2D array.
    S_in = new_train_X[:M].reshape(-1, N_in)
    # Generate the corresponding S_out, which is a fixed 2D-array, during this epoch of training.
    S_out = to_categorical(new_train_y[:M])
    return (S_in, S_out)


def mergesort(arr1d):
    """mergesort is stable, O(nlogn)."""
    arr = np.sort(arr1d, axis=-1, kind='mergesort')
    return arr


def timsort(arr1d):
    """timsort is stable, O(nlogn)."""
    arr = np.sort(arr1d, axis=-1, kind='timsort')
    return arr


def init_J(L, N):
    """
    Initializes the array J with random normalized values while ensuring
    the constraint np.sum(J[i, j, :] * J[i, j, :]) = N is satisfied exactly.

    Parameters:
    - L: Number of layers
    - N: Size of each layer

    Returns:
    - J: 3D numpy array of shape (L, N, N) with normalized values
    """
    # Generate random normal numbers for all layers and neurons at once
    J = np.random.normal(loc=0, scale=1, size=(L, N, N)).astype('float32')

    # Compute norms and apply normalization to meet the constraint
    norms = np.sqrt(np.sum(J ** 2, axis=-1, keepdims=True))
    J = J * np.sqrt(N) / norms

    return J


def init_J_v00(L, N):
    """This function normalize the generated array J[i,j,:] to sqrt(tmp), and
     then make sure the constraint J[i,j,:]*J[i,j,:]=N are satisfied exactly."""
    J = np.zeros(
        (L, N, N))  # axis=1 labels the backward-nodes (标记后一层结点), while axis=2 labels the forward-nodes (标记前一层结点)
    # Set the first layer J_{0,x}^{y} = 0 (x, y are any index.)
    for i in range(1, L):
        for j in range(N):
            # First, generate N random numbers
            J[i, j, :] = generate_rand_normal_numbers(N)
            # Add the constraint np.sum(J[i,j,:] * J[i,j,:]) = N.
            tmp = np.sum(J[i, j, :] * J[i, j, :])
            J[i, j, :] = (J[i, j, :] / np.sqrt(tmp)) * np.sqrt(N)
    return J


def init_J_in(N, N_in):
    """
    Initializes and normalizes the J_in array for input connections.

    Parameters:
    - N: Number of nodes in the first layer
    - N_in: Number of nodes in the input layer

    Returns:
    - J_in: A normalized array of shape (N, N_in)
    """
    # Generate random normal numbers for the entire array at once
    J_in = np.random.normal(loc=0, scale=1, size=(N, N_in)).astype('float32')

    # Compute norms for each row
    norms = np.sqrt(np.sum(J_in ** 2, axis=1, keepdims=True))

    # Normalize to satisfy the constraint np.sum(J_in[j, :]**2) = N_in
    J_in = J_in * np.sqrt(N_in) / norms

    return J_in


def init_J_out(N_out, N):
    """
    Initializes and normalizes the J_out array for output connections.

    Parameters:
    - N_out: Number of neurons (or classes) in the last layer
    - N: Number of neurons in the second-last layer

    Returns:
    - J_out: A normalized array of shape (N_out, N)
    """
    # Generate random normal numbers for the entire array at once
    J_out = np.random.normal(loc=0, scale=1, size=(N_out, N)).astype('float32')

    # Compute norms for each row
    norms = np.sqrt(np.sum(J_out ** 2, axis=1, keepdims=True))

    # Normalize to satisfy the constraint np.sum(J_out[j, :]**2) = N
    J_out = J_out * np.sqrt(N) / norms

    return J_out


def init_S(M, L, N):
    """
    Initialize the 3D array S with random -1 or 1 values.

    Parameters:
    - M: Number of samples
    - L: Number of hidden layers
    - N: Number of neurons in each layer

    Returns:
    - S: 3D numpy array with shape (M, L, N) containing values -1 or 1
    """
    # Directly generate random -1 or 1 values using numpy
    S = np.random.choice([-1, 1], size=(M, L, N)).astype('int8')
    return S


def ener_for_mu(r_mu):
    '''Ref: Yoshino2019, eqn (31a)
       energy for single sample (mu).'''
    H_mu = soft_core_potential(r_mu).sum()
    return H_mu


def soft_core_potential(h):
    '''Ref: Yoshino2019, eqn (32)
       This function is tested and it is correct.
    '''
    x2 = 1.0
    return np.heaviside(-h, x2) * np.power(h, 2)
    ## The same as the following.
    # epsilon = 1
    # return epsilon * np.heaviside(-h, x2) * np.power(h,2)
# @njit
# def soft_core_potential(h):
#     x2 = 1.0
#     results = np.empty_like(h)
#     for i in range(h.size):
#         heaviside = 1.0 if h[i] < 0 else x2
#         results[i] = heaviside * h[i]**2
#     return results


def soft_core_potential_torch(h, device='cuda'):
    # Create a constant tensor for x2 (since it's a scalar)
    x2 = torch.tensor(1.0, device=device)

    # Use torch.heaviside and torch.pow for the operation
    return torch.heaviside(-h, x2) * torch.pow(h, 2)


def calc_ener(r):
    '''Ref: Yoshino2019, eqn (31a)'''
    # The argument r can be any array
    H = soft_core_potential(r).sum()
    return H


def calc_ener_torch(r, device):
    H = soft_core_potential_torch(r, device).sum()
    return H


def calc_ener_torch_batch(r, device):
    H = soft_core_potential_torch(r, device).sum(dim=1)
    return H


def calc_ener_batch(r_batch):
    """
    Calculate energy for a batch of inputs.
    Args:
        r_batch: 2D array of shape (batch_size, num_elements),
                 where each row represents an array of distances.
    Returns:
        energies: 1D array of shape (batch_size,), where each entry is the energy for the corresponding batch.
    """
    # Apply the soft core potential function to the entire batch
    H_batch = soft_core_potential(r_batch).sum(axis=1)  # Sum over num_elements for each batch
    return H_batch


def calc_ener_batch_torch(r_batch, device='cuda'):
    H_batch = soft_core_potential_torch(r_batch, device).sum(axis=1)  # Sum over num_elements for each batch
    return H_batch


# ======================================================================================================
# The following three functions are used for getting the argument of locations of the initial configurations
# ======================================================================================================
def list_only_dir(directory):
    """This function list the directorys only, under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    full_li = []
    # directory = '../data'
    for item in list_dir:
        li = [directory, item]
        full_li.append("/".join(li))
    return full_li


def list_only_naked_dir(directory):
    """This function list the naked directory names only, under a given direcotry."""
    import os
    list_dir = next(os.walk(directory))[1]
    for item in list_dir:
        li = [directory, item]
    return list_dir


def list_only_naked_dir_v2(directory):
    """This function list the naked directory names only, under a given direcotry.
       os.walk is a generator and calling next will get the first result in the form of a 3-tuple (dirpath, dirnames, filenames). Thus the [1] index returns only the dirnames from that tuple.
    """
    import os
    DIRNAMES = 1
    list_subfolders_with_paths = []
    for root, dirs, files in os.walk(directory):
        for term in dirs:
            list_subfolders_with_paths.append(os.path.join(root, term))
        break
    print("list_subfolders_with_paths: {}".format(list_subfolders_with_paths))
    try:
        subfolder_with_path = list_subfolders_with_paths[0]
        splited_str = subfolder_with_path.split("/")
        dir_name = splited_str[-1]
        print("dir_name:{}".format(dir_name))
        return dir_name
    except:
        print("Error: There is no subdirectory. Please make sure there is a subdirectory in the directory data1.")


def str2int(list_dir):
    res = []
    for item in range(len(list_dir)):
        res.append(int(list_dir[item]))
    return res


def digit2unitvec(i, N=10):
    '''
    Convert a number (0 to 9) to a unit vector in 10-D Euclidean space.
    '''
    import numpy as np
    vec = np.zeros(N)
    vec[i] = 1
    return vec


def digit_arr2unitvec_arr(arr, N=10):
    '''
    Convert an array (1D) of numbers (0 to 9) to an array of unit vector in N-Dim Euclidean space.
    N=10, is the total number of digits: 0,1,...,9.
    IMPORTANT NOTE: The function to_categorical(train_y, num_classes = 10) does the same thing.
    '''
    import numpy as np
    M = arr.shape[0]
    vec = np.zeros((M, N))
    for i, term in enumerate(arr):
        vec[i][int(term)] = 1
    return vec


def digit_arr2spindist_arr(arr, N=10):
    '''
    Convert an array (1D) of numbers (0 to 9) to an array of spins. The difference between digit_arr2unitvec_arr() and digit_arr2spindist_arr() can be shown in the following example:
    if the result of digit_arr2unitvec_arr() is [[1,0,0,0,0,0,0,0,0,0],...] then the result of digit_arr2spindist_arr() is [[1,-1,-1,-1,-1,-1,-1,-1,-1,-1],...].
    N=10, is the total number of digits:0,1,...,9.
    '''
    import numpy as np
    M = arr.shape[0]
    vec = -np.ones((M, N))
    for i, term in enumerate(arr):
        vec[i][int(term)] = 1
    return vec


# Calculate the relaxation time tau 
def relaxation_time_p(corr, tot_steps, cc=1 / math.e):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    """
    # from numpy.polynomial import Chebyshev
    from numpy.polynomial import Polynomial
    # Generate x_list
    x_list = np.zeros_like(corr)
    for j, iterm in enumerate(corr):
        x_list[j] = 2 ** j
    corr = corr - cc
    p = Polynomial.fit(x_list, corr, 5, domain=[0, tot_steps], window=[0, 1])
    res = p.roots()
    return res[0]


# Calculate the relaxation time tau
def relaxation_time_p4(corr, tot_steps, num_of_dof, step_list, cc=1 / math.e):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    """
    # from numpy.polynomial import Chebyshev
    from numpy.polynomial import Polynomial
    # critical value of corr, 1/e, about 0.3679
    # Generate x_list
    tot_steps_ = corr.shape[0]
    x_list = np.array(step_list[:tot_steps_]) / num_of_dof

    corr = corr - cc
    p = Polynomial.fit(x_list, corr, 5, domain=[0, tot_steps], window=[0, 1])
    res = p.roots()
    return res[0]


# Calculate the relaxation time tau
def relaxation_time_DATE(corr):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    #NOTE: 
    #WE do NOT use this funtion, because the denominator in some formula may be quite SMALL, then the result will be not accurate!
    """
    cc = 1 / math.e  # critical value of corr, 1/e
    # Generate x_list
    x_list = np.zeros_like(corr)
    for j, iterm in enumerate(corr):
        x_list[j] = 2 ** j

    if min(corr) > cc:
        for i, iterm in enumerate(corr):
            # determine tau by interpotation. 
            return (cc - corr[i]) * (x_list[i - 1] - x_list[i]) / (corr[i - 1] - corr[i]) + x_list[i]
    else:
        for i in range(corr.shape[0]):
            if corr[i] < cc:
                # determine tau by interpotation. 
                return ((corr[i - 1] - cc) * (x_list[i] - x_list[i - 1])) / (corr[i - 1] - corr[i]) + x_list[i - 1]
            else:
                pass


def relaxation_time_v5(corr, step_list, dof, cutoff=1 / math.e):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    #NOTE:
    cc: critical value of corr, default: 1/e
    #WE do NOT use this funtion, because the denominator in some formula may be quite SMALL, then the result will be not accurate!
    """
    cc = cutoff
    tot_steps_ = corr.shape[0]
    x_list = np.array(step_list[:tot_steps_]) / dof
    # Generate x_list
    start = 2
    for i in range(start, corr.shape[0]):
        if corr[i] < cc and (corr[i - 1] > cc and corr[i - 2] > cc):
            # determine tau by interpotation.
            return ((corr[i - 1] - cc) * (x_list[i] - x_list[i - 1])) / (corr[i - 1] - corr[i]) + x_list[i - 1]
        else:
            pass


def IS_SOLID(corr, step_list, dof, cutoff=1 / math.e):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    #NOTE:
    cc: critical value of corr, default: 1/e
    #WE do NOT use this funtion, because the denominator in some formula may be quite SMALL, then the result will be not accurate!
    """
    cc = cutoff
    tot_steps_ = corr.shape[0]
    x_list = np.array(step_list[:tot_steps_]) / dof
    # Generate x_list
    start = 2
    i = tot_steps
    if corr[i] > cc:
        return 1
    else:
        return 0


# Calculate the relaxation time tau (2nd method)
def relaxation_time(corr):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    This method is only defined for 2**k time series.
    """
    # Const.s
    cc = 1 / math.e  # critical value of corr, 1/e
    n = 3

    # Generate x_list
    x_list = np.zeros_like(corr)
    for j, iterm in enumerate(corr):
        x_list[j] = 2 ** j

    if min(corr) > cc:
        # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
        # 1. select an integer n
        # 2. calculate the mean of last n elements of the array x, and y
        x_mean = np.mean(x_list[-n:])
        y_mean = np.mean(corr[-n:])
        # 3. calculat the slope and intersept
        k = np.sum((x_list[-n:] - x_mean) * (corr[-n:] - y_mean)) / np.sum(
            (x_list[-n:] - x_mean) * (x_list[-n:] - x_mean))
        b = y_mean - k * x_mean
        # 4. calculate tau
        tau = (cc - b) / k
        return tau
    else:
        for i in range(corr.shape[0]):
            if corr[i] < cc:
                # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
                # 1. select an integer n
                # 2. calculate the mean of last n elements of the array x, and y
                x_mean = np.mean(x_list[(i + 1) - n:i + 1])
                y_mean = np.mean(corr[(i + 1) - n:i + 1])
                # 3. calculat the slope and intersept
                k = np.sum((x_list[i + 1 - n:i + 1] - x_mean) * (corr[i + 1 - n:i + 1] - y_mean)) / np.sum(
                    (x_list[i + 1 - n:i + 1] - x_mean) * (x_list[i + 1 - n:i + 1] - x_mean))
                b = y_mean - k * x_mean
                # 4. calculate tau
                tau = (cc - b) / k
                return tau


# Calculate the relaxation time tau (4th method)
def relaxation_time_v4(corr, dof, step_list):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    This method is only defined for 2**k time series.
    """
    # Const.s
    cc = 1 / math.e  # critical value of corr, 1/e
    n = 3

    # Generate x_list
    x_list = np.zeros_like(corr)
    tot_steps_ = corr.shape[0]
    x_list = np.array(step_list[:tot_steps_]) / dof

    if min(corr) > cc:
        # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
        # 1. select an integer n
        # 2. calculate the mean of last n elements of the array x, and y
        x_mean = np.mean(x_list[-n:])
        y_mean = np.mean(corr[-n:])
        # 3. calculat the slope and intersept
        k = np.sum((x_list[-n:] - x_mean) * (corr[-n:] - y_mean)) / np.sum(
            (x_list[-n:] - x_mean) * (x_list[-n:] - x_mean))
        b = y_mean - k * x_mean
        # 4. calculate tau
        tau = (cc - b) / k
        return tau / num_of_dof
    else:
        for i in range(corr.shape[0]):
            if corr[i] < cc:
                # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
                # 1. select an integer n
                # 2. calculate the mean of last n elements of the array x, and y
                x_mean = np.mean(x_list[(i + 1) - n:i + 1])
                y_mean = np.mean(corr[(i + 1) - n:i + 1])
                # 3. calculat the slope and intersept
                k = np.sum((x_list[i + 1 - n:i + 1] - x_mean) * (corr[i + 1 - n:i + 1] - y_mean)) / np.sum(
                    (x_list[i + 1 - n:i + 1] - x_mean) * (x_list[i + 1 - n:i + 1] - x_mean))
                b = y_mean - k * x_mean
                # 4. calculate tau
                tau = (cc - b) / k
                return tau / num_of_dof


# Calculate the relaxation time tau (3rd method)
def relaxation_time_v3(corr, num_of_dof):
    """
    Calculate the relaxation time of a correlation function. 
    Input: a 1-array.
    Output: a real value.
    This method is only defined for 2**k time series.
    """
    # Const.s
    cc = 1 / math.e  # critical value of corr, 1/e
    n = 3

    # Generate x_list
    x_list = np.zeros_like(corr)
    for j, iterm in enumerate(corr):
        x_list[j] = int(ratio ** j)

    if min(corr) > cc:
        # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
        # 1. select an integer n
        # 2. calculate the mean of last n elements of the array x, and y
        x_mean = np.mean(x_list[-n:])
        y_mean = np.mean(corr[-n:])
        # 3. calculat the slope and intersept
        k = np.sum((x_list[-n:] - x_mean) * (corr[-n:] - y_mean)) / np.sum(
            (x_list[-n:] - x_mean) * (x_list[-n:] - x_mean))
        b = y_mean - k * x_mean
        # 4. calculate tau
        tau = (cc - b) / k
        return tau / num_of_dof
    else:
        for i in range(corr.shape[0]):
            if corr[i] < cc:
                # Use LSF: https://www.varsitytutors.com/hotmath/hotmath_help/topics/line-of-best-fit
                # 1. select an integer n
                # 2. calculate the mean of last n elements of the array x, and y
                x_mean = np.mean(x_list[(i + 1) - n:i + 1])
                y_mean = np.mean(corr[(i + 1) - n:i + 1])
                # 3. calculat the slope and intersept
                k = np.sum((x_list[i + 1 - n:i + 1] - x_mean) * (corr[i + 1 - n:i + 1] - y_mean)) / np.sum(
                    (x_list[i + 1 - n:i + 1] - x_mean) * (x_list[i + 1 - n:i + 1] - x_mean))
                b = y_mean - k * x_mean
                # 4. calculate tau
                tau = (cc - b) / k
                return tau / num_of_dof


def relaxation_time_lr(Q, step_array_v2, dof, cutoff=1 / np.e, n=6):
    """
    n: number of data points you want to use in linear regression.
    cutoff: 1/e.
    dof: rescaling coefficient for time.
    Q: the array of Features. Q represents overlap, e.g., Q, q, etc.
    step_list_v2: the array of targets.
    Return: a tuple, including relaxation time and a score of lienar regression.
    """
    from sklearn.linear_model import LinearRegression

    tot_steps_ = Q.size  # Number of Features

    X_train = Q[-n:].reshape(-1, 1)  # X_train shoud be a 2D array
    positive_indices = np.where(np.any(X_train > cutoff, axis=1))  # Return of boolean indexing
    X_train = X_train[:][positive_indices]  # New traing set

    y_train = step_array_v2[:tot_steps_][-n:] / dof  # Obtain corresponding target and rescaling it
    y_train = y_train[positive_indices]  # Corresponding new target

    X_test = np.array([[cutoff]])  # Test point

    lr = LinearRegression()  # Generate a linear model
    lr.fit(X_train, y_train)  # Fit

    # The target and score
    tau = lr.coef_[0] * X_test[0, 0] + lr.intercept_
    score = lr.score(X_train, y_train)

    return tau, score


def relaxation_time_lr_neighbors(Q, step_array_v2, dof, cutoff=1 / np.e, n=10):
    """
    n: number of data points you want to use in linear regression, which are the closest point near the cutoff.
    cutoff: 1/e.
    dof: rescaling coefficient for time.
    Q: the array of Features. Q represents overlap, e.g., Q, q, etc.
    step_list_v2: the array of targets.
    Return: a tuple, including relaxation time and a score of lienar regression.
    """
    from sklearn.linear_model import LinearRegression

    tot_steps_ = Q.size  # Number of Features
    ind_neighbors = index_neighbors(Q, cutoff, n)
    X_train = Q[ind_neighbors].reshape(-1, 1)  # X_train shoud be a 2D array
    y_train = step_array_v2[:tot_steps_][ind_neighbors] / dof  # Obtain corresponding target and rescaling it

    X_test = np.array([[cutoff]])  # Test point

    lr = LinearRegression()  # Generate a linear model
    lr.fit(X_train, y_train)  # Fit

    # The target and score
    tau = lr.coef_[0] * X_test[0, 0] + lr.intercept_
    score = lr.score(X_train, y_train)

    return tau, score


def generate_new_train_X(train_X, train_y, M, init):
    """Generate a clean array new_train_X, which have equal number of each digit. 
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init : the configuration index. """
    Nout = 10  # The number of classes of digits
    a = int(M / Nout)
    train_X0 = train_X[train_y == 0]
    train_X1 = train_X[train_y == 1]
    train_X2 = train_X[train_y == 2]
    train_X3 = train_X[train_y == 3]
    train_X4 = train_X[train_y == 4]
    train_X5 = train_X[train_y == 5]
    train_X6 = train_X[train_y == 6]
    train_X7 = train_X[train_y == 7]
    train_X8 = train_X[train_y == 8]
    train_X9 = train_X[train_y == 9]

    train = np.zeros((M, 28, 28))
    train[0:a] = train_X0[init * a:(init + 1) * a]
    train[a:2 * a] = train_X1[init * a:(init + 1) * a]
    train[2 * a:3 * a] = train_X2[init * a:(init + 1) * a]
    train[3 * a:4 * a] = train_X3[init * a:(init + 1) * a]
    train[4 * a:5 * a] = train_X4[init * a:(init + 1) * a]
    train[5 * a:6 * a] = train_X5[init * a:(init + 1) * a]
    train[6 * a:7 * a] = train_X6[init * a:(init + 1) * a]
    train[7 * a:8 * a] = train_X7[init * a:(init + 1) * a]
    train[8 * a:9 * a] = train_X8[init * a:(init + 1) * a]
    train[9 * a:10 * a] = train_X9[init * a:(init + 1) * a]
    return train


def generate_new_train_y(train_y, M, init):
    """Generate a clean array new_train_y corresponding to the clean array new_train_X.
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init: the configuration index. """
    Nout = 10  # The number of classes of digits
    a = int(M / Nout)
    train_y0 = train_y[train_y == 0]
    train_y1 = train_y[train_y == 1]
    train_y2 = train_y[train_y == 2]
    train_y3 = train_y[train_y == 3]
    train_y4 = train_y[train_y == 4]
    train_y5 = train_y[train_y == 5]
    train_y6 = train_y[train_y == 6]
    train_y7 = train_y[train_y == 7]
    train_y8 = train_y[train_y == 8]
    train_y9 = train_y[train_y == 9]
    train = np.zeros(M)
    train[0:a] = train_y0[init * a:(init + 1) * a]
    train[a: 2 * a] = train_y1[init * a:(init + 1) * a]
    train[2 * a: 3 * a] = train_y2[init * a:(init + 1) * a]
    train[3 * a: 4 * a] = train_y3[init * a:(init + 1) * a]
    train[4 * a: 5 * a] = train_y4[init * a:(init + 1) * a]
    train[5 * a: 6 * a] = train_y5[init * a:(init + 1) * a]
    train[6 * a: 7 * a] = train_y6[init * a:(init + 1) * a]
    train[7 * a: 8 * a] = train_y7[init * a:(init + 1) * a]
    train[8 * a: 9 * a] = train_y8[init * a:(init + 1) * a]
    train[9 * a: 10 * a] = train_y9[init * a:(init + 1) * a]
    return train


def generate_new_train_X2b(train_X, train_y, M, init):
    """Generate a clean array new_train_X, which have equal number of each digit. 
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init : the configuration index. Different init values, implies different input (pictures)."""
    Nout = 2  # The number of classes of digits
    a = int(M / Nout)
    train_X0 = train_X[train_y == 0]
    train_X1 = train_X[train_y == 1]

    train = np.zeros((M, 28, 28))
    print("The init={}".format(init))
    print(train[0:a].shape)
    print(train_X0[init * a:(init + 1) * a].shape)
    train[0:a] = train_X0[init * a:(init + 1) * a]
    train[a:2 * a] = train_X1[init * a:(init + 1) * a]
    return train


def generate_new_train_y2b(train_y, M, init):
    """Generate a clean array new_train_y corresponding to the clean array new_train_X.
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init: the configuration index. """
    Nout = 2  # The number of classes of digits
    a = int(M / Nout)
    train_y0 = train_y[train_y == 0]
    train_y1 = train_y[train_y == 1]
    train = np.zeros(M)
    train[0:a] = train_y0[init * a:(init + 1) * a]
    train[a: 2 * a] = train_y1[init * a:(init + 1) * a]
    return train


def generate_new_train_X2(train_X, train_y, M, init, Nout=2):
    """Generate a clean array new_train_X, which have equal number of 0 and 1. 
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init: the configuration index. 
    Nout: The number of classes of digits: eg, 0 < Nout <= 1
    """
    a = int(M / Nout)
    d = {}
    for i in range(Nout):
        d["train_X{}".format(i)] = train_X[train_y == i]
    train = np.zeros((M, 28, 28))
    for i in range(Nout):
        train[i * a:(i + 1) * a] = d["train_X{}".format(i)][init * a:(init + 1) * a]
        # train[0:a] = train_X0[init*a:(init+1)*a]
        # train[a:2*a] = train_X1[init*a:(init+1)*a]
    return train


def generate_new_train_y2(train_y, M, init, Nout=2):
    """Generate a clean array new_train_y corresponding to the clean array new_train_X, which have equal number of 0 and 1.
    train_X: the train_X from MNIST dataset;
    train_y: the train_y from MNIST dataset;
    M: the total number of samples; 
    init: the configuration index.
    Nout: The number of classes of digits
    """
    a = int(M / Nout)
    train_y0 = train_y[train_y == 0]
    train_y1 = train_y[train_y == 1]
    train = np.zeros(M)
    train[0:a] = train_y0[init * a:(init + 1) * a]
    train[a: 2 * a] = train_y1[init * a:(init + 1) * a]
    return train


def index_neighbors(Q, cutoff, n):
    """求一个1D数组Q中距离临界值cutoff最近的n个点的索引位置."""
    distance = (Q - cutoff) * (Q - cutoff)
    return distance.argsort()[:n]


def sharpen(train_X):
    train_X[train_X > 0] = 1  # An important UPDATE!!


def spinization(train_X):
    train_X[train_X == 0] = -1  # DO this step in the last AT ONCE, for the largest training set!


def df_column_swaps(df, n):
    '''Exchange two columns (except the first and the last columns) of a dataframe randomly n TIMES.'''
    i = list(df.columns)
    D = len(i) - 1  # The last column are LABELS, we do not need to swap it. That is the reason of "-1".
    for _ in range(n):
        column1 = randrange(1, D)  # The first column should not be changed
        column2 = randrange(1, D)
        a, b = i.index(column1), i.index(column2)
        i[b], i[a] = i[a], i[b]
        df = df[i]
    return df


def draw_history(water_level_history, path="../../figures/H_history.png"):
    plt.clf()
    plt.plot(water_level_history)
    plt.title("H over Episodes")
    plt.xlabel("step")
    plt.ylabel("H")
    plt.savefig(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_numbers_with_gap(range_start, range_end, count, min_gap):
    """
    Generate `count` random numbers in the range [range_start, range_end),
    ensuring that the gap between any two numbers is greater than `min_gap`.
    """
    # Validate inputs
    available_slots = range_end - range_start
    max_possible_count = (available_slots + min_gap) // (min_gap + 1)
    if count > max_possible_count:
        raise ValueError(
            f"Cannot generate {count} numbers with a gap of {min_gap} in the range {range_start} to {range_end - 1}."
        )

    # Initialize candidate numbers
    result = np.array([range_start + i * (min_gap + 1) for i in range(count)])
    remain = range_end - result[-1] - 1
    while remain > 0:
        if remain > 1:
            num = np.random.randint(1, remain)
        else:
            num = 1
        if count > 1:
            index = np.random.randint(0, count - 1)
        else:
            index = 0
        result[index:] += num
        remain -= num

    return result


# def random_numbers_with_gap_cp(range_start, range_end, count, min_gap):
#     """
#     Generate `count` random numbers in the range [range_start, range_end),
#     ensuring that the gap between any two numbers is greater than `min_gap`.
#     """
#     # Validate inputs
#     available_slots = range_end - range_start
#     max_possible_count = (available_slots + min_gap) // (min_gap + 1)
#     if count > max_possible_count:
#         raise ValueError(
#             f"Cannot generate {count} numbers with a gap of {min_gap} in the range {range_start} to {range_end - 1}."
#         )

#     # Initialize candidate numbers
#     result = cp.array([range_start + i * (min_gap + 1) for i in range(count)])
#     remain = range_end - result[-1] - 1
#     while remain > 0:
#         if remain > 1:
#             num = cp.random.randint(1, remain)
#         else:
#             num = 1
#         if count > 1:
#             index = cp.random.randint(0, count - 1)
#         else:
#             index = 0
#         result[index:] += num
#         remain -= num

#     return result


if __name__ == '__main__':
    print(random_numbers_with_gap(0, 9, 4, 1))
