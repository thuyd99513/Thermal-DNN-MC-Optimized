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

#=========================================
#Import the Module with self-defined class
#=========================================
from Network import l_list, l_S_list, l_S_index_list, step_list_v2 
from utilities import num_dof, num_dof_inner_representation
#from utilities import *

#==============
#Import Modules 
#==============
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib as mpl

tau_min = 0.0
tau_max = 1.0
tau_min_q = 0.001
tau_max_q = 1.0
tmin = 0.001
tmax = 100
ol_min = 0.001
ol_max = 1.001

# Const.s
sample_index_list = [1, 2, 3, 4, 5, 6, 7]
nsample_list = [i*32 for i in sample_index_list]

# Markers
marker_list = ['.','*','o','v','^','<','>','x','|','d','h','+','1','2','3','4','p','P','D']
#==========
# Functions
#==========
def autocorr(x):
    """Correlation Function."""
    x = np.array(x)
    length = np.size(x)
    c = np.ones(length)
    for i in range(1,length):
        c[i]=np.sum(x[:-i]*x[i:])/np.sum(x[:-i]*x[:-i])
    return c
def line2intlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(int(x))
    return res_list
def line2floatlist(line):
    line_split=line.strip().split(' ')
    res_list = []
    for x in line_split:
        res_list.append(float(x))
    return res_list
def get_N_HexCol(N=5):
    """Define N elegant colors and return a list of the N colors. Each element of the list is represented as a string.
       and it can be used as an argument of the kwarg color in plt.plot(), or plt.hist()."""
    import colorsys
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

def plot_autocorr_S(q,index,init,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((ol_min,ol_max))
    x = -1 
    b = -1
    num_color = 9
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color=color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color=color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color=color_list[7],marker='1',label="l=8")
    ax.plot(time_arr[ex:],q[x+9][ex:],color=color_list[8],marker='2',label="l=9")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,l)$")
    ax.set_title("init={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,tw,L,M,N,beta))
    plt.savefig("../imag/Autocorr_S_{:s}_init{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,tw,L,M,N,beta,tot_steps),format='eps')

def print_q(C_S,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list,inner_representation=False):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:
        num = num_dof(L,M,N,N_in,N_out)
    x = -1
    time_arr = np.array(step_list[:tot_steps_])/num
    #Print the data C_S(l,t)
    for l in l_S_index_list[:L-1]: 
        f = open("../data1/C_S_L{:d}_M{:d}_N{:d}_tw{}_l{:d}.dat".format(L,M,N,tw,l),"w")
        f.write("# time     C_S    std\n")
        for i in range(tot_steps_): 
            f.write("{:7.4f}  {:7.6f}  {:7.6f}\n".format(time_arr[i], C_S[l][i], std[l][i])) 
        f.close()
def print_qq(C_J,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list,inner_representation=False):
    """autocorr_J is the average one over different init's."""
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:
        num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    x = -1
    time_arr = np.array(step_list[:tot_steps_])/num
    #Print the data C_J(l,t)
    for l in l_list: 
        f = open("../data1/C_J_L{:d}_M{:d}_N{:d}_tw{:d}_l{:d}.dat".format(L,M,N,tw,l),"w")
        f.write("# time     C_J   std\n")
        for i in range(tot_steps_): 
            f.write("{:7.4f}  {:7.6f}  {:7.6f}\n".format(time_arr[i], C_J[x+l][i], std[x+l][i])) 
        f.close()
def print_qq_in(C_J_in,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list,inner_representation=False):
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:
        num = num_dof(L,M,N,N_in,N_out)
    time_arr = np.array(step_list[:tot_steps_])/num
    f = open("../data1/C_J_in_L{:d}_M{:d}_N{:d}_tw{:d}.dat".format(L,M,N,tw),"w")
    f.write("# time     C_J_in     std\n")
    for i in range(tot_steps_): 
        f.write("{:7.4f}  {:7.6f} {:7.6f}\n".format(time_arr[i], C_J_in[i], std[i])) 
    f.close()
def print_qq_out(C_J_out,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list,inner_representation=False):
    if inner_representation:
        num = num_dof_inner_representation(L,M,N,N_in,N_out)
    else:
        num = num_dof(L,M,N,N_in,N_out)
    time_arr = np.array(step_list[:tot_steps_])/num
    f = open("../data1/C_J_out_L{:d}_M{:d}_N{:d}_tw{:d}.dat".format(L,M,N,tw),"w")
    f.write("# time     C_J_out    std\n")
    for i in range(tot_steps_): 
        f.write("{:7.4f}  {:7.6f} {:7.6f}\n".format(time_arr[i], C_J_out[i], std[i])) 
    f.close()
def print_C_S(C_S,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    x = -1
    time_arr = np.array(step_list[:tot_steps_])/num
    #Print the data q(l,t)
    for l in l_S_list: 
        f = open("../data1/C_S_L{:d}_M{:d}_N{:d}_l{:d}.dat".format(L,M,N,l),"w")
        f.write("# time     C_S\n")
        for i in range(tot_steps_): 
            f.write("{:7.4f}  {:7.6f}\n".format(time_arr[i], C_S[x+l][i])) 
        f.close()
def print_C_J(C_J,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    x = -1
    time_arr = np.array(step_list[:tot_steps_])/num
    #Print the data Q(l,t)
    for l in l_list: 
        f = open("../data1/C_J_L{:d}_M{:d}_N{:d}_l{:d}.dat".format(L,M,N,l),"w")
        f.write("# time     C_J\n")
        for i in range(tot_steps_): 
            f.write("{:7.4f}  {:7.6f}\n".format(time_arr[i], C_J[x+l][i])) 
        f.close()
def plot_ave_autocorr_S(q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.01,1.1))
    x = -1 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ##Ref. line (If want to keep ref. line, uncomment R1-10)
    #A = 0.592924 #R1
    #k = 0.372156 #R2
    #B = 0.405378 #R3
    #l = 3.13494 #R4
    #q_ref = A*np.exp(-k*time_arr) + B*np.exp(-l*time_arr) #R5
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:],q[x+9][ex:],color=color_list[x+9],marker='2',label="l=9")
    #ax.plot(time_arr[ex:],q_ref,color="r",label="Ref.") # R6
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.png".format(index,tw,L,M,N,beta,tot_steps),format='png')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')
    #Print the data q(l,t)
    for l in l_S_list: 
        f = open("../data1/C_S_L{:d}_M{:d}_N{:d}_l{:d}.dat".format(L,M,N,l),"w")
        f.write("# time     C_S\n")
        for i in range(tot_steps_): 
            f.write("{:7.4f}  {:7.6f}\n".format(time_arr[i], q[x+l][i])) 
        f.close()
def plot_ave_doverlap_S(q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    end =-10
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.01,1.1))
    x = -1 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    #Ref. line [Ref Github/Yoshino2G*/ir_hf_L10_M420_init2_mp/data1]
    A = 0.592924
    k = 0.372156
    B = 0.405378
    l = 3.13494
    q_ref = A*np.exp(-k*time_arr) + B*np.exp(-l*time_arr)
    dq = np.zeros_like(q)
    dq = q - q_ref
    dq[dq<0] = 0 # Clean the data
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    #ax.plot(time_arr[ex:end],q[x+1][ex:end]-q_ref[ex:end],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:end],dq[x+1][ex:end],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:end],dq[x+2][ex:end],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:end],dq[x+3][ex:end],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:end],dq[x+4][ex:end],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:end],dq[x+5][ex:end],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:end],dq[x+6][ex:end],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:end],dq[x+7][ex:end],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:end],dq[x+8][ex:end],color=color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:end],dq[x+9][ex:end],color=color_list[x+9],marker='2',label="l=9")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)-q_{ref}$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_dq_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_dq_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')
    kappa_prime = np.zeros(L-1)
    t_prime = np.zeros(L-1)
    cutoff_list = [0.04,0.06,0.08,0.10,0.12,0.14,0.16]
    for cutoff in cutoff_list:
        for l in range(L-1):
            arr = dq[l]
            temp = np.argwhere(arr > cutoff)   
            if temp.size == 0:
                t_prime[l] = 10
            else:
                t_prime[l] = np.min(temp) # But we just obtain the index of time, we have to use the index of time to obtain the time
                t_prime[l] = step_list_v2[int(t_prime[l])]/num # This is time at which deviatime happens
                kappa_prime[l] = 1/t_prime[l]
        np.save("../data1/kappa_prime_S_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_cutoff{:3.2f}.npy".format(tw,L,M,N,beta,tot_steps,cutoff), kappa_prime) 
        np.save("../data1/t_prime_S_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_cutoff{:3.2f}.npy".format(tw,L,M,N,beta,tot_steps,cutoff), t_prime) 
    print("kappa_prime:{}".format(kappa_prime))
def plot_ave_autocorr_S_loglog(q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """autocorr_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.001,1))
    x = -1 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:],q[x+9][ex:],color=color_list[x+9],marker='2',label="l=9")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')
def plot_part_ave_overlap_S_loglog(q_high,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list,nsample_list,layer_index,marker_list=marker_list):
    '''The average one over some but all init's. q_high is a high-dim array which is composed of the average (q) of different samples, eg, 64, 128, 192, 256, etc. 
       The value of layer_index can be 0 to L-3.
    '''
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.001,1))
    num_color = len(nsample_list) 
    color_list = get_N_HexCol(num_color)
    print(color_list)
    print("shape of q_high.{}".format(q_high.shape))
    layer = layer_index + 1
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, term in enumerate(nsample_list):
        print("Number of samples: {}".format(term))
        ax.plot(time_arr[ex:],q_high[i][layer_index][ex:],color=color_list[i],marker=marker_list[i],label=r"{} samples".format(term) )
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d};beta={:3.1f}; l={:d}".format(tw,L,M,N,beta,layer))
    plt.savefig("../imag/part_ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog_layer{:d}.eps".format(index,tw,L,M,N,beta,tot_steps,layer),format='eps')
    plt.savefig("../imag/part_ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog_layer{:d}.pdf".format(index,tw,L,M,N,beta,tot_steps,layer),format='pdf')
def plot_ave_autocorr_S_linear(q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """overlap_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xlim((tmin,tmax))
    plt.ylim((-0.01,1.1))
    x = -1 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:],q[x+9][ex:],color=color_list[x+9],marker='2',label="l=9")
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')

def plot_ave_overlap_S_L3(q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """overlap_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.01,1.1))
    x = 0 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')

def plot_ave_overlap_S_yoshino_setting1(q,index,tw,L,M,N,beta,tot_steps_,tot_steps,step_list):
    """overlap_S is the average one over different init's."""
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.1,1.1))
    x = -1 
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],q[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],q[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],q[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],q[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],q[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],q[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],q[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')

def autocorr_arr(arr):
    """arr is a (T,N,M)-shape array."""
    T = arr.shape[0] # The 0-aixs is for time in log-scale.
    corr= np.zeros(T)
    for delta_t in range(T):
        corr[delta_t] = np.sum(arr[0,:,:] * arr[delta_t,:,:])
    return corr
def autocorr_J(J_traj,J_in_traj,J_out_traj):
    '''autocorr for J_traj C(t,l). Here we bound J_traj, J_in_traj, J_out_traj all together.'''
    T = J_traj.shape[0]
    L = J_traj.shape[1] + 2
    N = J_traj.shape[-1]
    N_in = J_in_traj.shape[-1]
    N_out = J_out_traj.shape[-2]
    N_SQ = N**2
    print("N_in:")
    print(N_in)
    print("N_out:")
    print(N_out)
    res = np.zeros((L,T),dtype='float32')
    for l in range(0,L):
        if l == 0:
            res[l] = autocorr_arr(J_in_traj)/(N*N_in) #DONE: Write an autocorrelation function for an array.
            print(res[l])
            #res[l][t] = np.sum(J_in_traj[t,:,:] * J0_in_traj[t,:,:])/(N*N_in)
            res[l] = round(res[l],5)
        elif l == L-1:
            res[l] = autocorr(J_out_traj)/(N*N_out)
            res[l] = round(res[l],5)
            print(res[l])
        else:
            l0 = l - 1 # l0 is the index in J, ie., J_hidden.
            J_traj_transpose = J_traj.transpose(1,0,2,3) #To make sure autocorr() is usabable here.
            res[l] = autocorr_J(J_traj_transpose[l0])/(N*N)
            res[l] = round(res[l],5)
            print(res[l])

def overlap_J_hidden_v3(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = J_traj.shape[0]
    L_hidden = J_traj.shape[1] 
    N = J_traj.shape[-1]
    res= np.zeros((N,L_hidden,T),dtype='float32')
    for l in range(L_hidden):
        for t in range(T):
            for n2 in range(N):
                res[n2][l][t] = np.sum(J_traj[t,l,n2,:] * J0_traj[t,l,n2,:] )/N
                res[n2][l][t] = round(res[n2][l][t],5)
                #print("res_temp[{}][{}][{}]: Q_hidden".format(n2,l,t))
                #print(res[n2][l][t])
    return res
def autocorr_J_hidden_v2(J_traj):
    '''autocorr for J_traj CJ(t,l).'''
    T = J_traj.shape[0]
    L_hidden = J_traj.shape[1] 
    N = J_traj.shape[-1]
    N_SQ = N**2
    res = np.zeros((L_hidden,T),dtype='float32')
    J_traj_transpose = J_traj.transpose(1,0,2,3)
    for l in range(L_hidden):
        res[l] = autocorr_arr(J_traj_transpose[l])/N_SQ
        for t in range(res.shape[1]):
            res[l][t] = round(res[l][t],5)
    return res
def autocorr_J_in(J_in_traj):
    '''JAN16: autocorr for J_in_traj CJ_in(t,l). J_in_traj's shape (T, N, Nin)'''
    T = J_in_traj.shape[0]
    N = J_in_traj.shape[1]
    Nin = J_in_traj.shape[-1]
    SCALE = N*Nin
    res = np.zeros(T,dtype='float32')
    res = autocorr_arr(J_in_traj)/SCALE
    for t in range(res.shape[0]):
        res[t] = round(res[t],5)
    return res
def autocorr_J_out(J_out_traj):
    '''JAN16: autocorr for J_out_traj CJ_out(t,l). J_out_traj's shape (T, Nout, N)'''
    T = J_out_traj.shape[0]
    Nout = J_out_traj.shape[1]
    N = J_out_traj.shape[-1]
    SCALE = N*Nout
    res = np.zeros(T,dtype='float32')
    res = autocorr_arr(J_out_traj)/SCALE
    for t in range(res.shape[0]):
        res[t] = round(res[t],5)
    return res
def overlap_J_6f(J_traj,J0_traj):
    '''overlap for J_traj and J0_traj Q(t,l).'''
    T = J_traj.shape[0]
    L = J_traj.shape[1] 
    N = J_traj.shape[-1]
    N_SQ = N**2
    res_temp = np.zeros((L_hidden,T,N),dtype='float32')
    res = np.zeros((L,T),dtype='float32')
    for l in range(0,L):
        for t in range(T):
            for n2 in range(N):
                res_temp[l][t][n2] = np.sum(J_traj[t,l,n2,:] * J0_traj[t,l,n2,:])/N
                res_temp[l][t][n2] = round(res_temp[l][t][n2],5)
    res = np.mean(res_temp,axis=2)
    return res

def autocorr_S(S_traj):
    '''autocorr for S_traj  CS(t,l).'''
    T = S_traj.shape[0]
    M = S_traj.shape[1]
    L = S_traj.shape[2]
    N = S_traj.shape[-1]
    NM = N*M
    res = np.zeros((L,T),dtype='float32')
    S_traj_transpose = S_traj.transpose(2,0,1,3)
    for l in range(0,L): # we do not need the spins in the input layer
        res[l] = autocorr_arr(S_traj_transpose[l])/NM
    return res

def plot_autocorr_J_tw(ol0,ol1,ol2,ol3,ol4,ol5,ol6,timestamp,init,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """olX: the overlap of J. The overlap is not averaged over init yet."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 7
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    """
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Autocorr_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Autocorr_J_hidden_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,init,l_index,L,N,beta,tot_steps),format='pdf')

def plot_autocorr_S_tw(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,init,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is not averaged over init yet."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    #ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    #ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    #ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,{:d})$".format(l_index))
    ax.set_title("init={:d};l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(init,l_index,L,M,N,beta))
    plt.savefig("../imag/Autocorr_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,init,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Autocorr_S_{:s}_init{:d}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,init,l_index,L,N,beta,tot_steps),format='pdf')

def plot_autocorr_J_tw_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 7
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    #ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    #ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    #ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')

def plot_autocorr_S_tw_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,ol8,index,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    #ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    #ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    #ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    #ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    #ax.plot(time_arr[ex:],ol8[x+l_index][ex:],color=color_list[8],marker='2',label="tw={}".format(tw_list[8]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_S(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,l_index,L,N,beta,tot_steps),format='pdf')

def plot_overlap_J_tw_X_ave_over_init_and_tw(ol,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    #print("step_list")
    #print(step_list)
    #print("tot_steps_:")
    #print(tot_steps_)
    #print("time_arr.shape")
    #print(time_arr.shape)
    ax.plot(time_arr[ex:],ol[x+l_index][ex:],color=color_list[0],marker='.',label="all tw")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_mean_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_mean_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init_4(ol0,ol1,ol2,ol3,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    """
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='x',label="tw={}".format(tw_list[7]))
    ax.plot(time_arr[ex:],ol8[x+l_index][ex:],color=color_list[8],marker='|',label="tw={}".format(tw_list[8]))
    ax.plot(time_arr[ex:],ol9[x+l_index][ex:],color=color_list[9],marker='d',label="tw={}".format(tw_list[9]))
    ax.plot(time_arr[ex:],ol10[x+l_index][ex:],color=color_list[10],marker='h',label="tw={}".format(tw_list[10]))
    ax.plot(time_arr[ex:],ol11[x+l_index][ex:],color=color_list[11],marker='+',label="tw={}".format(tw_list[11]))
    ax.plot(time_arr[ex:],ol12[x+l_index][ex:],color=color_list[12],marker='1',label="tw={}".format(tw_list[12]))
    ax.plot(time_arr[ex:],ol13[x+l_index][ex:],color=color_list[13],marker='2',label="tw={}".format(tw_list[13]))
    ax.plot(time_arr[ex:],ol14[x+l_index][ex:],color=color_list[14],marker='3',label="tw={}".format(tw_list[14]))
    ax.plot(time_arr[ex:],ol15[x+l_index][ex:],color=color_list[15],marker='4',label="tw={}".format(tw_list[15]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    """
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init_linear(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    #print("step_list")
    #print(step_list)
    #print("tot_steps_:")
    #print(tot_steps_)
    #print("time_arr.shape")
    #print(time_arr.shape)
    #print("ol0.shape")
    #print(ol0.shape)
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    """
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_N_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,N_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="N={}".format(N_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="N={}".format(N_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="N={}".format(N_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="N={}".format(N_list[3]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};beta={:3.1f}".format(l_index,L,M,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init_yoshino_setting1(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N 
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw=0")
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw=4096")
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw=8192")
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw=65536") 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw=262,000")
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw=524,000")
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw=1048,000")
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw=2097,000")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')

def plot_overlap_S_tw_X_ave_over_init_and_tw(ol,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol[x+l_index][ex:],color=color_list[0],marker='.',label="all tw")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_mean_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_mean_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_S_tw_X_ave_over_init_4(ol0,ol1,ol2,ol3,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.001,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='x',label="tw={}".format(tw_list[7]))
    ax.plot(time_arr[ex:],ol8[x+l_index][ex:],color=color_list[8],marker='|',label="tw={}".format(tw_list[8]))
    ax.plot(time_arr[ex:],ol9[x+l_index][ex:],color=color_list[9],marker='d',label="tw={}".format(tw_list[9]))
    ax.plot(time_arr[ex:],ol10[x+l_index][ex:],color=color_list[10],marker='h',label="tw={}".format(tw_list[10]))
    ax.plot(time_arr[ex:],ol11[x+l_index][ex:],color=color_list[11],marker='+',label="tw={}".format(tw_list[11]))
    ax.plot(time_arr[ex:],ol12[x+l_index][ex:],color=color_list[12],marker='1',label="tw={}".format(tw_list[12]))
    ax.plot(time_arr[ex:],ol13[x+l_index][ex:],color=color_list[13],marker='2',label="tw={}".format(tw_list[13]))
    ax.plot(time_arr[ex:],ol14[x+l_index][ex:],color=color_list[14],marker='3',label="tw={}".format(tw_list[14]))
    ax.plot(time_arr[ex:],ol15[x+l_index][ex:],color=color_list[15],marker='4',label="tw={}".format(tw_list[15]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_tw_X_ave_over_init_linear(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="upper right")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_N_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,N_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="N={}".format(N_list[0]))
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="N={}".format(N_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="N={}".format(N_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="N={}".format(N_list[3]))
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};beta={:3.1f}".format(l_index,L,M,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.eps".format(timestamp,l_index,L,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.pdf".format(timestamp,l_index,L,beta,tot_steps),format='pdf')
def plot_overlap_S_tw_X_ave_over_init(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_S_tw_X_ave_over_init_linear(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,index,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    ax.plot(time_arr[ex:],ol4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(time_arr[ex:],ol5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(time_arr[ex:],ol6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(time_arr[ex:],ol7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    ax.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,l_index,L,N,beta,tot_steps),format='pdf')
def plot_overlap_S_N_X_ave_over_init(ol0,ol1,ol2,ol3,index,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,N_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],ol0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(time_arr[ex:],ol1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(time_arr[ex:],ol2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(time_arr[ex:],ol3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};beta={:3.1f}".format(l_index,L,M,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,l_index,L,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,l_index,L,beta,tot_steps),format='pdf')

def plot_tau_J_tw(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_hidden_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,init,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_S_tw(grand_tau,index,init,L,M,N,beta,tot_steps):
    ex = 0
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("init={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(init,L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,init,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_init{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,init,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_J_tw_X_mean(grand_tau,index,L,M,N,beta,tot_steps,marker_list=marker_list):
    """
    The size of first axis of grand_tau should be equal to that of the array tw_list.
    """
    ex = 0 
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    num_color = 4 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[ex:],color=color_list[0],marker=marker_list[0],label="all tw")
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_mean_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_mean_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')
def plot_tau_J_tw_X(grand_tau,index,L,M,N,beta,tot_steps,tw_list,marker_list=marker_list):
    """
    The size of first axis of grand_tau should be equal to that of the array tw_list.
    """
    ex = 0 
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    b = -1
     
    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
   
    num_color = len(tw_list)
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(l_list[ex:],grand_tau[i][ex:],color=color_list[i],marker=marker_list[i],label="tw={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_tw{:s}.eps".format(index,L,M,N,beta,tot_steps,str_tw_list),format='eps')
    plt.savefig("../imag/tau_J_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_tw{:s}.pdf".format(index,L,M,N,beta,tot_steps,str_tw_list),format='pdf')

def plot_tau_S_tw_X(grand_tau,index,L,M,N,beta,tot_steps,tw_list,marker_list=marker_list):
    ex = 0 
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    b = -1
    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
    num_color = len(tw_list) 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(l_list,grand_tau[i][:],color=color_list[i],marker=marker_list[i])
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_tw{:s}.eps".format(index,L,M,N,beta,tot_steps,str_tw_list),format='eps')
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_tw{:s}.pdf".format(index,L,M,N,beta,tot_steps,str_tw_list),format='pdf')
def plot_tau_S_tw_X_mean(grand_tau,index,L,M,N,beta,tot_steps,marker_list=marker_list):
    ex = 0 
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    num_color = 4 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list,grand_tau[:],color=color_list[0],marker=marker_list[0])
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_S$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_mean_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_mean_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_J_tw_8(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_S_tw_8(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    ex = 1
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='',label="tw={}".format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='',label="tw={}".format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='',label="tw={}".format(tw_list[6]))
    ax.plot(l_list[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='',label="tw={}".format(tw_list[7]))
    ax.plot(l_list[ex:],grand_tau[x+9][ex:],color=color_list[8],marker='',label="tw={}".format(tw_list[8]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_J_l(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label='l=5')
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label='l=6')
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color=color_list[6],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_S_l(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="l=5")
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="l=6")
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_J_tw_ave_over_init(grand_tau,L,M,N,beta,tot_steps,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 1
    shape = grand_tau.shape
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    plt.ylabel(r"$\tau_J$")
    ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(L,M,N,beta,tot_steps),format='pdf')

def plot_tau_S_tw_ave_over_init(grand_tau,L,M,N,beta,tot_steps,tw_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 1
    shape = grand_tau.shape
    l_list = [1,2,3,4,5,6,7,8,9]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(l_list[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(l_list[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(l_list[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(l_list[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    ax.plot(l_list[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.plot(l_list[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.plot(l_list[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("l")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_S_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_S_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(L,M,N,beta,tot_steps),format='pdf')

def plot_ggrand_tau_J_ave_over_init_and_tw(tau,l,L,N,beta,alpha_list,marker_list=marker_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))
    num_color = 4 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    #ax.plot(alpha_list[ex:],tau[l],color=color_list[0],marker=marker_list[0],label="all tw")
    ax.plot(alpha_list[ex:],tau[l],color=color_list[0],marker=marker_list[0])
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(l,L,N,beta),format='pdf')
def plot_ggrand_tau_J_tw_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list,marker_list=marker_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))

    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
    b = -1
    num_color = len(tw_list) 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(alpha_list[ex:],tau[i][l],color=color_list[i],marker=marker_list[i],label="tw={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.eps".format(l,L,N,beta,str_tw_list),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.pdf".format(l,L,N,beta,str_tw_list),format='pdf')
def plot_ggrand_Q_J_tw_ave_over_init(tau,l,L,N,beta,alpha_list,tw_list,marker_list=marker_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations.
       we plot the relation between tau(alpha), where tau are the averaged tau over different initial configurations.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    plt.ylim((tau_min,tau_max))

    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
    b = -1
    num_color = len(tw_list) 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(alpha_list[ex:],tau[i][l],color=color_list[i],marker=marker_list[i],label="tw={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$Q[-1]$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_Q_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.eps".format(l,L,N,beta,str_tw_list),format='eps')
    plt.savefig("../imag/Ave_Q_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.pdf".format(l,L,N,beta,str_tw_list),format='pdf')

def plot_ggrand_tau_S_ave_over_init_and_tw(tau,l,L,N,beta,alpha_list,marker_list=marker_list):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all initial configurations."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    num_color = 4 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    #ax.plot(alpha_list[ex:],tau[l],color=color_list[0],marker=marker_list[0],label="all tw")
    ax.plot(alpha_list[ex:],tau[l],color=color_list[0],marker=marker_list[0])
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(l,L,N,beta),format='pdf')
def plot_ggrand_tau_S_tw_ave_over_sample(tau,l,L,N,beta,alpha_list,tw_list,marker_list=marker_list):
    """In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
    b = -1
    num_color = len(tw_list) 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(alpha_list[ex:],tau[i][l],color=color_list[i],marker=marker_list[i],label="tw={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_S$")
    ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.eps".format(l,L,N,beta,str_tw_list),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.pdf".format(l,L,N,beta,str_tw_list),format='pdf')
def plot_ggrand_q_S_tw_ave_over_sample(tau,l,L,N,beta,alpha_list,tw_list,marker_list=marker_list):
    """In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    plt.ylim((tau_min_q,tau_max_q))
    str_tw_list = '_'.join([str(elem) for elem in tw_list]) 
    b = -1
    num_color = len(tw_list) 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(tw_list):
        ax.plot(alpha_list[ex:],tau[i][l],color=color_list[i],marker=marker_list[i],label="tw={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$q[-1]$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l+1,N,beta))
    plt.savefig("../imag/Ave_grand_q_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.eps".format(l,L,N,beta,str_tw_list),format='eps')
    plt.savefig("../imag/Ave_grand_q_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}_tw{:s}.pdf".format(l,L,N,beta,str_tw_list),format='pdf')
def plot_ggrand_tau_J_tw_ave_over_sample_fixed_tw(selected_tau,selected_l_list,L,N,beta,alpha_list,tw_list,tw):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples."""
    ex = 0
    shape = selected_tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    
    plt.ylim((tau_min,tau_max))
    str_l_list = '_'.join([str(elem) for elem in selected_l_list]) 
    b = -1
    num_color = len(selected_l_list)
    color_list = get_N_HexCol(num_color)
    index_tw = tw_list.index(tw)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(selected_l_list):
        ax.plot(alpha_list[ex:],selected_tau[index_tw][i],color=color_list[i],marker=marker_list[i],label="l={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("tw={:d};N={:d}; beta={:3.1f}".format(tw,N,beta))
    plt.savefig("../imag/Ave_grand_tau_J_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_l{:s}.eps".format(tw,L,N,beta,str_l_list),format='eps')
    plt.savefig("../imag/Ave_grand_tau_J_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_l{:s}.pdf".format(tw,L,N,beta,str_l_list),format='pdf')
def plot_ggrand_tau_S_tw_ave_over_sample_fixed_tw(selected_tau,selected_l_list,L,N,beta,alpha_list,tw_list,tw):
    """In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples."""
    ex = 0
    shape = selected_tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    fig = plt.figure()
    
    plt.ylim((tau_min_q,tau_max_q))
    str_l_list = '_'.join([str(elem) for elem in selected_l_list]) 
    num_color = len(selected_l_list) 
    color_list = get_N_HexCol(num_color)
    index_tw = tw_list.index(tw)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    for i, item in enumerate(selected_l_list):
        ax.plot(alpha_list[ex:],selected_tau[index_tw][i],color=color_list[i],marker=marker_list[i],label="l={}".format(item))
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("tw={:d};N={:d}; beta={:3.1f}".format(tw,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_l{:s}.eps".format(tw,L,N,beta,str_l_list),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_l{:s}.pdf".format(tw,L,N,beta,str_l_list),format='pdf')

def plot_ggrand_tau_J_tw_X_ave_over_sample(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples.
       we plot the relation between tau(alpha), where tau are the averaged tau over different samples.
    """
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 6 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(alpha_list[ex:],tau[x+2][l],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color=color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color=color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    """
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_J_hidden_l{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(l,L,N,beta),format='pdf')

def plot_ggrand_tau_S_tw_X_ave_over_sample(tau,l,L,N,beta,alpha_list,tw_list):
    """In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples."""
    ex = 0
    shape = tau.shape
    print("shape of ggrand_tau_S:")
    print(shape)
    #l_list = [1,2,3,4,5,6,7,8]
    fig = plt.figure()
    
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 6 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[x+1][l],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.plot(alpha_list[ex:],tau[x+2][l],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.plot(alpha_list[ex:],tau[x+3][l],color=color_list[2],marker='o',label='tw={}'.format(tw_list[2]))
    ax.plot(alpha_list[ex:],tau[x+4][l],color=color_list[3],marker='v',label='tw={}'.format(tw_list[3]))
    """
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_S$ ($10^4$)")
    #ax.set_title("l={:d};N={:d}; beta={:3.1f}".format(l,N,beta))
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(l,L,N,beta),format='eps')
    plt.savefig("../imag/Ave_grand_tau_S_l{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(l,L,N,beta),format='pdf')

def plot_ggrand_tau_J_tw_X_l_ave_over_sample(tau,l_list,L,N,beta,alpha_list,tw_list,tw_index):
    """1. In functions with the end '_ave_over_init' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples.
       we plot the relation between tau(alpha), where tau are the averaged tau over different samples.
       2. tau includes all the data.
       3. tw_index = 0,1,2,3,4,5,...
    """
    ex = 0
    shape = tau.shape
    l = l_list
    print("shape of ggrand_tau_J:")
    print(shape)
    fig = plt.figure()
    
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[tw_index][l[0]],color=color_list[0],marker='.',label="l=2")
    ax.plot(alpha_list[ex:],tau[tw_index][l[1]],color=color_list[3],marker='*',label="l=5")
    ax.plot(alpha_list[ex:],tau[tw_index][l[2]],color=color_list[6],marker='o',label="l=8")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("tw={:d}; l={:d},{:d},{:d}; N={:d}; beta={:3.1f}".format(tw_list[tw_index],l[0]+1,l[1]+1,l[2]+1,N,beta))
    plt.savefig("../imag/Ave_tau_tw{:d}_J_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_tw{:d}_J_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='pdf')

def plot_ggrand_tau_S_tw_X_l_ave_over_sample(tau,l_list,L,N,beta,alpha_list,tw_llist,tw_index):
    """1. In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples.
       2. tau includes all the data.
       3. tw_index = 0,1,2,3,4,5,...
    """
    ex = 0
    shape = tau.shape
    l = l_list
    print("shape of ggrand_tau_S:")
    print(shape)
    fig = plt.figure()
    
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(alpha_list[ex:],tau[tw_index][l[0]],color=color_list[0],marker='.',label="tw=8192;l=2")
    ax.plot(alpha_list[ex:],tau[tw_index][l[1]],color=color_list[3],marker='*',label="tw=8192;l=5")
    ax.plot(alpha_list[ex:],tau[tw_index][l[2]],color=color_list[6],marker='o',label="tw=8192;l=8")
    plt.legend(loc="lower left")
    plt.xlabel(r"$\alpha$")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("tw={:d}; l={:d},{:d},{:d}; N={:d}; beta={:3.1f}".format(tw_list[tw_index],l[0]+1,l[1]+1,l[2]+1,N,beta))
    plt.savefig("../imag/Ave_tau_tw{:d}_S_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.eps".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='eps')
    plt.savefig("../imag/Ave_tau_tw{:d}_S_hidden_l{:d}{:d}{:d}_L{:d}_N{:d}_beta{:2.0f}.pdf".format(tw_list[tw_index],l[0],l[1],l[2],L,N,beta),format='pdf')

def plot_tau_J_l_ave_over_sample(grand_tau,index,L,M,N,beta,tot_steps,tw_list):
    """1. In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples.
       2. When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    
    plt.ylim((tau_min,tau_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label='l=5')
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label='l=6')
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color=color_list[6],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    #plt.ylabel(r"$\tau_J$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_tau_J_l_hidden_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_tau_S_l_ave_over_sample(grand_tau,index,L,M,N,beta,tot_steps, tw_list):
    """1. In functions with the end '_ave_over_samples' in their name, the argument grand_tau are calculated from the averaged overlaps over all samples.
       2. When l is fixed."""
    ex = 1
    shape = grand_tau.shape
    ls = tw_list
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    
    plt.ylim((tau_min_q,tau_max_q))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    grand_tau = grand_tau.T
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(ls[ex:],grand_tau[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(ls[ex:],grand_tau[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(ls[ex:],grand_tau[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(ls[ex:],grand_tau[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(ls[ex:],grand_tau[x+5][ex:],color=color_list[4],marker='^',label="l=5")
    ax.plot(ls[ex:],grand_tau[x+6][ex:],color=color_list[5],marker='<',label="l=6")
    ax.plot(ls[ex:],grand_tau[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(ls[ex:],grand_tau[x+8][ex:],color=color_list[7],marker='1',label='l=8')
    plt.legend(loc="lower left")
    plt.xlabel("$t_w$")
    #plt.ylabel(r"$\tau_S$")
    #ax.set_title("L={:d};M={:d};N={:d}; beta={:3.1f}".format(L,M,N,beta))
    plt.savefig("../imag/Ave_tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/tau_S_l_{:s}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,L,M,N,beta,tot_steps),format='pdf')

def plot_autocorr_J(Q,index,sample,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps):
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[7],marker='1',label="l=8")
    ax.plot(time_arr[ex:],Q[x+9][ex:],color=color_list[8],marker='2',label="l=9")
    ax.plot(time_arr[ex:],Q[x+10][ex:],color=color_list[9],marker='3',label="l=10")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,l)$")
    #ax.set_title("sample={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(sample,tw,L,M,N,beta))
    plt.savefig("../imag/Autocorr_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,sample,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Autocorr_J_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,sample,tw,L,N,beta,tot_steps),format='pdf')

def num_dof(L,M,N,N_in,N_out):
    SQ_N = N ** 2
    num_hidden_node_layers = L - 1
    num_hidden_bond_layers = L - 2
    num_variables = int(N * M * num_hidden_node_layers)
    num_bonds = int(N * N_in + SQ_N * num_hidden_bond_layers + N_out * N)
    num = num_variables + num_bonds
    return num 
def plot_autocorr_J_hidden(Q,index,sample,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[1],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[2],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[3],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[4],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[5],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[6],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[7],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,l)$")
    #ax.set_title("sample={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(sample,tw,L,M,N,beta))
    plt.savefig("../imag/Autocorr_J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,sample,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Autocorr_J_hidden_{:s}_init{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,sample,tw,L,N,beta,tot_steps),format='pdf')
def plot_overlap_J_hidden_L3(Q,index,sample,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[0],marker='.',label="l=1")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q_h(t,l)$")
    #ax.set_title("sample={:d};tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(sample,tw,L,M,N,beta))
    plt.savefig("../imag/Overlap_J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,sample,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Overlap_J_hidden_{:s}_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,sample,tw,L,N,beta,tot_steps),format='pdf')

def plot_ave_autocorr_J(Q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different samples."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[8],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')


def plot_ave_autocorr_J_loglog(Q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.0001,1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[8],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_loglog.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')
def plot_ave_autocorr_J_linear(Q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[8],marker='1',label="l=8")
    plt.legend(loc="upper right")
    #plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$C_J(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_autocorr_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')
def plot_ave_overlap_J_L3(Q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[1],marker='.',label="l=1")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')

def plot_ave_overlap_J_yoshino_setting1(Q,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((ol_min,ol_max))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(time_arr[ex:],Q[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.plot(time_arr[ex:],Q[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.plot(time_arr[ex:],Q[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.plot(time_arr[ex:],Q[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.plot(time_arr[ex:],Q[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.plot(time_arr[ex:],Q[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.plot(time_arr[ex:],Q[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.plot(time_arr[ex:],Q[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    ax.plot(time_arr[ex:],Q[x+9][ex:],color=color_list[x+9],marker='2',label="l=9")
    ax.plot(time_arr[ex:],Q[x+10][ex:],color=color_list[x+10],marker='3',label="l=10")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    #ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')

def plot_test_delta_e_bond(s0,s1,s2):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[0],marker='.',label="bond in")
    ax.plot(s1[-10000:],color=color_list[1],marker='*',label="bond hidden")
    ax.plot(s2[-10000:],color=color_list[2],marker='o',label="bond out")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_bond.eps",format='eps')
    plt.savefig("../imag/Delta_E_bond.pdf",format='pdf')

def plot_test_delta_e_node(s0,s1,s2):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[0],marker='.',label="node in")
    ax.plot(s1[-10000:],color=color_list[1],marker='*',label="node hidden")
    ax.plot(s2[-10000:],color=color_list[2],marker='o',label="node out")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_node.eps",format='eps')
    plt.savefig("../imag/Delta_E_node.pdf",format='pdf')

def plot_test_delta_e_bond_node_hidden(s0,s1):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[0],marker='.',label="bond hidden")
    ax.plot(s1[-10000:],color=color_list[1],marker='*',label="node hidden")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_bond_node_hidden.eps",format='eps')
    plt.savefig("../imag/Delta_E_bond_node_hidden.pdf",format='pdf')

def plot_test_delta_e_bond_hidden(s0):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[0],marker='.',label="bond hidden")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_bond_hidden.eps",format='eps')
    plt.savefig("../imag/Delta_E_bond_hidden.pdf",format='pdf')
def plot_test_delta_e_node_hidden(s0):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[1],marker='*',label="node hidden")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_node_hidden.eps",format='eps')
    plt.savefig("../imag/Delta_E_node_hidden.pdf",format='pdf')

def plot_test_delta_e_bond_in(s0):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[0],marker='.',label="bond in")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_bond_in.eps",format='eps')
    plt.savefig("../imag/Delta_E_bond_in.pdf",format='pdf')
def plot_test_delta_e_node_in(s0):
    fig = plt.figure()
    plt.xscale('log')
    num_color = 7 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.plot(s0[-10000:],color=color_list[1],marker='*',label="node in")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$\Delta e$")
    plt.savefig("../imag/Delta_E_node_in.eps",format='eps')
    plt.savefig("../imag/Delta_E_node_in.pdf",format='pdf')
def errorbar_ave_overlap_S(q,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """overlap_S is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    #num = M*L*N + L*N*N #Maybe this formula is also possible sometime.
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.ylim((-0.01,1.1))
    x = -1 
    b = -1
    num_color = 9 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.errorbar(time_arr[ex:],q[x+1][ex:],yerr=std[x+1][ex:],color=color_list[x+1],marker='.',label="l=1")
    ax.errorbar(time_arr[ex:],q[x+2][ex:],yerr=std[x+2][ex:],color=color_list[x+2],marker='*',label="l=2")
    ax.errorbar(time_arr[ex:],q[x+3][ex:],yerr=std[x+3][ex:],color=color_list[x+3],marker='o',label="l=3")
    ax.errorbar(time_arr[ex:],q[x+4][ex:],yerr=std[x+4][ex:],color=color_list[x+4],marker='v',label="l=4")
    ax.errorbar(time_arr[ex:],q[x+5][ex:],yerr=std[x+5][ex:],color=color_list[x+5],marker='^',label='l=5')
    ax.errorbar(time_arr[ex:],q[x+6][ex:],yerr=std[x+6][ex:],color=color_list[x+6],marker='<',label='l=6')
    ax.errorbar(time_arr[ex:],q[x+7][ex:],yerr=std[x+7][ex:],color=color_list[x+7],marker='>',label='l=7')
    ax.errorbar(time_arr[ex:],q[x+8][ex:],yerr=std[x+8][ex:],color=color_list[x+8],marker='1',label="l=8")
    ax.errorbar(time_arr[ex:],q[x+9][ex:],yerr=std[x+9][ex:],color=color_list[x+9],marker='2',label="l=9")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(index,tw,L,M,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(index,tw,L,M,N,beta,tot_steps),format='pdf')
def errorbar_overlap_J_tw_X_ave_over_samples_4(ol0,ol1,ol2,ol3,std0,std1,std2,std3,std4,std5,std6,std7,std8,std9,std10,std11,std12,std13,std14,std15,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.errorbar(time_arr[ex:],ol0[x+l_index][ex:],yerr=std0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.errorbar(time_arr[ex:],ol1[x+l_index][ex:],yerr=std1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.errorbar(time_arr[ex:],ol2[x+l_index][ex:],yerr=std2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.errorbar(time_arr[ex:],ol3[x+l_index][ex:],yerr=std3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.errorbar(time_arr[ex:],ol4[x+l_index][ex:],yerr=std4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.errorbar(time_arr[ex:],ol5[x+l_index][ex:],yerr=std5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.errorbar(time_arr[ex:],ol6[x+l_index][ex:],yerr=std6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.errorbar(time_arr[ex:],ol7[x+l_index][ex:],yerr=std7[x+l_index][ex:],color=color_list[7],marker='x',label="tw={}".format(tw_list[7]))
    ax.errorbar(time_arr[ex:],ol8[x+l_index][ex:],yerr=std8[x+l_index][ex:],color=color_list[8],marker='|',label="tw={}".format(tw_list[8]))
    ax.errorbar(time_arr[ex:],ol9[x+l_index][ex:],yerr=std9[x+l_index][ex:],color=color_list[9],marker='d',label="tw={}".format(tw_list[9]))
    ax.errorbar(time_arr[ex:],ol10[x+l_index][ex:],yerr=std10[x+l_index][ex:],color=color_list[10],marker='h',label="tw={}".format(tw_list[10]))
    ax.errorbar(time_arr[ex:],ol11[x+l_index][ex:],yerr=std11[x+l_index][ex:],color=color_list[11],marker='+',label="tw={}".format(tw_list[11]))
    ax.errorbar(time_arr[ex:],ol12[x+l_index][ex:],yerr=std12[x+l_index][ex:],color=color_list[12],marker='1',label="tw={}".format(tw_list[12]))
    ax.errorbar(time_arr[ex:],ol13[x+l_index][ex:],yerr=std13[x+l_index][ex:],color=color_list[13],marker='2',label="tw={}".format(tw_list[13]))
    ax.errorbar(time_arr[ex:],ol14[x+l_index][ex:],yerr=std14[x+l_index][ex:],color=color_list[14],marker='3',label="tw={}".format(tw_list[14]))
    ax.errorbar(time_arr[ex:],ol15[x+l_index][ex:],yerr=std15[x+l_index][ex:],color=color_list[15],marker='4',label="tw={}".format(tw_list[15]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def errorbar_overlap_S_tw_X_ave_over_init_4(ol0,ol1,ol2,ol3,std0,std1,std2,std3,std4,std5,std6,std7,std8,std9,std10,std11,std12,std13,std14,std15,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 4
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.errorbar(time_arr[ex:],ol0[x+l_index][ex:],yerr=std0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.errorbar(time_arr[ex:],ol1[x+l_index][ex:],yerr=std1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.errorbar(time_arr[ex:],ol2[x+l_index][ex:],yerr=std2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.errorbar(time_arr[ex:],ol3[x+l_index][ex:],yerr=std3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.errorbar(time_arr[ex:],ol4[x+l_index][ex:],yerr=std4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.errorbar(time_arr[ex:],ol5[x+l_index][ex:],yerr=std5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.errorbar(time_arr[ex:],ol6[x+l_index][ex:],yerr=std6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.errorbar(time_arr[ex:],ol7[x+l_index][ex:],yerr=std7[x+l_index][ex:],color=color_list[7],marker='x',label="tw={}".format(tw_list[7]))
    ax.errorbar(time_arr[ex:],ol8[x+l_index][ex:],yerr=std8[x+l_index][ex:],color=color_list[8],marker='|',label="tw={}".format(tw_list[8]))
    ax.errorbar(time_arr[ex:],ol9[x+l_index][ex:],yerr=std9[x+l_index][ex:],color=color_list[9],marker='d',label="tw={}".format(tw_list[9]))
    ax.errorbar(time_arr[ex:],ol10[x+l_index][ex:],yerr=std10[x+l_index][ex:],color=color_list[10],marker='h',label="tw={}".format(tw_list[10]))
    ax.errorbar(time_arr[ex:],ol11[x+l_index][ex:],yerr=std11[x+l_index][ex:],color=color_list[11],marker='+',label="tw={}".format(tw_list[11]))
    ax.errorbar(time_arr[ex:],ol12[x+l_index][ex:],yerr=std12[x+l_index][ex:],color=color_list[12],marker='1',label="tw={}".format(tw_list[12]))
    ax.errorbar(time_arr[ex:],ol13[x+l_index][ex:],yerr=std13[x+l_index][ex:],color=color_list[13],marker='2',label="tw={}".format(tw_list[13]))
    ax.errorbar(time_arr[ex:],ol14[x+l_index][ex:],yerr=std14[x+l_index][ex:],color=color_list[14],marker='3',label="tw={}".format(tw_list[14]))
    ax.errorbar(time_arr[ex:],ol15[x+l_index][ex:],yerr=std15[x+l_index][ex:],color=color_list[15],marker='4',label="tw={}".format(tw_list[15]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def errorbar_overlap_S_tw_X_ave_over_samples(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,std0,std1,std2,std3,std4,std5,std6,std7,index,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of S. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.errorbar(time_arr[ex:],ol0[x+l_index][ex:],yerr=std0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.errorbar(time_arr[ex:],ol1[x+l_index][ex:],yerr=std1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.errorbar(time_arr[ex:],ol2[x+l_index][ex:],yerr=std2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.errorbar(time_arr[ex:],ol3[x+l_index][ex:],yerr=std3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3])) 
    ax.errorbar(time_arr[ex:],ol4[x+l_index][ex:],yerr=std4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.errorbar(time_arr[ex:],ol5[x+l_index][ex:],yerr=std5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.errorbar(time_arr[ex:],ol6[x+l_index][ex:],yerr=std6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.errorbar(time_arr[ex:],ol7[x+l_index][ex:],yerr=std7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(index,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_S_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(index,l_index,L,N,beta,tot_steps),format='pdf')
def errorbar_overlap_J_tw_X_ave_over_samples(ol0,ol1,ol2,ol3,ol4,ol5,ol6,ol7,std0,std1,std2,std3,std4,std5,std6,std7,timestamp,l_index,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,tw_list,step_list):
    """olX: the overlap of J. The overlap is the averaged overlap over initial configurations."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0 
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((-0.1,1.1))
    x = -1 # Define a tempary integer
    num_color = 8
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    """
    print("step_list")
    print(step_list)
    print("tot_steps_:")
    print(tot_steps_)
    print("time_arr.shape")
    print(time_arr.shape)
    print("ol0.shape")
    print(ol0.shape)
    """
    ax.errorbar(time_arr[ex:],ol0[x+l_index][ex:],yerr=std0[x+l_index][ex:],color=color_list[0],marker='.',label="tw={}".format(tw_list[0]))
    """
    ax.errorbar(time_arr[ex:],ol1[x+l_index][ex:],yerr=std1[x+l_index][ex:],color=color_list[1],marker='*',label="tw={}".format(tw_list[1]))
    ax.errorbar(time_arr[ex:],ol2[x+l_index][ex:],yerr=std2[x+l_index][ex:],color=color_list[2],marker='o',label="tw={}".format(tw_list[2]))
    ax.errorbar(time_arr[ex:],ol3[x+l_index][ex:],yerr=std3[x+l_index][ex:],color=color_list[3],marker='v',label="tw={}".format(tw_list[3]))
    ax.errorbar(time_arr[ex:],ol4[x+l_index][ex:],yerr=std4[x+l_index][ex:],color=color_list[4],marker='^',label="tw={}".format(tw_list[4]))
    ax.errorbar(time_arr[ex:],ol5[x+l_index][ex:],yerr=std5[x+l_index][ex:],color=color_list[5],marker='<',label="tw={}".format(tw_list[5]))
    ax.errorbar(time_arr[ex:],ol6[x+l_index][ex:],yerr=std6[x+l_index][ex:],color=color_list[6],marker='>',label="tw={}".format(tw_list[6]))
    ax.errorbar(time_arr[ex:],ol7[x+l_index][ex:],yerr=std7[x+l_index][ex:],color=color_list[7],marker='1',label="tw={}".format(tw_list[7]))
    """
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,{:d})$".format(l_index))
    ax.set_title("l={:d};L={:d};M={:d};N={:d};beta={:3.1f}".format(l_index,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(timestamp,l_index,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_l{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(timestamp,l_index,L,N,beta,tot_steps),format='pdf')
def errorbar_ave_overlap_J(Q,std,index,tw,L,M,N,N_in,N_out,beta,tot_steps_,tot_steps,step_list):
    """The argument overlap_J is the average one over different init's."""
    num = num_dof(L,M,N,N_in,N_out)
    ex = 0
    time_arr = np.array(step_list[:tot_steps_])/num
    fig = plt.figure()
    plt.xscale('log')
    plt.xlim((tmin,tmax))
    plt.ylim((0.0,1.1))
    x = -1 # Define a tempary integer
    b = -1
    num_color = 10 
    color_list = get_N_HexCol(num_color)
    ax = fig.add_subplot(111) # add_subplot() adds an axes to a figure, it returns a (subclass of a) matplotlib.axes.Axes object.
    ax.errorbar(time_arr[ex:],Q[x+1][ex:],yerr=std[x+1][ex:],color=color_list[1],marker='.',label="l=1")
    ax.errorbar(time_arr[ex:],Q[x+2][ex:],yerr=std[x+2][ex:],color=color_list[2],marker='*',label="l=2")
    ax.errorbar(time_arr[ex:],Q[x+3][ex:],yerr=std[x+3][ex:],color=color_list[3],marker='o',label="l=3")
    ax.errorbar(time_arr[ex:],Q[x+4][ex:],yerr=std[x+4][ex:],color=color_list[4],marker='v',label="l=4")
    ax.errorbar(time_arr[ex:],Q[x+5][ex:],yerr=std[x+5][ex:],color=color_list[5],marker='^',label='l=5')
    ax.errorbar(time_arr[ex:],Q[x+6][ex:],yerr=std[x+6][ex:],color=color_list[6],marker='<',label='l=6')
    ax.errorbar(time_arr[ex:],Q[x+7][ex:],yerr=std[x+7][ex:],color=color_list[7],marker='>',label='l=7')
    ax.errorbar(time_arr[ex:],Q[x+8][ex:],yerr=std[x+8][ex:],color=color_list[8],marker='1',label="l=8")
    #plt.legend(loc="upper right")
    plt.legend(loc="lower left")
    plt.xlabel("t")
    plt.ylabel(r"$Q(t,l)$")
    ax.set_title("tw={:d};L={:d};M={:d};N={:d}; beta={:3.1f}".format(tw,L,M,N,beta))
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.eps".format(index,tw,L,N,beta,tot_steps),format='eps')
    plt.savefig("../imag/Ave_overlap_J_hidden_{:s}_tw{:d}_L{:d}_N{:d}_beta{:2.0f}_step{:d}_log_errorbar.pdf".format(index,tw,L,N,beta,tot_steps),format='pdf')

