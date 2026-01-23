#!/bin/bash 

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
B=66.7
I=784
J=2
L=7
N=6
R=64 # IMPORTANT: PLEASE CHANGE R FOR A NEW SYSTEM. 
S=200
#R: the number of replicas

#Changable parameters
M=120
F=0 # sample_start
G=15 # sample_end
# the first sleep time should be larger, here set it as w second.
python3 main_mean_acc_testset.py -L $L -M $M -N $N -S ${S[0]} -B $B -I $I -J $J -R $R 
mv imag/*dat imag/100samples/
wait
