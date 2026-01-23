#!/bin/bash 

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
B=66.7
I=784
J=0
L=0
N=0
R=0 # IMPORTANT: PLEASE CHANGE R FOR A NEW SYSTEM. 
S=0
#R: the number of replicas

#Changable parameters
C=0
M=0
# the first sleep time should be larger, here set it as w second.
python3 main_acc_testset.py -L $L -M $M -N $N -S ${S[0]} -B $B -I $I -J $J -C $C -R $R 
wait
