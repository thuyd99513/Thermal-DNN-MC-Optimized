#!/bin/bash 

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
B=66.7
I=784
J=2
L=10
N=3
R=64 # IMPORTANT: PLEASE CHANGE R FOR A NEW SYSTEM. 
S=2000
#R: the number of replicas

#Changable parameters
C=0
M=120
# the first sleep time should be larger, here set it as w second.
python3 main_acc_tw_one_replica.py -L $L -M $M -N $N -S ${S[0]} -B $B -I $I -J $J -C $C -R $R 
wait
