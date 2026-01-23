#!/bin/bash 

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
I=784
J=2
L=10
N=3
R=40 # IMPORTANT: PLEASE CHANGE R FOR A NEW SYSTEM. 
#R: the number of replicas

#Changable parameters
C=1
M=480
# the first sleep time should be larger, here set it as w second.
python3 main_acc.py -L $L -M $M -N $N -I $I -J $J -C $C -R $R 
