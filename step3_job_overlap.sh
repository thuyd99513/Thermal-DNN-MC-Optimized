#!/bin/bash 
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n  1
#SBATCH -c  40
#SBATCH -J ol3

module load anaconda/3.7.3-zyq

#Metadata
OUTPUT_LOG='std.log'

#Fixed parameters
B=66.7
I=784
J=2
L=10
N=3
R=6 # IMPORTANT: PLEASE CHANGE R FOR A NEW SYSTEM. 
S=100
#R: the number of replicas

#Changable parameters
C=21
M=120

# the first sleep time should be larger, here set it as w second.
python3 main_overlap_log_tw_JAN.py -L $L -M $M -N $N -S ${S[0]} -B $B -I $I -J $J -C $C -R $R 
wait
