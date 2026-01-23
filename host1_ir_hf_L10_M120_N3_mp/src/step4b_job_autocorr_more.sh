#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64 
#SBATCH -J DigitOverlap

# This script calcuate the average overlap for S and J, over different initial configurations (i.e., different init's). 
B=66.7
I=784
J=2
L=10
M=120
N=3
R=64
S=1200

#R: the number of replicas
srun -n 1 python3 main_autocorr_log_tw_ave_over_sample_tw.py -L $L -M $M -N $N -B $B -I $I -J $J -R $R -S $S
