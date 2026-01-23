#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64 
#SBATCH -J DigitOverlap

OUTPUT_LOG='std_part_overlap_ave_over_sample.log'
# This script calcuate the average overlap for S and J, over different initial configurations (i.e., different init's). 
B=66.7
I=784
J=2
L=10
M=60
N=3
R=64
S=100

#R: the number of replicas
srun -n 1 python3  main_overlap_log_tw_ave_over_part_sample.py -L $L -M $M -N $N -B $B -I $I -J $J -R $R -S $S
