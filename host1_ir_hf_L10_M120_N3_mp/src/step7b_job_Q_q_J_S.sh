#!/bin/bash 
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J OL_clean

L=10
M=240
N=3
C=2
S=128
OUTPUT_LOG='std_Q_q_corr.log'
# the first sleep time should be larger, here set it as 20 s.
python3 grand_Q_J_and_q_S.py -C $C -L $L -S $S -N $N & 
wait

# To submit a job, use `sbatch job.sh` or `u job.sh`
