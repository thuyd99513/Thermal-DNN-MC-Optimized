#!/bin/bash 
#SBATCH -p vip_33 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -J DigitOverlap
module load anaconda/3.7.3-zyq

L=10
M=480
N=3
C=2
S=100

OUTPUT_LOG='std_tau_corr.log'
# the first sleep time should be larger, here set it as 20 s.
srun -n 1 python main_tau_of_corr.py -L $L -M $M -N $N -S $S & 

# To submit a job, use `sbatch job.sh` or `u job.sh`
