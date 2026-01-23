#!/bin/bash 
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -J DigitOverlap

module load anaconda/3.7.3-zyq

B=66.7
I=784
J=2
L=10
M=480
N=3
C=2
S=100

OUTPUT_LOG='std_overlap_more_ave_over_sample.log'
srun -n 1 python main_overlap_more_ave_over_sample.py -B 66.7 -S $S -L $L -M $M -N $N -I $I -J $J & 
wait

# To submit a job, use `sbatch job.sh` 
