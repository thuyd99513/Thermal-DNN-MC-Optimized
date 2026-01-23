#!/bin/bash
#SBATCH -p amd_256 
#SBATCH -N 1
#SBATCH -n  1
#SBATCH -c  64
#SBATCH -J 99CleanIn
#module load /public1/home/sc91981/anaconda3
export PATH=$PATH:/public1/home/sc91981/anaconda3/bin
B=$1 # invers temperature
C=$2 # sample index. Differernt inex will use different pictures in the database.
I=$3 # number of bits for input
J=$4 # number of bits for output
L=$5 # number of layers
M=$6 # number of samples to be stored
N=$7 # num of node at each layer
S=$8 # total steps

OUTPUT_LOG='log.log'
date >> $OUTPUT_LOG
python3 main.py -L $L -M $M -N $N -S $S -B $B -I $I -J $J -C $C >> $OUTPUT_LOG
