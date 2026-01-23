#!/bin/bash
#SBATCH -p vip_33  #NOTE: Use `#SBATCH -p vip_33` please, except you have a reason.
#SBATCH -N  1
#SBATCH -n  1
#SBATCH -c  24  #NOTE: Please DO NOT change this parameter, unless you know that you had set associated ones in the main code.
#SBATCH -J m1n3_100  # Your job's name.
#SBATCH --array=0-1  # You HAVE TO SET by yourself.

##SBATCH -c 1
module purge 
#module load anaconda/3.7.3-zyq
#source /public1/soft/modules/module.sh
module load miniforge/24.11
source activate nnsm

# THE NUMBER OF WORK ITEMS we have, eg, from 0-20. 
N_WORK=2

#======================
# General scripts  HERE
#======================
# calculate how many work units each array task will process
# quotient and remainder of jobs to processes 
quot=$((N_WORK/SLURM_ARRAY_TASK_COUNT))
rem=$((N_WORK-quot*SLURM_ARRAY_TASK_COUNT))
# the first "rem" tasks get one extra iteration
if [ "$SLURM_ARRAY_TASK_ID" -lt "$rem" ]; then
	step=$(( quot+1 ))
	step_min=$(( step*SLURM_ARRAY_TASK_ID ))
else
	step=$quot
	step_min=$(( step*SLURM_ARRAY_TASK_ID + rem ))
fi
step_max=$(( step_min+step-1 ))
# "step_min" is the first work unit we process, and "step_max" is the last
# Loop through all steps from step_min to step_max, and do one unit of
# work each iteration.

#=========================
# Specific scripts  HERE
#=========================
# Define comon parameters
A=0.1
B=66.7
I=784
J=2
L=10
M=120
N=3
S=100
tw=1024


python3 main.py
