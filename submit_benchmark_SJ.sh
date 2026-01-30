#!/bin/bash
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J benchmark_SJ
#SBATCH -o logs/benchmark_SJ_%j.out
#SBATCH -e logs/benchmark_SJ_%j.err
#SBATCH --time=01:00:00

module purge
module load miniforge/24.11
source activate nnsm

mkdir -p logs

echo "============================================"
echo "S 和 J 更新优化加速比对比测试"
echo "开始时间: $(date)"
echo "节点: $HOSTNAME"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo "分区: $SLURM_JOB_PARTITION"
echo "============================================"

cd py_functions
python3 benchmark_S_J_speedup.py

echo "============================================"
echo "测试完成"
echo "结束时间: $(date)"
echo "============================================"
