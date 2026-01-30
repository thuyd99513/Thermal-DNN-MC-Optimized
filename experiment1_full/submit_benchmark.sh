#!/bin/bash
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J benchmark
#SBATCH -o logs/benchmark_%j.out
#SBATCH -e logs/benchmark_%j.err

module purge
module load miniforge/24.11
source activate nnsm

# 创建目录
mkdir -p logs

echo "============================================"
echo "S/J 优化加速比测试"
echo "开始时间: $(date)"
echo "节点: $HOSTNAME"
echo "============================================"

# 运行测试 (MCS=100)
python3 benchmark_speedup.py \
    --mc-steps 100 \
    --M 2000 \
    --N 20 \
    --L 10 \
    --output-dir .

echo "============================================"
echo "测试完成"
echo "结束时间: $(date)"
echo "============================================"
