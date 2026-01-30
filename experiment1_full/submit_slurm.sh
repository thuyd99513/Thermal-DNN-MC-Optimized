#!/bin/bash
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J exp1_full
#SBATCH --array=0-7
#SBATCH -o logs/exp1_%A_%a.out
#SBATCH -e logs/exp1_%A_%a.err

module purge
module load miniforge/24.11
source activate nnsm

# 创建目录
mkdir -p logs

# 设置输出目录 (所有作业共享)
OUTPUT_DIR="results/experiment1_paper_$(date +%Y%m%d)"
mkdir -p $OUTPUT_DIR

# 获取当前样本 ID
SAMPLE_ID=$SLURM_ARRAY_TASK_ID

echo "============================================"
echo "实验一完整复现 - 样本 $SAMPLE_ID"
echo "开始时间: $(date)"
echo "节点: $HOSTNAME"
echo "输出目录: $OUTPUT_DIR"
echo "============================================"

# 运行模拟
python3 run_experiment1_full.py \
    --config paper \
    --sample-id $SAMPLE_ID \
    --output-dir $OUTPUT_DIR \
    --skip-warmup

echo "============================================"
echo "样本 $SAMPLE_ID 完成"
echo "结束时间: $(date)"
echo "============================================"
