#!/bin/bash
#SBATCH --job-name=exp1_full
#SBATCH --output=logs/exp1_%A_%a.out
#SBATCH --error=logs/exp1_%A_%a.err
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# ============================================================================
# 实验一完整复现 - SLURM 作业数组脚本
# 
# 使用方法:
#   sbatch submit_slurm.sh
#
# 这将提交 8 个并行作业，每个作业运行一个独立样本。
# 所有作业完成后，运行汇总脚本。
# ============================================================================

# 创建日志目录
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

# 激活 Python 环境 (根据您的环境修改)
# source /path/to/your/venv/bin/activate
# 或者
# conda activate your_env

# 运行模拟
python run_experiment1_full.py \
    --config paper \
    --sample-id $SAMPLE_ID \
    --output-dir $OUTPUT_DIR \
    --skip-warmup

echo "============================================"
echo "样本 $SAMPLE_ID 完成"
echo "结束时间: $(date)"
echo "============================================"
