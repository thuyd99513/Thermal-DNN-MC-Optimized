#!/bin/bash
# ============================================================================
# 实验一完整复现 - 顺序运行脚本
#
# 使用方法:
#   chmod +x run_sequential.sh
#   ./run_sequential.sh [config]
#
# 参数:
#   config: paper, medium, quick (默认: paper)
#
# 示例:
#   ./run_sequential.sh paper    # 完整参数
#   ./run_sequential.sh medium   # 中等规模
#   ./run_sequential.sh quick    # 快速测试
# ============================================================================

CONFIG=${1:-paper}

echo "============================================"
echo "实验一完整复现 - 顺序运行"
echo "配置: $CONFIG"
echo "开始时间: $(date)"
echo "============================================"

# 运行完整实验
python run_experiment1_full.py --config $CONFIG

echo "============================================"
echo "实验完成!"
echo "结束时间: $(date)"
echo "============================================"
