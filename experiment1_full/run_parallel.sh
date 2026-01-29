#!/bin/bash
# ============================================================================
# 实验一完整复现 - 并行运行脚本 (非 SLURM 环境)
#
# 使用方法:
#   chmod +x run_parallel.sh
#   ./run_parallel.sh [config] [num_parallel]
#
# 参数:
#   config: paper, medium, quick (默认: paper)
#   num_parallel: 并行运行的样本数 (默认: 4)
#
# 示例:
#   ./run_parallel.sh paper 4    # 完整参数，4个并行
#   ./run_parallel.sh medium 2   # 中等规模，2个并行
#   ./run_parallel.sh quick 8    # 快速测试，8个并行
# ============================================================================

CONFIG=${1:-paper}
NUM_PARALLEL=${2:-4}

# 根据配置确定样本数
case $CONFIG in
    paper)
        NUM_SAMPLES=8
        ;;
    medium)
        NUM_SAMPLES=4
        ;;
    quick)
        NUM_SAMPLES=2
        ;;
    *)
        echo "未知配置: $CONFIG"
        exit 1
        ;;
esac

# 设置输出目录
OUTPUT_DIR="results/experiment1_${CONFIG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR
mkdir -p logs

echo "============================================"
echo "实验一完整复现 - 并行运行"
echo "配置: $CONFIG"
echo "样本数: $NUM_SAMPLES"
echo "并行数: $NUM_PARALLEL"
echo "输出目录: $OUTPUT_DIR"
echo "开始时间: $(date)"
echo "============================================"

# 预热 JIT (只需要一次)
echo "预热 JIT 编译..."
python run_experiment1_full.py --config quick --sample-id 0 --output-dir /tmp/warmup 2>/dev/null &
WARMUP_PID=$!
sleep 30
kill $WARMUP_PID 2>/dev/null
echo "JIT 预热完成"

# 并行运行样本
echo "开始并行运行 $NUM_SAMPLES 个样本..."

for ((i=0; i<NUM_SAMPLES; i+=NUM_PARALLEL)); do
    PIDS=()
    
    for ((j=0; j<NUM_PARALLEL && i+j<NUM_SAMPLES; j++)); do
        SAMPLE_ID=$((i + j))
        echo "启动样本 $SAMPLE_ID..."
        
        python run_experiment1_full.py \
            --config $CONFIG \
            --sample-id $SAMPLE_ID \
            --output-dir $OUTPUT_DIR \
            --skip-warmup \
            > logs/sample_${SAMPLE_ID}.log 2>&1 &
        
        PIDS+=($!)
    done
    
    # 等待当前批次完成
    echo "等待样本 $i - $((i + j - 1)) 完成..."
    for PID in ${PIDS[@]}; do
        wait $PID
    done
    echo "样本 $i - $((i + j - 1)) 完成"
done

echo "============================================"
echo "所有样本运行完成"
echo "============================================"

# 汇总结果
echo "汇总结果..."
python run_experiment1_full.py \
    --config $CONFIG \
    --aggregate-only \
    --output-dir $OUTPUT_DIR

echo "============================================"
echo "实验完成!"
echo "结果保存在: $OUTPUT_DIR"
echo "结束时间: $(date)"
echo "============================================"
