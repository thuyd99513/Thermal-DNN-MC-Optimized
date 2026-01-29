# 实验一完整复现 - 液固相图构建 (MC版本)

本目录包含复现论文 "Liquid and solid layers in a thermal deep learning machine" 中 Figure 1b 的完整代码。

## 论文参数 vs 实现参数

| 参数 | 论文 (MD) | 本实现 (MC) | 说明 |
|:-----|:---------|:------------|:-----|
| L (层数) | 10 | 10 | ✓ 一致 |
| M (样本数) | 2000 | 2000 | ✓ 一致 |
| N (神经元数) | 5, 10, 20, 40 | 5, 10, 20, 40 | ✓ 一致 |
| ln α 范围 | 3.91 ~ 5.99 | 3.91 ~ 5.99 | ✓ 一致 |
| β (逆温度) | 10^5 | 10^5 | ✓ 一致 |
| 模拟步数 | 5×10^5 (MD) | 5×10^5 (MC) | 类型不同但规模一致 |
| N_s (独立样本) | 8 | 8 | ✓ 一致 |
| N_r (副本数) | 6-40 | 8 | 略有简化 |

## 预计运行时间

基于测试估算（单核 CPU）：

| 配置 | 单个样本 | 全部样本 (8个) | 并行 (4核) |
|:-----|:--------|:--------------|:-----------|
| quick | ~30 分钟 | ~4 小时 | ~1 小时 |
| medium | ~4 小时 | ~32 小时 | ~8 小时 |
| **paper** | **~20 小时** | **~160 小时** | **~40 小时** |

**建议**：使用 HPC 集群并行运行，或使用 `medium` 配置进行验证。

## 快速开始

### 1. 安装依赖

```bash
pip install numpy numba matplotlib
```

### 2. 快速测试 (验证代码正确性)

```bash
cd experiment1_full
python run_experiment1_full.py --config quick
```

### 3. 完整运行

#### 方式一：顺序运行 (单机)

```bash
./run_sequential.sh paper
```

#### 方式二：并行运行 (多核)

```bash
./run_parallel.sh paper 4  # 4 个并行进程
```

#### 方式三：SLURM 集群

```bash
sbatch submit_slurm.sh
```

## 文件说明

```
experiment1_full/
├── README.md                    # 本文件
├── run_experiment1_full.py      # 主程序
├── run_sequential.sh            # 顺序运行脚本
├── run_parallel.sh              # 并行运行脚本
├── submit_slurm.sh              # SLURM 作业脚本
└── results/                     # 输出结果目录
    └── experiment1_paper_YYYYMMDD/
        ├── config.json          # 配置文件
        ├── sample_00_N_05_result.json  # 样本结果
        ├── sample_00_N_10_result.json
        ├── ...
        └── aggregated_results.json     # 汇总结果
```

## 输出格式

### 单个样本结果 (`sample_XX_N_YY_result.json`)

```json
{
  "N": 5,
  "ln_alpha": 5.99,
  "q_l_star": [0.72, 0.65, 0.45, 0.32, 0.28, 0.31, 0.42, 0.61, 0.70],
  "q_l_std": [0.05, 0.06, 0.08, 0.07, 0.06, 0.07, 0.08, 0.06, 0.05],
  "final_energy": 123.45,
  "simulation_time": 72000.5
}
```

### 汇总结果 (`aggregated_results.json`)

```json
{
  "config": { ... },
  "timestamp": "2026-01-29T12:00:00",
  "results": {
    "5": {
      "N": 5,
      "ln_alpha": 5.99,
      "q_l_mean": [0.72, 0.65, ...],
      "q_l_std": [0.03, 0.04, ...],
      "q_l_sem": [0.01, 0.01, ...],
      "num_samples": 8
    },
    "10": { ... },
    "20": { ... },
    "40": { ... }
  }
}
```

## 运行后的数据分析

完成运行后，将 `aggregated_results.json` 文件提供给 Manus AI，我将为您：

1. 绘制 Figure 1b 风格的相图热图
2. 绘制各层 q_l* 随 ln α 的变化曲线
3. 与论文结果进行定量比较
4. 分析固-液-固分层结构

## 配置选项详解

### `--config paper` (完整论文参数)

```python
{
    'L': 10,
    'M': 2000,
    'N_values': [5, 10, 20, 40],
    'beta': 1e5,
    'mc_steps': 500000,
    'num_samples': 8,
    'num_replicas': 8,
}
```

### `--config medium` (中等规模)

```python
{
    'L': 10,
    'M': 1000,
    'N_values': [5, 10, 20, 40],
    'beta': 1e5,
    'mc_steps': 200000,
    'num_samples': 4,
    'num_replicas': 6,
}
```

### `--config quick` (快速测试)

```python
{
    'L': 10,
    'M': 500,
    'N_values': [5, 10, 20, 40],
    'beta': 1e5,
    'mc_steps': 50000,
    'num_samples': 2,
    'num_replicas': 4,
}
```

## 常见问题

### Q: 如何恢复中断的运行？

使用 `--sample-id` 参数只运行未完成的样本：

```bash
python run_experiment1_full.py --config paper --sample-id 5 --output-dir results/experiment1_paper_20260129
```

### Q: 如何只汇总已有结果？

```bash
python run_experiment1_full.py --aggregate-only --output-dir results/experiment1_paper_20260129
```

### Q: 内存不足怎么办？

减小 M 值或使用 `medium` 配置：

```bash
python run_experiment1_full.py --config medium
```

### Q: 如何监控运行进度？

查看日志文件：

```bash
tail -f logs/sample_0.log
```

## 物理背景

实验测量的是**层重叠参数** q_l*，定义为：

$$q_l^{ab}(t, t_w) = \frac{1}{N_l M} \sum_{i=1}^{N_l} \sum_{\mu=1}^{M} S_{l,i}^{\mu,a}(t+t_w) \cdot S_{l,i}^{\mu,b}(t+t_w)$$

其中 a, b 是两个不同的副本。

**相变边界**：q* = 1/e ≈ 0.368
- q_l* > 1/e → 固态层 (配置被约束)
- q_l* < 1/e → 液态层 (配置自由)

论文预期在高 ln α 时观察到 "固-液-固" 分层结构。

---

*文档创建时间：2026-01-29*
