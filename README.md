# Thermal Deep Learning Machine - Optimized MC Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目是对热深度学习机器（Thermal Deep Learning Machine, TDLM）蒙特卡洛模拟的优化实现，基于论文 *"Liquid and solid layers in a thermal deep learning machine"* (G. Huang et al., arXiv:2506.06789, 2025)。

## 项目概述

该项目将深度神经网络（DNN）的训练过程映射为统计物理系统，使用蒙特卡洛方法研究网络内部的"固-液-固"分层结构。

### 主要特性

- **按层并行向量化更新**：将传统的串行MC更新优化为按层并行的向量化实现
- **精确的能量计算**：严格复现源代码的软核势能函数和 part_gap 方法
- **预期加速 20-100 倍**：通过 NumPy 向量化和 Numba JIT 编译

### 网络结构

```
输入层 (N_in=784) → 隐藏层 (L-1层, 每层N个节点) → 输出层 (N_out=2)
```

默认配置：L=10, M=120, N=3

## 快速开始

### 安装依赖

```bash
pip install numpy numba matplotlib
```

### 基本使用

```python
from py_functions.Network_optimized import NetworkOptimized

# 创建网络实例
net = NetworkOptimized(
    sample_index=1,
    tw=1024,
    L=10,
    M=120,
    N=3,
    N_in=784,
    N_out=2,
    tot_steps=1000,
    beta=66.7,
    timestamp="your_timestamp",
    h=0
)

# 初始化能量 (重要!)
net.set_vars()

# 运行MC模拟
net.mc_main_vectorized(num_steps=1000, verbose=True)

# 保存结果
net.save_results('./output')
```

## 目录结构

```
├── README.md                    # 本文件
├── MANUS_GUIDE.md              # Manus AI 继续开发指南
├── py_functions/               # 核心Python模块
│   ├── Network.py              # 原始实现
│   ├── Network_optimized.py    # 优化后的实现 ⭐
│   ├── test_correctness.py     # 正确性测试
│   ├── utilities.py            # 工具函数
│   └── README_vectorized.md    # 优化版本文档
├── main_gang.py                # 主程序入口
├── figures/                    # 输出图像
├── ir_hf_L_M_N_sample_mp/      # 数据目录模板
└── *.sh                        # SLURM作业脚本
```

## 能量计算

### 软核势能函数

$$V(h) = \begin{cases} h^2 & \text{if } h < 0 \\ 0 & \text{if } h \geq 0 \end{cases}$$

### Gap 定义

$$r = \frac{1}{\sqrt{N}} \left( \sum_k J_{nk} S_k \right) \cdot S_{\text{next}}$$

### 总能量

$$H = \sum_{\mu=1}^{M} \sum_{\text{all gaps}} V(r)$$

## 优化策略

### 按层并行原理

同一层的自旋之间没有直接耦合，因此可以并行更新：

```
对于第 l 层的自旋 S[mu, l, n]：
├── 能量只依赖于相邻层的自旋和权重
├── 同层的 S[mu, l, n1] 和 S[mu, l, n2] 之间无直接耦合
└── 可以一次性计算并更新整层的 M × N 个自旋
```

### 性能对比

| 方法 | 预期加速 |
|------|----------|
| 原始串行实现 | 1x (基准) |
| 按层并行 + NumPy | 10-20x |
| + Numba JIT | 20-35x |
| + GPU (未来) | 100-1000x |

## 测试

```bash
cd py_functions

# 正确性测试
python test_correctness.py

# 性能测试
python test_vectorized.py
```

## 使用 Manus AI 继续开发

本项目支持使用 [Manus AI](https://manus.im) 继续开发。详见 [MANUS_GUIDE.md](MANUS_GUIDE.md)。

### 快速开始

1. 在 Manus 中关联本 GitHub 仓库
2. 告诉 Manus："克隆并继续开发 Thermal-DNN-MC-Optimized 项目"
3. 参考 `MANUS_GUIDE.md` 中的任务建议

## 参考文献

1. G. Huang, L. S. Chan, H. Yoshino, G. Zhang, and Y. Jin, "Liquid and solid layers in a thermal deep learning machine," arXiv:2506.06789, 2025.
2. H. Yoshino, "Replica theory of the rigidity of structural glasses," J. Chem. Phys. 136, 214108 (2012).

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！
