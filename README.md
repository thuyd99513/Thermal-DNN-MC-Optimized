# Thermal Deep Learning Machine - Optimized MC Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目是对热深度学习机器（Thermal Deep Learning Machine, TDLM）蒙特卡洛模拟的优化实现，基于论文 *"Liquid and solid layers in a thermal deep learning machine"* (G. Huang et al., arXiv:2506.06789, 2025)。

## 项目概述

该项目将深度神经网络（DNN）的训练过程映射为统计物理系统，使用蒙特卡洛方法研究网络内部的"固-液-固"分层结构。

### 主要特性

- **完全向量化更新 (V3)**：不仅实现了自旋 (S) 的按层并行更新，还实现了权重 (J) 的完全向量化更新。
- **Numba JIT 加速**：核心计算逻辑均采用 Numba 进行即时编译，大幅提升执行效率。
- **精确的能量计算**：严格复现源代码的软核势能函数和 part_gap 方法，误差低于 `1e-14`。
- **显著的性能提升**：相比原始串行实现，完整 MC 步加速比可达 **37x - 95x**。

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

### 基本使用 (V3 优化版)

```python
from py_functions.Network_optimized_v3 import NetworkOptimizedV3, warmup_jit

# 预热 JIT 编译
warmup_jit()

# 创建网络实例
net = NetworkOptimizedV3(M=120, N=3, L=10)

# 运行完整 MC 步 (包含 S 和 J 更新)
net.mc_main_vectorized(num_steps=1000, verbose=True)
```

## 目录结构

```
├── README.md                    # 本文件
├── MANUS_GUIDE.md              # Manus AI 继续开发指南
├── py_functions/               # 核心Python模块
│   ├── Network.py              # 原始实现
│   ├── Network_optimized_v2.py # 仅优化 S 更新的版本
│   ├── Network_optimized_v3.py # S 和 J 均优化的版本 ⭐
│   ├── test_J_correctness.py   # J 更新正确性测试
│   ├── benchmark_full_mc.py    # 完整 MC 步性能测试
│   └── README_vectorized.md    # 优化版本文档
├── reports/                    # 优化报告
│   ├── J_optimization_report.md # J 更新优化详细报告
│   └── ...
└── ...
```

## 优化进度

| 版本 | 优化内容 | 状态 | 加速比 (vs 串行) |
| :--- | :--- | :---: | :---: |
| V1 | 简单向量化 (NumPy) | ✅ 完成 | 10-20x |
| V2 | S 更新完全向量化 + Numba | ✅ 完成 | 70-160x (仅S) |
| **V3** | **S & J 更新完全向量化 + Numba** | ✅ 完成 | **37-95x (完整MC)** |

## 测试

```bash
cd py_functions

# 验证 J 更新正确性
python test_J_correctness.py

# 运行完整 MC 步性能基准测试
python benchmark_full_mc.py
```

## 参考文献

1. G. Huang, L. S. Chan, H. Yoshino, G. Zhang, and Y. Jin, "Liquid and solid layers in a thermal deep learning machine," arXiv:2506.06789, 2025.
2. H. Yoshino, "Replica theory of the rigidity of structural glasses," J. Chem. Phys. 136, 214108 (2012).

## 许可证

MIT License
