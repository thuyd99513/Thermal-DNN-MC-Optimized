# Manus AI 继续开发指南

本文档为使用 [Manus AI](https://manus.im) 继续开发本项目提供指导。

## 项目背景

本项目实现了热深度学习机器（TDLM）的蒙特卡洛模拟，用于研究深度神经网络中的玻璃态物理特性。

### 已完成的工作

1. ✅ 分析了原始 `Network.py` 的能量计算逻辑
2. ✅ 实现了按层并行的向量化MC更新 (`Network_optimized.py`)
3. ✅ 编写了正确性测试 (`test_correctness.py`)
4. ✅ 验证了能量计算与源代码的一致性

### 待完成的工作

1. ⬜ GPU 加速实现 (PyTorch/CuPy)
2. ⬜ 权重更新的进一步向量化
3. ⬜ 性能基准测试和对比
4. ⬜ 与论文结果的对比验证

## 如何使用 Manus 继续开发

### 步骤 1：关联仓库

在 Manus 任务中，确保已关联本 GitHub 仓库：
- 仓库名：`thuyd99513/Thermal-DNN-MC-Optimized`

### 步骤 2：克隆项目

告诉 Manus：
```
克隆 Thermal-DNN-MC-Optimized 仓库并查看项目结构
```

### 步骤 3：选择开发任务

以下是推荐的开发任务：

#### 任务 A：GPU 加速实现

```
请基于 py_functions/Network_optimized.py，使用 PyTorch 实现 GPU 加速版本。
主要需要：
1. 将 NumPy 数组转换为 PyTorch 张量
2. 将能量计算移到 GPU 上
3. 实现批量的 Metropolis 接受/拒绝
```

#### 任务 B：性能基准测试

```
请对比 Network.py 和 Network_optimized.py 的性能：
1. 在不同网络规模 (M, N, L) 下测试
2. 记录每个 MC 步的耗时
3. 生成性能对比图表
```

#### 任务 C：物理验证

```
请运行 MC 模拟并验证结果：
1. 检查能量是否正确弛豫
2. 计算 overlap 和 autocorrelation
3. 与论文中的图表对比
```

## 关键文件说明

### 核心代码

| 文件 | 说明 | 状态 |
|------|------|------|
| `py_functions/Network.py` | 原始实现 | 参考 |
| `py_functions/Network_optimized.py` | 优化实现 | ⭐ 主要 |
| `py_functions/utilities.py` | 工具函数 | 稳定 |

### 测试文件

| 文件 | 说明 |
|------|------|
| `py_functions/test_correctness.py` | 正确性测试 |
| `py_functions/test_vectorized.py` | 性能测试 |

### 数据目录

| 目录 | 说明 |
|------|------|
| `ir_hf_L_M_N_sample_mp/` | 数据目录模板 |
| `host1_ir_hf_L10_M120_N3_mp/` | 主机模拟实例 |

## 能量计算详解

### 软核势能函数

```python
def soft_core_potential(h):
    return np.heaviside(-h, 1.0) * np.power(h, 2)
```

**物理含义**：当 gap `h < 0` 时，系统受到惩罚（能量增加）。

### Part Gap 方法

当自旋 `S[mu, l, n]` 翻转时，只有 **N+1 个 gap** 受影响：

```python
# part_gap[0]: 前一层到当前层的 gap
part_gap[0] = (J_hidden[l-1, n, :] @ S[mu, l-1, :]) / sqrt(N) * S[mu, l, n]

# part_gap[1:N+1]: 当前层到下一层的 N 个 gap
for n2 in range(N):
    part_gap[1+n2] = (J_hidden[l, n2, :] @ S[mu, l, :]) / sqrt(N) * S[mu, l+1, n2]
```

### Metropolis 接受准则

```python
EPS = 0.000001
if delta_E < EPS:
    accept = True
else:
    prob = np.exp(-delta_E * beta)
    accept = (random() < prob)
```

## 常见问题

### Q: 如何加载数据？

数据文件位于 `ir_hf_L_M_N_sample_mp/data/{timestamp}/` 目录下。需要先运行主机模拟生成初始配置。

### Q: 如何验证优化后代码的正确性？

```bash
cd py_functions
python test_correctness.py
```

所有测试应该通过。

### Q: 如何调整网络参数？

修改 `NetworkOptimized` 的初始化参数：
- `L`: 网络层数
- `M`: 训练样本数
- `N`: 每层隐藏节点数
- `beta`: 逆温度

## 开发建议

1. **保持物理正确性**：任何优化都要通过 `test_correctness.py` 验证
2. **增量开发**：每次只修改一个部分，确保测试通过后再继续
3. **记录性能**：使用 `@timethis` 装饰器记录各函数耗时
4. **文档更新**：修改代码后更新相应文档

## 联系方式

如有问题，请在 GitHub 仓库提交 Issue。
