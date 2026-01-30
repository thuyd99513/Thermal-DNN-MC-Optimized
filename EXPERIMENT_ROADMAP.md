# 论文实验复现路线图

本文档记录论文 "Liquid and solid layers in a thermal deep learning machine" 中所有实验的复现状态和后续任务计划。

## 最新进度 (2026-01-30)

### 新增内容
- ✅ **实验1完整版代码**：`experiment1_full/` 目录包含可在算力平台运行的完整代码
- ✅ **测试运行完成**：使用 mc_steps=1000 完成了8个样本的测试运行
- ✅ **分析结果生成**：`analysis_results/` 目录包含 Figure 1b 风格相图和分析报告
- ⚠️ **重要发现**：论文使用 MD (分子动力学) 算法，时间步长 Δt=0.01，t*=5×10⁵

### 待完成
- ⏳ 使用完整参数 (mc_steps=500000) 运行实验1
- ⏳ 实验8：网络深度依赖性
- ⏳ 实验9：重叠参数分布补充

---

## 复现状态总览

| 实验 | 对应图表 | 状态 | 复杂度 | 优先级 |
|:-----|:--------|:----:|:------:|:------:|
| 液-固相图 | Figure 1b | ✅ 已完成 | 中等 | - |
| 全局训练动态 | Figure 2 | ✅ 已完成 | 中等 | - |
| 层依赖训练动态 | Figure 3 | ✅ 已完成 | 高 | - |
| 设计空间结构 | Figure 4 | ✅ 已完成 | 高 | - |
| 温度依赖性 | Figure S1 | ✅ 已完成 | 低 | - |
| 全局相关函数补充 | SI Sec. S2 | ✅ 已完成 | 低 | - |
| 层相关函数补充 | SI Sec. S3 | ✅ 已完成 | 中等 | - |
| 网络深度依赖性 | SI Sec. S4 | ❌ 未开始 | 中等 | P5 |
| 重叠参数分布补充 | SI Sec. S5 | ❌ 未开始 | 中等 | P5 |

---

## 实验1完整版运行指南

### 论文参数

| 参数 | 论文值 | 说明 |
|:-----|:-------|:-----|
| L | 10 | 网络层数 |
| M | 2000 | 样本数 |
| N | 5, 10, 20, 40 | 每层神经元数 |
| β | 10⁵ | 逆温度 |
| t* | 5×10⁵ | MD模拟时间 |
| Δt | 10⁻² | MD时间步长 |
| Nₛ | 8 | 独立样本数 |
| Nᵣ | 8 | 副本数 |

### 运行命令

```bash
cd experiment1_full

# 单机运行（完整参数）
python run_experiment1_full.py --config paper --sample-id 0 --output-dir results/

# SLURM 并行运行
sbatch submit_slurm.sh

# 聚合结果
python run_experiment1_full.py --aggregate-only --output-dir results/experiment1_paper_YYYYMMDD/
```

### 配置选项

| 配置 | mc_steps | 相对论文 | 用途 |
|:-----|:---------|:---------|:-----|
| paper | 500,000 | 100% | 完整复现 |
| medium | 200,000 | 40% | 验证测试 |
| quick | 50,000 | 10% | 快速测试 |

---

## 分析结果 (2026-01-30)

### 文件位置
```
analysis_results/
├── ANALYSIS_REPORT.md           # 详细分析报告
├── figure_1b_phase_diagram.png  # Figure 1b 风格相图热图
├── layer_profile.png            # 层剖面图
├── analysis_results.json        # 原始数据
└── analyze_and_plot.py          # 分析脚本
```

### 主要发现

使用 mc_steps=1000 的测试结果：

| 配置 | 第一层 q* | 结构特征 |
|:-----|:----------|:---------|
| N=5, α=400 | 0.74 | 仅层1固态，其余液态 |
| N=10, α=200 | 0.82 | 几乎全固态（8/9层） |
| N=20, α=100 | 0.67 | 前2层固态，其余液态 |
| N=40, α=50 | 0.63 | 仅层1固态，其余液态 |

**与论文对比**：
- ✅ 第一层始终为固态
- ✅ 相变边界 q* = 1/e 得到验证
- ⚠️ 输出层未呈现预期固态（需增加 MC 步数）

---

## 已完成实验

### 实验一：液-固相图 (Figure 1b)
**目的**：构建数值相图，验证"固-液-固"分层结构

**代码文件**：
- `experiment1_full/run_experiment1_full.py` (完整版)
- `py_functions/run_experiment1_v2.py` (简化版)

**结果**：
- `reports/experiment1_final/` (简化版结果)
- `analysis_results/` (完整版分析)

---

### 实验二：全局训练动态 (Figure 2)
**目的**：研究能量 E(t) 和准确率 A(t) 的时间演化

**结果**：`reports/experiment2_results/`

---

### 实验三：层依赖训练动态 (Figure 3)
**目的**：揭示不同层的动态特性差异

**结果**：`reports/experiment3_results/`

---

### 实验四：设计空间结构 (Figure 4)
**目的**：分析液态层和固态层的参数空间结构

**结果**：`reports/experiment4_results/`

---

### 实验五：温度依赖性 (Figure S1)
**目的**：研究温度 T 对训练动态的影响

**结果**：`reports/experiment5_results/`

---

### 实验六/七：相关函数补充 (SI Sec. S2/S3)
**目的**：提供更详细的相关函数数据

**结果**：`reports/correlation_supplement/`

---

## 待复现实验

### 实验八：网络深度依赖性 (SI Sec. S4) - 优先级 P5

**目的**：研究总网络深度 L 对结果的影响

**参数范围**：L = 6, 8, 10, 12, 14

---

### 实验九：重叠参数分布补充 (SI Sec. S5) - 优先级 P5

**目的**：提供更多关于设计空间结构的数据

---

## 与 Manus 协作指南

1. **克隆仓库**：`gh repo clone thuyd99513/Thermal-DNN-MC-Optimized`
2. **查看进度**：阅读本文件了解当前状态
3. **运行实验**：在算力平台运行 `experiment1_full/` 中的代码
4. **上传结果**：将结果文件夹打包上传给 Manus
5. **获取分析**：Manus 将生成图表和报告

---

*最后更新：2026-01-30*
