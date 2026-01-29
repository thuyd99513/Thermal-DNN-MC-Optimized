# 论文数据使用要求分析

## 论文原文说明 (Appendix A, Page 7)

> **Appendix A: Input dataset.** The input dataset contains the MNIST handwritten digit images [27, 28], which are widely used in machine learning benchmarking. The original images are coarse-grained into 28 × 28-pixel pictures, and each pixel is either white (+1) or black (−1). We only select images of two digits, "0" and "1", from the database, for a simple task of binary classification. The input dataset is divided into training (2000 images) and test (400 images) parts.

## 关键要求

1. **数据来源**: MNIST 手写数字数据集
2. **数据处理**: 
   - 28×28 像素图像
   - 每个像素二值化为 +1 (白) 或 -1 (黑)
3. **类别选择**: 只使用数字 "0" 和 "1" (二分类任务)
4. **数据划分**:
   - 训练集: 2000 张图像 (M = 2000)
   - 测试集: 400 张图像

## 当前代码实现

当前代码使用**随机生成的数据**:
```python
S_in_shared = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float64)
S_out_shared = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float64)
```

这与论文要求**不一致**。

## 需要修改的内容

1. 加载真实 MNIST 数据集
2. 筛选数字 "0" 和 "1"
3. 将像素值二值化为 ±1
4. 使用 2000 张训练图像作为边界条件 S_0
5. 标签 (0→-1, 1→+1) 作为输出边界条件 S_L

## 物理意义

在 TDLM 模型中:
- **S_0 (输入边界)**: 固定为训练图像的像素值
- **S_L (输出边界)**: 固定为训练标签
- **S_l (隐藏层)**: 自由变量，通过 MC/MD 模拟演化
- **J_l (权重)**: 自由变量，通过 MC/MD 模拟演化

边界条件是"淬火无序"(quenched disorder)，决定了系统的能量景观。
