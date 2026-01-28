#!/usr/bin/env python3
"""
experiment2_global_dynamics.py - 复现论文实验二：全局训练动态 (Figure 2)

本脚本复现论文 "Liquid and solid layers in a thermal deep learning machine" 的 Figure 2 结果。

测量内容：
1. 能量 E(t)/E(0) 随时间的衰减曲线
2. 训练集和测试集准确率 A_train(t), A_test(t) 的时间演化
3. 自相关函数 c(t, t_w)
4. 副本间重叠 q(t, t_w)

关键改进：
- 加载真实 MNIST 数据（数字 0 和 1）
- 实现标准 DNN 前向传播计算测试集准确率
- 按论文定义实现能量计算

作者：Manus AI
日期：2026-01-28
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from datetime import datetime
from time import time
from numba import njit, prange

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Network_optimized_v3 import NetworkOptimizedV3

# ============================================================================
# MNIST 数据加载
# ============================================================================

def load_mnist_binary(train_size=2000, test_size=400, seed=42):
    """
    加载 MNIST 数据集中的数字 0 和 1 进行二分类
    
    按照论文 Appendix A 的描述：
    - 28×28 像素图像
    - 每个像素为白色(+1)或黑色(-1)
    - 只选择数字 0 和 1
    - 训练集 2000 张，测试集 400 张
    
    Args:
        train_size: 训练集大小
        test_size: 测试集大小
        seed: 随机种子
    
    Returns:
        X_train: 训练集输入 (train_size, 784)，值为 ±1
        y_train: 训练集标签 (train_size, 2)，one-hot 编码为 ±1
        X_test: 测试集输入 (test_size, 784)，值为 ±1
        y_test: 测试集标签 (test_size, 2)，one-hot 编码为 ±1
    """
    import urllib.request
    import gzip
    import struct
    
    # 直接下载 MNIST 数据
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = '/tmp/mnist_data'
    os.makedirs(data_dir, exist_ok=True)
    
    def download_and_parse_mnist(filename, is_images=True):
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  下载 {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        
        with gzip.open(filepath, 'rb') as f:
            if is_images:
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            else:
                magic, num = struct.unpack('>II', f.read(8))
                data = np.frombuffer(f.read(), dtype=np.uint8)
        return data
    
    try:
        print("正在下载 MNIST 数据集...")
        X_train_raw = download_and_parse_mnist(files['train_images'], True)
        y_train_raw = download_and_parse_mnist(files['train_labels'], False)
        X_test_raw = download_and_parse_mnist(files['test_images'], True)
        y_test_raw = download_and_parse_mnist(files['test_labels'], False)
        
        X_all = np.vstack([X_train_raw, X_test_raw])
        y_all = np.concatenate([y_train_raw, y_test_raw])
        print(f"MNIST 数据加载成功: {X_all.shape}")
    except Exception as e:
        print(f"无法下载 MNIST 数据: {e}")
        print("使用合成数据...")
        return generate_synthetic_binary_data(train_size, test_size, seed)
    
    # 筛选数字 0 和 1
    mask = (y_all == 0) | (y_all == 1)
    X_binary = X_all[mask]
    y_binary = y_all[mask]
    
    print(f"筛选后的数据量: {len(X_binary)} (数字 0 和 1)")
    
    # 随机打乱
    np.random.seed(seed)
    indices = np.random.permutation(len(X_binary))
    X_binary = X_binary[indices]
    y_binary = y_binary[indices]
    
    # 划分训练集和测试集
    X_train = X_binary[:train_size]
    y_train_raw = y_binary[:train_size]
    X_test = X_binary[train_size:train_size + test_size]
    y_test_raw = y_binary[train_size:train_size + test_size]
    
    # 将像素值转换为 ±1（论文中的二值化处理）
    # 使用阈值 127.5 进行二值化
    X_train = np.where(X_train > 127.5, 1.0, -1.0)
    X_test = np.where(X_test > 127.5, 1.0, -1.0)
    
    # 将标签转换为 one-hot 编码，值为 ±1
    # 数字 0 -> [+1, -1]
    # 数字 1 -> [-1, +1]
    y_train = np.zeros((train_size, 2))
    y_train[y_train_raw == 0, 0] = 1
    y_train[y_train_raw == 0, 1] = -1
    y_train[y_train_raw == 1, 0] = -1
    y_train[y_train_raw == 1, 1] = 1
    
    y_test = np.zeros((test_size, 2))
    y_test[y_test_raw == 0, 0] = 1
    y_test[y_test_raw == 0, 1] = -1
    y_test[y_test_raw == 1, 0] = -1
    y_test[y_test_raw == 1, 1] = 1
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"训练集标签分布: 0={np.sum(y_train_raw==0)}, 1={np.sum(y_train_raw==1)}")
    
    return X_train, y_train, X_test, y_test


def generate_synthetic_binary_data(train_size=2000, test_size=400, seed=42):
    """
    生成合成的二分类数据（当无法加载 MNIST 时使用）
    """
    np.random.seed(seed)
    N_in = 784
    N_out = 2
    
    # 生成随机二值输入
    X_train = np.random.choice([-1, 1], size=(train_size, N_in)).astype(np.float64)
    X_test = np.random.choice([-1, 1], size=(test_size, N_in)).astype(np.float64)
    
    # 生成随机标签
    y_train = np.random.choice([-1, 1], size=(train_size, N_out)).astype(np.float64)
    y_test = np.random.choice([-1, 1], size=(test_size, N_out)).astype(np.float64)
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# 网络类扩展：支持真实数据
# ============================================================================

class TDLMWithData(NetworkOptimizedV3):
    """
    扩展的热力学深度学习机，支持真实 MNIST 数据
    """
    
    def __init__(self, X_train, y_train, N, L, beta=1e5, seed=42):
        """
        初始化网络
        
        Args:
            X_train: 训练集输入 (M, N_in)，值为 ±1
            y_train: 训练集标签 (M, N_out)，值为 ±1
            N: 每层隐藏神经元数
            L: 网络总层数
            beta: 逆温度
            seed: 随机种子
        """
        M = X_train.shape[0]
        N_in = X_train.shape[1]
        N_out = y_train.shape[1]
        
        # 调用父类初始化
        super().__init__(M, N, L, N_in=N_in, N_out=N_out, beta=beta, seed=seed)
        
        # 设置固定的边界条件（输入和输出）
        self.S_in = X_train.copy()  # 输入层固定为训练数据
        self.S_out = y_train.copy()  # 输出层固定为标签
        
        # 保存原始数据引用
        self.X_train = X_train
        self.y_train = y_train
        
        print(f"网络初始化: M={M}, N={N}, L={L}, N_in={N_in}, N_out={N_out}")
        print(f"逆温度 β = {beta:.2e}, 温度 T = {1/beta:.2e}")


# ============================================================================
# 能量计算（按论文定义）
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_layer_energy_numba(gap):
    """
    计算单层能量 (Numba 加速)
    
    按论文 Eq. (2) 定义：
    E_l = sqrt(sum_{μ,i} [S_{l,i}^μ - tanh(h_{l,i}^μ)]^2)
    
    其中 gap = h * S，当 gap < 0 时表示违反约束
    
    使用软核势能：V(gap) = gap^2 if gap < 0 else 0
    """
    total = 0.0
    for i in range(gap.size):
        val = gap.flat[i]
        if val < 0:
            total += val * val
    return total


def compute_total_energy(net):
    """
    计算总能量 E = Σ_l E_l
    
    按论文 Eq. (3) 定义
    """
    total_energy = 0.0
    
    # 第一层 (输入层到第一隐藏层)
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN  # (M, N)
    gap_0 = h_0 * net.S[:, 0, :]  # (M, N)
    total_energy += compute_layer_energy_numba(gap_0)
    
    # 中间层
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N  # (M, N)
        gap_l = h_l * net.S[:, l+1, :]  # (M, N)
        total_energy += compute_layer_energy_numba(gap_l)
    
    # 输出层
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N  # (M, N_out)
    gap_out = h_out * net.S_out  # (M, N_out)
    total_energy += compute_layer_energy_numba(gap_out)
    
    return total_energy


# ============================================================================
# 准确率计算
# ============================================================================

def compute_training_accuracy(net):
    """
    计算训练集准确率
    
    对于每个样本，如果所有层的约束都满足（所有 gap > 0），则认为正确分类
    """
    correct = 0
    
    for mu in range(net.M):
        all_satisfied = True
        
        # 检查第一层
        h_0 = net.J_in @ net.S_in[mu] / net.SQRT_N_IN
        gap_0 = h_0 * net.S[mu, 0, :]
        if np.any(gap_0 < 0):
            all_satisfied = False
        
        # 检查中间层
        if all_satisfied:
            for l in range(net.num_hidden_bond_layers):
                h_l = net.J_hidden[l] @ net.S[mu, l, :] / net.SQRT_N
                gap_l = h_l * net.S[mu, l+1, :]
                if np.any(gap_l < 0):
                    all_satisfied = False
                    break
        
        # 检查输出层
        if all_satisfied:
            h_out = net.J_out @ net.S[mu, -1, :] / net.SQRT_N
            gap_out = h_out * net.S_out[mu]
            if np.any(gap_out < 0):
                all_satisfied = False
        
        if all_satisfied:
            correct += 1
    
    return correct / net.M


def dnn_forward_pass(X, J_in, J_hidden, J_out, SQRT_N_IN, SQRT_N):
    """
    标准 DNN 前向传播
    
    按论文 Appendix D (Eq. 12) 定义：
    S_{l,i} = sgn(1/sqrt(N_{l-1}) * sum_j J_{l,i,j} S_{l-1,j})
    
    Args:
        X: 输入数据 (batch_size, N_in)，值为 ±1
        J_in: 输入层权重 (N, N_in)
        J_hidden: 隐藏层权重 (L-2, N, N)
        J_out: 输出层权重 (N_out, N)
        SQRT_N_IN: sqrt(N_in)
        SQRT_N: sqrt(N)
    
    Returns:
        output: 网络输出 (batch_size, N_out)，值为 ±1
    """
    batch_size = X.shape[0]
    L_hidden = J_hidden.shape[0] + 1  # 隐藏层数
    
    # 第一隐藏层
    h = X @ J_in.T / SQRT_N_IN  # (batch_size, N)
    S = np.sign(h)
    S[S == 0] = 1  # 处理零值
    
    # 中间隐藏层
    for l in range(J_hidden.shape[0]):
        h = S @ J_hidden[l].T / SQRT_N  # (batch_size, N)
        S = np.sign(h)
        S[S == 0] = 1
    
    # 输出层
    h = S @ J_out.T / SQRT_N  # (batch_size, N_out)
    output = np.sign(h)
    output[output == 0] = 1
    
    return output


def compute_test_accuracy(net, X_test, y_test):
    """
    计算测试集准确率
    
    使用标准 DNN 前向传播，将 TDLM 学到的权重应用于测试数据
    
    Args:
        net: 训练好的网络
        X_test: 测试集输入 (test_size, N_in)
        y_test: 测试集标签 (test_size, N_out)
    
    Returns:
        accuracy: 测试集准确率
    """
    # 前向传播
    output = dnn_forward_pass(
        X_test, net.J_in, net.J_hidden, net.J_out,
        net.SQRT_N_IN, net.SQRT_N
    )
    
    # 计算准确率
    # 对于二分类，比较输出和标签的符号
    correct = np.sum(np.all(output == y_test, axis=1))
    accuracy = correct / len(y_test)
    
    return accuracy


# ============================================================================
# 自相关函数和副本重叠
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_spin_autocorrelation_numba(S_t, S_tw):
    """
    计算自旋自相关函数 (Numba 加速)
    
    按论文 Eq. (15) 定义：
    c(t, t_w) = 1/(L-1) Σ_l [1/(N_l*M) Σ_{i,μ} S_{l,i}^μ(t+t_w) S_{l,i}^μ(t_w)]
    
    简化版本：对所有隐藏层自旋求平均
    c(t, t_w) = <S(t) · S(t_w)> / N_dof
    """
    N_dof = S_t.size
    total = 0.0
    for i in range(N_dof):
        total += S_t.flat[i] * S_tw.flat[i]
    return total / N_dof


@njit(cache=True, fastmath=True)
def compute_replica_overlap_numba(S_a, S_b):
    """
    计算副本间重叠 (Numba 加速)
    
    按论文 Eq. (19) 定义：
    q^{ab}(t, t_w) = 1/(L-1) Σ_l [1/(N_l*M) Σ_{i,μ} S_{l,i}^{μ,a}(t+t_w) S_{l,i}^{μ,b}(t+t_w)]
    
    简化版本：
    q(t) = <S^a(t) · S^b(t)> / N_dof
    """
    N_dof = S_a.size
    total = 0.0
    for i in range(N_dof):
        total += S_a.flat[i] * S_b.flat[i]
    return total / N_dof


def compute_layer_autocorrelation(S_t, S_tw, layer):
    """
    计算特定层的自相关函数
    
    c_l(t, t_w) = 1/(N_l*M) Σ_{i,μ} S_{l,i}^μ(t+t_w) S_{l,i}^μ(t_w)
    """
    S_t_layer = S_t[:, layer, :]  # (M, N)
    S_tw_layer = S_tw[:, layer, :]  # (M, N)
    return np.mean(S_t_layer * S_tw_layer)


def compute_layer_overlap(S_a, S_b, layer):
    """
    计算特定层的副本间重叠
    
    q_l(t) = 1/(N_l*M) Σ_{i,μ} S_{l,i}^{μ,a}(t) S_{l,i}^{μ,b}(t)
    """
    S_a_layer = S_a[:, layer, :]  # (M, N)
    S_b_layer = S_b[:, layer, :]  # (M, N)
    return np.mean(S_a_layer * S_b_layer)


# ============================================================================
# 实验主函数
# ============================================================================

def run_experiment2(X_train, y_train, X_test, y_test, N, L, beta, 
                    mc_steps, t_w_list, log_interval=100, seed=42):
    """
    运行实验二：全局训练动态
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        N: 每层神经元数
        L: 层数
        beta: 逆温度
        mc_steps: MC 步数
        t_w_list: 等待时间列表
        log_interval: 记录间隔
        seed: 随机种子
    
    Returns:
        results: 包含所有测量结果的字典
    """
    M = X_train.shape[0]
    alpha = M / N
    ln_alpha = np.log(alpha)
    
    print(f"\n{'='*60}")
    print(f"实验二：全局训练动态")
    print(f"M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    print(f"β={beta:.0e}, T={1/beta:.0e}")
    print(f"MC steps={mc_steps}, t_w={t_w_list}")
    print(f"{'='*60}")
    
    # 初始化两个副本（用于计算副本重叠）
    np.random.seed(seed)
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    
    # 副本共享权重初始化，但自旋配置独立演化
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    # 计算初始能量
    E0 = compute_total_energy(net_a)
    print(f"初始能量 E(0) = {E0:.4f}")
    
    # 初始准确率
    A_train_0 = compute_training_accuracy(net_a)
    A_test_0 = compute_test_accuracy(net_a, X_test, y_test)
    print(f"初始准确率: 训练集={A_train_0:.4f}, 测试集={A_test_0:.4f}")
    
    # 存储结果
    time_points = []
    energy_ratio = []
    accuracy_train = []
    accuracy_test = []
    
    # 存储用于自相关的配置
    S_tw_configs = {tw: None for tw in t_w_list}
    autocorr_data = {tw: {'t': [], 'c': []} for tw in t_w_list}
    overlap_data = {tw: {'t': [], 'q': []} for tw in t_w_list}
    
    # 层相关数据
    layer_autocorr_data = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    layer_overlap_data = {l: {'t': [], 'q': []} for l in range(L-1)}
    
    start_time = time()
    
    # 记录初始状态
    time_points.append(0)
    energy_ratio.append(1.0)
    accuracy_train.append(A_train_0)
    accuracy_test.append(A_test_0)
    
    for step in range(1, mc_steps + 1):
        # MC 步：更新两个副本
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        # 同步权重（副本共享权重演化）
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        # 记录数据
        if step % log_interval == 0 or step == 1:
            t = step
            time_points.append(t)
            
            # 计算能量
            E_t = compute_total_energy(net_a)
            energy_ratio.append(E_t / E0 if E0 > 0 else 1.0)
            
            # 计算准确率
            A_train_t = compute_training_accuracy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            accuracy_train.append(A_train_t)
            accuracy_test.append(A_test_t)
            
            # 保存 t_w 时刻的配置
            for tw in t_w_list:
                if step == tw:
                    S_tw_configs[tw] = net_a.S.copy()
                    print(f"  [t={tw}] 保存配置用于自相关计算")
            
            # 计算自相关和副本重叠
            for tw in t_w_list:
                if S_tw_configs[tw] is not None and step >= tw:
                    # 全局自相关
                    c_t = compute_spin_autocorrelation_numba(net_a.S, S_tw_configs[tw])
                    autocorr_data[tw]['t'].append(t)
                    autocorr_data[tw]['c'].append(c_t)
                    
                    # 全局副本重叠
                    q_t = compute_replica_overlap_numba(net_a.S, net_b.S)
                    overlap_data[tw]['t'].append(t)
                    overlap_data[tw]['q'].append(q_t)
                    
                    # 层自相关
                    for l in range(L-1):
                        c_l = compute_layer_autocorrelation(net_a.S, S_tw_configs[tw], l)
                        layer_autocorr_data[tw][l]['t'].append(t)
                        layer_autocorr_data[tw][l]['c'].append(c_l)
            
            # 层重叠
            for l in range(L-1):
                q_l = compute_layer_overlap(net_a.S, net_b.S, l)
                layer_overlap_data[l]['t'].append(t)
                layer_overlap_data[l]['q'].append(q_l)
        
        # 进度输出
        if step % max(1, mc_steps // 10) == 0:
            elapsed = time() - start_time
            eta = elapsed / step * (mc_steps - step)
            E_t = compute_total_energy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            print(f"  Step {step:>6}/{mc_steps} ({100*step/mc_steps:>5.1f}%), "
                  f"E/E0={E_t/E0:.4e}, A_test={A_test_t:.4f}, "
                  f"ETA: {eta/60:.1f}min", flush=True)
    
    total_time = time() - start_time
    print(f"\n模拟完成，总耗时: {total_time/60:.1f} 分钟")
    
    # 最终结果
    E_final = compute_total_energy(net_a)
    A_train_final = compute_training_accuracy(net_a)
    A_test_final = compute_test_accuracy(net_a, X_test, y_test)
    print(f"最终能量: E*/E(0) = {E_final/E0:.4e}")
    print(f"最终准确率: 训练集={A_train_final:.4f}, 测试集={A_test_final:.4f}")
    
    # 整理结果
    results = {
        'M': M, 'N': N, 'L': L, 'beta': beta,
        'alpha': alpha, 'ln_alpha': ln_alpha,
        'E0': E0,
        'time_points': np.array(time_points),
        'energy_ratio': np.array(energy_ratio),
        'accuracy_train': np.array(accuracy_train),
        'accuracy_test': np.array(accuracy_test),
        'autocorr_data': autocorr_data,
        'overlap_data': overlap_data,
        'layer_autocorr_data': layer_autocorr_data,
        'layer_overlap_data': layer_overlap_data,
        't_w_list': t_w_list,
        'total_time': total_time
    }
    
    return results


# ============================================================================
# 绘图函数
# ============================================================================

def plot_experiment2_results(results_list, output_dir):
    """
    绘制实验二结果图表（类似论文 Figure 2）
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # (a) E(t)/E(0) vs t (固定 M)
    ax = axes[0, 0]
    for i, results in enumerate(results_list):
        ln_alpha = results['ln_alpha']
        t = results['time_points']
        E_ratio = results['energy_ratio']
        mask = t > 0
        ax.loglog(t[mask], E_ratio[mask], 'o-', markersize=2, 
                  color=colors[i], label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('E(t)/E(0)', fontsize=12)
    ax.set_title('(a) Energy decay', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (b) A_train(t) vs t
    ax = axes[0, 1]
    for i, results in enumerate(results_list):
        ln_alpha = results['ln_alpha']
        ax.semilogx(results['time_points'], results['accuracy_train'],
                    'o-', markersize=2, color=colors[i], label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('A_train(t)', fontsize=12)
    ax.set_title('(b) Training accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (c) A_test(t) vs t
    ax = axes[0, 2]
    for i, results in enumerate(results_list):
        ln_alpha = results['ln_alpha']
        ax.semilogx(results['time_points'], results['accuracy_test'],
                    'o-', markersize=2, color=colors[i], label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('A_test(t)', fontsize=12)
    ax.set_title('(c) Test accuracy', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (d) E* and A* vs ln α
    ax = axes[0, 3]
    ln_alphas = [r['ln_alpha'] for r in results_list]
    E_stars = [r['energy_ratio'][-1] for r in results_list]
    A_test_stars = [r['accuracy_test'][-1] for r in results_list]
    
    ax2 = ax.twinx()
    line1, = ax.semilogy(ln_alphas, E_stars, 'bo-', markersize=8, linewidth=2, label='E*/E(0)')
    line2, = ax2.plot(ln_alphas, A_test_stars, 'rs-', markersize=8, linewidth=2, label='A*_test')
    ax.set_xlabel('ln α', fontsize=12)
    ax.set_ylabel('E*/E(0)', color='b', fontsize=12)
    ax2.set_ylabel('A*_test', color='r', fontsize=12)
    ax.set_title('(d) Final values vs ln α', fontsize=14)
    ax.legend(handles=[line1, line2], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (e) c(t, t_w) vs t (全局自相关)
    ax = axes[1, 0]
    if results_list:
        results = results_list[0]  # 使用第一个配置
        tw_colors = plt.cm.plasma(np.linspace(0, 1, len(results['t_w_list'])))
        for i, tw in enumerate(results['t_w_list']):
            data = results['autocorr_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['c'], 'o-', markersize=3,
                            color=tw_colors[i], label=f"t_w = {tw}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('c(t, t_w)', fontsize=12)
    ax.set_title('(e) Spin autocorrelation', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (f) q(t, t_w) vs t (全局副本重叠)
    ax = axes[1, 1]
    if results_list:
        results = results_list[0]
        for i, tw in enumerate(results['t_w_list']):
            data = results['overlap_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['q'], 'o-', markersize=3,
                            color=tw_colors[i], label=f"t_w = {tw}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('q(t, t_w)', fontsize=12)
    ax.set_title('(f) Replica overlap', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (g) 层自相关 c_l(t, t_w)
    ax = axes[1, 2]
    if results_list:
        results = results_list[0]
        tw = results['t_w_list'][0] if results['t_w_list'] else None
        if tw is not None:
            L = results['L']
            layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
            for l in range(L-1):
                data = results['layer_autocorr_data'][tw][l]
                if len(data['t']) > 0:
                    ax.semilogx(data['t'], data['c'], 'o-', markersize=2,
                                color=layer_colors[l], label=f"l = {l+1}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel(f'c_l(t, t_w={tw})', fontsize=12)
    ax.set_title('(g) Layer autocorrelation', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (h) 层重叠 q_l(t)
    ax = axes[1, 3]
    if results_list:
        results = results_list[0]
        L = results['L']
        layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
        for l in range(L-1):
            data = results['layer_overlap_data'][l]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['q'], 'o-', markersize=2,
                            color=layer_colors[l], label=f"l = {l+1}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('q_l(t)', fontsize=12)
    ax.set_title('(h) Layer overlap', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'experiment2_figure2_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表保存到: {filepath}")
    return filepath


def plot_summary_figure(results_list, output_dir):
    """
    绘制简化的摘要图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # 能量衰减
    ax = axes[0]
    for i, results in enumerate(results_list):
        ln_alpha = results['ln_alpha']
        t = results['time_points']
        E_ratio = results['energy_ratio']
        mask = t > 0
        ax.loglog(t[mask], E_ratio[mask], 'o-', markersize=3, 
                  color=colors[i], label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('E(t)/E(0)', fontsize=12)
    ax.set_title('Energy Decay', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 测试准确率
    ax = axes[1]
    for i, results in enumerate(results_list):
        ln_alpha = results['ln_alpha']
        ax.semilogx(results['time_points'], results['accuracy_test'],
                    'o-', markersize=3, color=colors[i], label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=12)
    ax.set_ylabel('A_test(t)', fontsize=12)
    ax.set_title('Test Accuracy', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 最终值 vs ln α
    ax = axes[2]
    ln_alphas = [r['ln_alpha'] for r in results_list]
    A_test_stars = [r['accuracy_test'][-1] for r in results_list]
    A_train_stars = [r['accuracy_train'][-1] for r in results_list]
    
    ax.plot(ln_alphas, A_train_stars, 'bo-', markersize=10, linewidth=2, label='A*_train')
    ax.plot(ln_alphas, A_test_stars, 'rs-', markersize=10, linewidth=2, label='A*_test')
    ax.set_xlabel('ln α', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Final Accuracy vs ln α', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'experiment2_summary_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"摘要图保存到: {filepath}")
    return filepath


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("实验二：全局训练动态 (Figure 2)")
    print("复现论文 'Liquid and solid layers in a thermal deep learning machine'")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'reports', 'experiment2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载 MNIST 数据
    print("\n" + "=" * 50)
    print("加载 MNIST 数据集 (数字 0 和 1)")
    print("=" * 50)
    X_train, y_train, X_test, y_test = load_mnist_binary(
        train_size=2000, test_size=400, seed=42
    )
    
    # 实验参数
    L = 10  # 网络层数
    beta = 1e5  # 逆温度 (T = 10^-5)
    mc_steps = 10000  # MC 步数（可以增加以获得更好的结果）
    t_w_list = [100, 1000, 5000]  # 等待时间列表
    log_interval = 50  # 记录间隔
    
    # 不同的 ln α 配置
    # α = M/N, 固定 M=2000
    configs = [
        {'N': 200},   # ln α ≈ 2.3
        {'N': 100},   # ln α ≈ 3.0
        {'N': 50},    # ln α ≈ 3.7
        {'N': 20},    # ln α ≈ 4.6
    ]
    
    print(f"\n实验配置:")
    print(f"  L = {L}, β = {beta:.0e}, T = {1/beta:.0e}")
    print(f"  MC steps = {mc_steps}")
    print(f"  t_w = {t_w_list}")
    print(f"  配置数: {len(configs)}")
    for i, config in enumerate(configs):
        N = config['N']
        M = X_train.shape[0]
        ln_alpha = np.log(M / N)
        print(f"    配置 {i+1}: N={N}, α={M/N:.1f}, ln α={ln_alpha:.2f}")
    
    # 预热 JIT
    print("\n预热 JIT 编译...")
    warmup_net = TDLMWithData(X_train[:10], y_train[:10], N=5, L=5, beta=beta, seed=0)
    for _ in range(5):
        warmup_net.mc_step_vectorized()
    print("JIT 预热完成")
    
    # 运行实验
    results_list = []
    for i, config in enumerate(configs):
        N = config['N']
        M = X_train.shape[0]
        ln_alpha = np.log(M / N)
        
        print(f"\n{'='*60}")
        print(f"配置 {i+1}/{len(configs)}: N={N}, α={M/N:.1f}, ln α={ln_alpha:.2f}")
        print(f"{'='*60}")
        
        results = run_experiment2(
            X_train, y_train, X_test, y_test,
            N, L, beta, mc_steps, t_w_list,
            log_interval=log_interval, seed=42+i
        )
        results_list.append(results)
    
    # 绘制结果
    print("\n" + "=" * 50)
    print("生成图表...")
    print("=" * 50)
    
    filepath1 = plot_experiment2_results(results_list, output_dir)
    filepath2 = plot_summary_figure(results_list, output_dir)
    
    # 保存数据
    data_filepath = os.path.join(output_dir, 
                                 f'experiment2_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(data_filepath, results_list=results_list)
    print(f"数据保存到: {data_filepath}")
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("实验二结果摘要")
    print("=" * 70)
    
    print(f"\n最终值 (t* = {mc_steps})")
    print("-" * 60)
    print(f"{'ln α':>8} | {'α':>8} | {'E*/E(0)':>12} | {'A*_train':>10} | {'A*_test':>10}")
    print("-" * 60)
    
    for results in results_list:
        ln_alpha = results['ln_alpha']
        alpha = results['alpha']
        E_star = results['energy_ratio'][-1]
        A_train_star = results['accuracy_train'][-1]
        A_test_star = results['accuracy_test'][-1]
        print(f"{ln_alpha:>8.2f} | {alpha:>8.1f} | {E_star:>12.4e} | {A_train_star:>10.4f} | {A_test_star:>10.4f}")
    
    print("-" * 60)
    
    return results_list, filepath1, filepath2


if __name__ == "__main__":
    main()
