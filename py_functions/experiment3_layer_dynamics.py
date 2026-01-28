#!/usr/bin/env python3
"""
experiment3_layer_dynamics.py - 实验三：层依赖训练动态 (Figure 3)

测量内容：
1. 层自相关函数 c_l(t, t_w) - 不同层的自旋自相关
2. 弛豫时间 τ_l^c - c_l 衰减到 1/e 所需时间
3. 层副本相关函数 q_l(t, t_w) - 不同层的副本重叠
4. 最终层重叠 q_l* 和穿透深度 ξ_l
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from time import time
from numba import njit
import urllib.request
import gzip
import struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Network_optimized_v3 import NetworkOptimizedV3

# ============================================================================
# MNIST 数据加载
# ============================================================================

def load_mnist_binary(train_size=500, test_size=100, seed=42):
    """加载 MNIST 数据集中的数字 0 和 1"""
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    data_dir = '/tmp/mnist_data'
    os.makedirs(data_dir, exist_ok=True)
    
    def download_and_parse(filename, is_images=True):
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
        print("加载 MNIST 数据...")
        X_train_raw = download_and_parse(files['train_images'], True)
        y_train_raw = download_and_parse(files['train_labels'], False)
        X_test_raw = download_and_parse(files['test_images'], True)
        y_test_raw = download_and_parse(files['test_labels'], False)
        
        X_all = np.vstack([X_train_raw, X_test_raw])
        y_all = np.concatenate([y_train_raw, y_test_raw])
        
        mask = (y_all == 0) | (y_all == 1)
        X_binary = X_all[mask]
        y_binary = y_all[mask]
        
        np.random.seed(seed)
        indices = np.random.permutation(len(X_binary))
        X_binary = X_binary[indices]
        y_binary = y_binary[indices]
        
        X_train = X_binary[:train_size]
        y_train_labels = y_binary[:train_size]
        X_test = X_binary[train_size:train_size + test_size]
        y_test_labels = y_binary[train_size:train_size + test_size]
        
        # 二值化
        X_train = np.where(X_train > 127.5, 1.0, -1.0)
        X_test = np.where(X_test > 127.5, 1.0, -1.0)
        
        # One-hot 编码
        y_train = np.zeros((train_size, 2))
        y_train[y_train_labels == 0, 0] = 1
        y_train[y_train_labels == 0, 1] = -1
        y_train[y_train_labels == 1, 0] = -1
        y_train[y_train_labels == 1, 1] = 1
        
        y_test = np.zeros((test_size, 2))
        y_test[y_test_labels == 0, 0] = 1
        y_test[y_test_labels == 0, 1] = -1
        y_test[y_test_labels == 1, 0] = -1
        y_test[y_test_labels == 1, 1] = 1
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"MNIST 加载失败: {e}, 使用合成数据")
        return generate_synthetic_data(train_size, test_size, seed)


def generate_synthetic_data(train_size, test_size, seed):
    np.random.seed(seed)
    N_in, N_out = 784, 2
    
    def gen_class(n, cid):
        X = np.random.choice([-1, 1], size=(n, N_in)).astype(np.float64)
        if cid == 0:
            X[:, :N_in//2] = np.where(np.random.rand(n, N_in//2) > 0.3, 1, -1)
        else:
            X[:, N_in//2:] = np.where(np.random.rand(n, N_in//2) > 0.3, 1, -1)
        return X
    
    n_train = train_size // 2
    X_train = np.vstack([gen_class(n_train, 0), gen_class(n_train, 1)])
    y_train = np.zeros((train_size, N_out))
    y_train[:n_train, 0], y_train[:n_train, 1] = 1, -1
    y_train[n_train:, 0], y_train[n_train:, 1] = -1, 1
    
    n_test = test_size // 2
    X_test = np.vstack([gen_class(n_test, 0), gen_class(n_test, 1)])
    y_test = np.zeros((test_size, N_out))
    y_test[:n_test, 0], y_test[:n_test, 1] = 1, -1
    y_test[n_test:, 0], y_test[n_test:, 1] = -1, 1
    
    idx_train = np.random.permutation(train_size)
    idx_test = np.random.permutation(test_size)
    return X_train[idx_train], y_train[idx_train], X_test[idx_test], y_test[idx_test]


# ============================================================================
# 网络类
# ============================================================================

class TDLMWithData(NetworkOptimizedV3):
    def __init__(self, X_train, y_train, N, L, beta=1e5, seed=42):
        M, N_in, N_out = X_train.shape[0], X_train.shape[1], y_train.shape[1]
        super().__init__(M, N, L, N_in=N_in, N_out=N_out, beta=beta, seed=seed)
        self.S_in = X_train.copy()
        self.S_out = y_train.copy()


# ============================================================================
# 物理量计算
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_energy_numba(gap):
    """计算能量 E = sum of gap^2 for gap < 0"""
    total = 0.0
    for i in range(gap.size):
        if gap.flat[i] < 0:
            total += gap.flat[i] ** 2
    return total


def compute_total_energy(net):
    """计算总能量"""
    E = 0.0
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    E += compute_energy_numba(h_0 * net.S[:, 0, :])
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        E += compute_energy_numba(h_l * net.S[:, l+1, :])
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N
    E += compute_energy_numba(h_out * net.S_out)
    return E


def dnn_forward(X, J_in, J_hidden, J_out, SQRT_N_IN, SQRT_N):
    """标准 DNN 前向传播"""
    S = np.sign(X @ J_in.T / SQRT_N_IN)
    S[S == 0] = 1
    for l in range(J_hidden.shape[0]):
        S = np.sign(S @ J_hidden[l].T / SQRT_N)
        S[S == 0] = 1
    out = np.sign(S @ J_out.T / SQRT_N)
    out[out == 0] = 1
    return out


def compute_test_accuracy(net, X_test, y_test):
    """计算测试集准确率"""
    out = dnn_forward(X_test, net.J_in, net.J_hidden, net.J_out, net.SQRT_N_IN, net.SQRT_N)
    return np.sum(np.all(out == y_test, axis=1)) / len(y_test)


def compute_layer_autocorr(S_t, S_tw, L):
    """计算每层的自相关函数 c_l(t, t_w)"""
    c_l = np.zeros(L - 1)
    for l in range(L - 1):
        c_l[l] = np.mean(S_t[:, l, :] * S_tw[:, l, :])
    return c_l


def compute_layer_overlap(S_a, S_b, L):
    """计算每层的副本重叠 q_l(t, t_w)"""
    q_l = np.zeros(L - 1)
    for l in range(L - 1):
        q_l[l] = np.mean(S_a[:, l, :] * S_b[:, l, :])
    return q_l


def find_relaxation_time(t_array, c_array, threshold=1/np.e):
    """找到弛豫时间 τ：c(τ) = threshold"""
    if len(t_array) == 0 or len(c_array) == 0:
        return np.nan
    
    # 找到第一个 c < threshold 的点
    for i in range(len(c_array)):
        if c_array[i] < threshold:
            if i == 0:
                return t_array[0]
            # 线性插值
            t1, t2 = t_array[i-1], t_array[i]
            c1, c2 = c_array[i-1], c_array[i]
            if c1 == c2:
                return t1
            tau = t1 + (threshold - c1) * (t2 - t1) / (c2 - c1)
            return tau
    
    # 如果没有衰减到阈值以下，返回最大时间
    return np.nan


# ============================================================================
# 实验
# ============================================================================

def run_experiment(X_train, y_train, X_test, y_test, N, L, beta, 
                   mc_steps, t_w_list, log_interval=10, seed=42):
    """运行实验三：层依赖训练动态"""
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    
    np.random.seed(seed)
    
    # 创建两个副本，共享权重但独立演化
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    # 复制权重
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    E0 = compute_total_energy(net_a)
    A_test_0 = compute_test_accuracy(net_a, X_test, y_test)
    print(f"E(0)={E0:.2f}, A_test={A_test_0:.4f}")
    
    # 数据存储
    time_pts = [0]
    E_ratio = [1.0]
    A_test = [A_test_0]
    
    # 保存 t_w 时刻的配置
    S_tw = {tw: None for tw in t_w_list}
    
    # 层自相关函数 c_l(t, t_w)
    layer_autocorr = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    
    # 层副本重叠 q_l(t, t_w)
    layer_overlap = {tw: {l: {'t': [], 'q': []} for l in range(L-1)} for tw in t_w_list}
    
    # 弛豫时间数据
    relaxation_times = {tw: {l: [] for l in range(L-1)} for tw in t_w_list}
    
    t0 = time()
    for step in range(1, mc_steps + 1):
        # MC 更新
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        # 同步权重（两个副本共享权重）
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        # 保存 t_w 配置
        for tw in t_w_list:
            if step == tw:
                S_tw[tw] = net_a.S.copy()
                print(f"  保存 t_w={tw} 配置")
        
        # 记录数据
        if step % log_interval == 0 or step == 1:
            time_pts.append(step)
            E_ratio.append(compute_total_energy(net_a) / E0)
            A_test.append(compute_test_accuracy(net_a, X_test, y_test))
            
            # 计算层自相关和层重叠
            for tw in t_w_list:
                if S_tw[tw] is not None and step >= tw:
                    # 层自相关
                    c_l = compute_layer_autocorr(net_a.S, S_tw[tw], L)
                    for l in range(L-1):
                        layer_autocorr[tw][l]['t'].append(step)
                        layer_autocorr[tw][l]['c'].append(c_l[l])
                    
                    # 层副本重叠
                    q_l = compute_layer_overlap(net_a.S, net_b.S, L)
                    for l in range(L-1):
                        layer_overlap[tw][l]['t'].append(step)
                        layer_overlap[tw][l]['q'].append(q_l[l])
        
        # 进度报告
        if step % max(1, mc_steps // 5) == 0:
            print(f"  Step {step}/{mc_steps}: E/E0={E_ratio[-1]:.4e}, A_test={A_test[-1]:.4f}")
    
    # 计算弛豫时间
    print("\n计算弛豫时间...")
    for tw in t_w_list:
        for l in range(L-1):
            t_arr = np.array(layer_autocorr[tw][l]['t'])
            c_arr = np.array(layer_autocorr[tw][l]['c'])
            if len(t_arr) > 0:
                tau = find_relaxation_time(t_arr, c_arr)
                relaxation_times[tw][l] = tau
                print(f"  t_w={tw}, l={l+1}: τ_l^c = {tau:.1f}" if not np.isnan(tau) else f"  t_w={tw}, l={l+1}: τ_l^c = N/A")
    
    print(f"\n完成，耗时: {time()-t0:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta, 'alpha': alpha, 'ln_alpha': ln_alpha, 'E0': E0,
        'time_points': np.array(time_pts), 'energy_ratio': np.array(E_ratio),
        'accuracy_test': np.array(A_test),
        'layer_autocorr': layer_autocorr,
        'layer_overlap': layer_overlap,
        'relaxation_times': relaxation_times,
        't_w_list': t_w_list
    }


def plot_results(results, output_dir):
    """绘制 Figure 3 风格的结果图"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    L = results['L']
    t_w_list = results['t_w_list']
    
    # 颜色设置
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
    
    # (a) 层自相关函数 c_l(t, t_w) - 选择液态层和固态层
    ax = axes[0, 0]
    liquid_layer = (L - 1) // 2  # 中间层（液态）
    solid_layer = 0  # 第一层（固态）
    
    for i, tw in enumerate(t_w_list):
        # 液态层
        d = results['layer_autocorr'][tw][liquid_layer]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=tw_colors[i], 
                       label=f"l={liquid_layer+1} (liquid), t_w={tw}")
        
        # 固态层
        d = results['layer_autocorr'][tw][solid_layer]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], 's--', ms=3, color=tw_colors[i], alpha=0.7,
                       label=f"l={solid_layer+1} (solid), t_w={tw}")
    
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e threshold')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$c_l(t, t_w)$', fontsize=11)
    ax.set_title('(a) Layer autocorrelation: liquid vs solid', fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # (b) 弛豫时间 τ_l^c vs t_w
    ax = axes[0, 1]
    for l in range(L-1):
        tau_vs_tw = []
        tw_vals = []
        for tw in t_w_list:
            tau = results['relaxation_times'][tw][l]
            if not np.isnan(tau):
                tau_vs_tw.append(tau)
                tw_vals.append(tw)
        
        if len(tau_vs_tw) > 0:
            ax.loglog(tw_vals, tau_vs_tw, 'o-', ms=6, color=layer_colors[l], 
                     label=f"l={l+1}")
    
    ax.axhline(y=t_w_list[-1], color='gray', linestyle=':', alpha=0.5, label='$τ_{th}$')
    ax.set_xlabel('$t_w$', fontsize=11)
    ax.set_ylabel('$τ_l^c$', fontsize=11)
    ax.set_title('(b) Relaxation time vs waiting time', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # (c) 层副本重叠 q_l(t, t_w) - 所有层
    ax = axes[1, 0]
    tw = t_w_list[-1]  # 使用最大的 t_w
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['q'], 'o-', ms=3, color=layer_colors[l], 
                       label=f"l={l+1}")
    
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e threshold')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$q_l(t, t_w)$', fontsize=11)
    ax.set_title(f'(c) Layer replica overlap (t_w={tw})', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # (d) 最终层重叠 q_l* vs 层 l
    ax = axes[1, 1]
    layers = np.arange(1, L)
    
    # 获取最终时刻的 q_l*
    q_l_final = []
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
        if len(d['q']) > 0:
            q_l_final.append(d['q'][-1])
        else:
            q_l_final.append(np.nan)
    
    ax.bar(layers, q_l_final, color=layer_colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2, label='1/e (phase boundary)')
    
    # 标注固态和液态区域
    ax.fill_between([0.5, L-0.5], 1/np.e, 1.0, alpha=0.1, color='blue', label='Solid region')
    ax.fill_between([0.5, L-0.5], 0, 1/np.e, alpha=0.1, color='red', label='Liquid region')
    
    ax.set_xlabel('Layer l', fontsize=11)
    ax.set_ylabel('$q_l^*$', fontsize=11)
    ax.set_title('(d) Final layer overlap profile', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, L - 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment3_figure3_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表保存到: {fp}")
    return fp


def plot_all_layer_autocorr(results, output_dir):
    """绘制所有层的自相关函数详细图"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    L = results['L']
    t_w_list = results['t_w_list']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    
    for l in range(min(L-1, 9)):
        ax = axes[l]
        for i, tw in enumerate(t_w_list):
            d = results['layer_autocorr'][tw][l]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=tw_colors[i], 
                           label=f"t_w={tw}")
        
        ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('$c_l(t, t_w)$', fontsize=10)
        ax.set_title(f'Layer {l+1}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
    
    # 隐藏多余的子图
    for l in range(L-1, 9):
        axes[l].set_visible(False)
    
    plt.suptitle(f'Layer Autocorrelation Functions (ln α = {results["ln_alpha"]:.2f})', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment3_all_layers_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"详细图表保存到: {fp}")
    return fp


def main():
    print("=" * 60)
    print("实验三：层依赖训练动态 (Figure 3)")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'experiment3_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_binary(train_size=300, test_size=60, seed=42)
    
    # 参数设置
    L = 10  # 使用 10 层以更好地观察层依赖效应
    N = 15  # ln α ≈ 3.0
    beta = 1e5
    mc_steps = 2000
    t_w_list = [100, 300, 800]  # 等待时间
    log_interval = 10
    
    print(f"\n参数: L={L}, N={N}, β={beta:.0e}, MC steps={mc_steps}")
    print(f"t_w = {t_w_list}")
    print(f"ln α = {np.log(X_train.shape[0] / N):.2f}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行实验
    results = run_experiment(X_train, y_train, X_test, y_test, N, L, beta, 
                            mc_steps, t_w_list, log_interval, seed=42)
    
    # 绘图
    print("\n生成图表...")
    fp1 = plot_results(results, output_dir)
    fp2 = plot_all_layer_autocorr(results, output_dir)
    
    # 保存数据
    data_fp = os.path.join(output_dir, f'experiment3_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(data_fp, results=results)
    print(f"数据保存到: {data_fp}")
    
    # 结果摘要
    print("\n" + "=" * 60)
    print("实验三结果摘要")
    print("=" * 60)
    
    print(f"\n配置: ln α = {results['ln_alpha']:.2f}, L = {L}")
    
    print("\n最终层重叠 q_l*:")
    print("-" * 40)
    tw = t_w_list[-1]
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
        q_final = d['q'][-1] if len(d['q']) > 0 else np.nan
        phase = "固态" if q_final > 1/np.e else "液态"
        print(f"  层 {l+1}: q_l* = {q_final:.4f} ({phase})")
    
    print("\n弛豫时间 τ_l^c:")
    print("-" * 40)
    for tw in t_w_list:
        print(f"\nt_w = {tw}:")
        for l in range(L-1):
            tau = results['relaxation_times'][tw][l]
            if not np.isnan(tau):
                print(f"  层 {l+1}: τ_l^c = {tau:.1f}")
            else:
                print(f"  层 {l+1}: τ_l^c = N/A (未衰减到 1/e)")
    
    return results


if __name__ == "__main__":
    main()
