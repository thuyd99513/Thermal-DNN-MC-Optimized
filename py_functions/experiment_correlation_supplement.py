#!/usr/bin/env python3
"""
experiment_correlation_supplement.py - 相关函数补充实验

包含两个补充实验：
1. 全局相关函数补充 (SI Sec. S2) - 更多 t_w 值的 c(t,t_w), C(t,t_w), q(t,t_w), Q(t,t_w)
2. 层相关函数补充 (SI Sec. S3) - 所有层的 c_l(t,t_w), q_l(t,t_w) 和弛豫时间分析

测量内容：
- 全局自旋自相关函数 c(t, t_w)
- 全局权重自相关函数 C(t, t_w)
- 全局副本自旋重叠 q(t, t_w)
- 全局副本权重重叠 Q(t, t_w)
- 层自旋自相关函数 c_l(t, t_w)
- 层副本重叠 q_l(t, t_w)
- 弛豫时间 τ_l^c
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
        
        X_train = np.where(X_train > 127.5, 1.0, -1.0)
        X_test = np.where(X_test > 127.5, 1.0, -1.0)
        
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
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"MNIST 加载失败: {e}")
        raise


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
    total = 0.0
    for i in range(gap.size):
        if gap.flat[i] < 0:
            total += gap.flat[i] ** 2
    return total


def compute_total_energy(net):
    E = 0.0
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    E += compute_energy_numba(h_0 * net.S[:, 0, :])
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        E += compute_energy_numba(h_l * net.S[:, l+1, :])
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N
    E += compute_energy_numba(h_out * net.S_out)
    return E


def compute_global_spin_autocorr(S_t, S_tw):
    """全局自旋自相关函数 c(t, t_w) - Eq. 15"""
    L_minus_1 = S_t.shape[1]
    c_layers = np.zeros(L_minus_1)
    for l in range(L_minus_1):
        c_layers[l] = np.mean(S_t[:, l, :] * S_tw[:, l, :])
    return np.mean(c_layers)


def compute_global_weight_autocorr(J_in_t, J_hidden_t, J_out_t, 
                                    J_in_tw, J_hidden_tw, J_out_tw):
    """全局权重自相关函数 C(t, t_w) - Eq. 16"""
    C_layers = []
    # 输入层权重
    C_layers.append(np.mean(J_in_t * J_in_tw))
    # 隐藏层权重
    for l in range(J_hidden_t.shape[0]):
        C_layers.append(np.mean(J_hidden_t[l] * J_hidden_tw[l]))
    # 输出层权重
    C_layers.append(np.mean(J_out_t * J_out_tw))
    return np.mean(C_layers)


def compute_global_spin_overlap(S_a, S_b):
    """全局副本自旋重叠 q(t, t_w) - Eq. 19"""
    L_minus_1 = S_a.shape[1]
    q_layers = np.zeros(L_minus_1)
    for l in range(L_minus_1):
        q_layers[l] = np.mean(S_a[:, l, :] * S_b[:, l, :])
    return np.mean(q_layers)


def compute_global_weight_overlap(J_in_a, J_hidden_a, J_out_a,
                                   J_in_b, J_hidden_b, J_out_b):
    """全局副本权重重叠 Q(t, t_w) - Eq. 20"""
    Q_layers = []
    Q_layers.append(np.mean(J_in_a * J_in_b))
    for l in range(J_hidden_a.shape[0]):
        Q_layers.append(np.mean(J_hidden_a[l] * J_hidden_b[l]))
    Q_layers.append(np.mean(J_out_a * J_out_b))
    return np.mean(Q_layers)


def compute_layer_autocorr(S_t, S_tw, L):
    """层自旋自相关函数 c_l(t, t_w) - Eq. 13"""
    c_l = np.zeros(L - 1)
    for l in range(L - 1):
        c_l[l] = np.mean(S_t[:, l, :] * S_tw[:, l, :])
    return c_l


def compute_layer_overlap(S_a, S_b, L):
    """层副本重叠 q_l(t, t_w) - Eq. 17"""
    q_l = np.zeros(L - 1)
    for l in range(L - 1):
        q_l[l] = np.mean(S_a[:, l, :] * S_b[:, l, :])
    return q_l


def find_relaxation_time(t_array, c_array, threshold=1/np.e):
    """找到弛豫时间 τ：c(τ) = threshold"""
    if len(t_array) == 0 or len(c_array) == 0:
        return np.nan
    
    for i in range(len(c_array)):
        if c_array[i] < threshold:
            if i == 0:
                return t_array[0]
            t1, t2 = t_array[i-1], t_array[i]
            c1, c2 = c_array[i-1], c_array[i]
            if c1 == c2:
                return t1
            tau = t1 + (threshold - c1) * (t2 - t1) / (c2 - c1)
            return tau
    
    return np.nan


# ============================================================================
# 实验
# ============================================================================

def run_correlation_experiment(X_train, y_train, N, L, beta, 
                               mc_steps, t_w_list, log_interval=20, seed=42):
    """运行相关函数补充实验"""
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    
    np.random.seed(seed)
    
    # 创建两个副本
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    
    # 复制权重
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    E0 = compute_total_energy(net_a)
    print(f"E(0)={E0:.2f}")
    
    # 数据存储
    time_pts = [0]
    E_ratio = [1.0]
    
    # 保存 t_w 时刻的配置
    S_tw = {tw: None for tw in t_w_list}
    J_in_tw = {tw: None for tw in t_w_list}
    J_hidden_tw = {tw: None for tw in t_w_list}
    J_out_tw = {tw: None for tw in t_w_list}
    
    # 全局相关函数数据
    global_c = {tw: {'t': [], 'c': []} for tw in t_w_list}  # 自旋自相关
    global_C = {tw: {'t': [], 'C': []} for tw in t_w_list}  # 权重自相关
    global_q = {tw: {'t': [], 'q': []} for tw in t_w_list}  # 自旋重叠
    global_Q = {tw: {'t': [], 'Q': []} for tw in t_w_list}  # 权重重叠
    
    # 层相关函数数据
    layer_c = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    layer_q = {tw: {l: {'t': [], 'q': []} for l in range(L-1)} for tw in t_w_list}
    
    t0 = time()
    for step in range(1, mc_steps + 1):
        # MC 更新
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        # 同步权重
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        # 保存 t_w 配置
        for tw in t_w_list:
            if step == tw:
                S_tw[tw] = net_a.S.copy()
                J_in_tw[tw] = net_a.J_in.copy()
                J_hidden_tw[tw] = net_a.J_hidden.copy()
                J_out_tw[tw] = net_a.J_out.copy()
                print(f"  保存 t_w={tw} 配置")
        
        # 记录数据
        if step % log_interval == 0 or step == 1:
            time_pts.append(step)
            E_ratio.append(compute_total_energy(net_a) / E0)
            
            for tw in t_w_list:
                if S_tw[tw] is not None and step >= tw:
                    # 全局自旋自相关
                    c = compute_global_spin_autocorr(net_a.S, S_tw[tw])
                    global_c[tw]['t'].append(step)
                    global_c[tw]['c'].append(c)
                    
                    # 全局权重自相关
                    C = compute_global_weight_autocorr(
                        net_a.J_in, net_a.J_hidden, net_a.J_out,
                        J_in_tw[tw], J_hidden_tw[tw], J_out_tw[tw]
                    )
                    global_C[tw]['t'].append(step)
                    global_C[tw]['C'].append(C)
                    
                    # 全局自旋重叠
                    q = compute_global_spin_overlap(net_a.S, net_b.S)
                    global_q[tw]['t'].append(step)
                    global_q[tw]['q'].append(q)
                    
                    # 全局权重重叠
                    Q = compute_global_weight_overlap(
                        net_a.J_in, net_a.J_hidden, net_a.J_out,
                        net_b.J_in, net_b.J_hidden, net_b.J_out
                    )
                    global_Q[tw]['t'].append(step)
                    global_Q[tw]['Q'].append(Q)
                    
                    # 层自旋自相关
                    c_l = compute_layer_autocorr(net_a.S, S_tw[tw], L)
                    for l in range(L-1):
                        layer_c[tw][l]['t'].append(step)
                        layer_c[tw][l]['c'].append(c_l[l])
                    
                    # 层副本重叠
                    q_l = compute_layer_overlap(net_a.S, net_b.S, L)
                    for l in range(L-1):
                        layer_q[tw][l]['t'].append(step)
                        layer_q[tw][l]['q'].append(q_l[l])
        
        # 进度报告
        if step % max(1, mc_steps // 10) == 0:
            print(f"  Step {step}/{mc_steps}: E/E0={E_ratio[-1]:.4e}")
    
    # 计算弛豫时间
    relaxation_times_c = {}  # 自旋自相关弛豫时间
    relaxation_times_C = {}  # 权重自相关弛豫时间
    
    for tw in t_w_list:
        # 全局自旋弛豫时间
        t_arr = np.array(global_c[tw]['t'])
        c_arr = np.array(global_c[tw]['c'])
        relaxation_times_c[tw] = find_relaxation_time(t_arr, c_arr)
        
        # 全局权重弛豫时间
        t_arr = np.array(global_C[tw]['t'])
        C_arr = np.array(global_C[tw]['C'])
        relaxation_times_C[tw] = find_relaxation_time(t_arr, C_arr)
    
    # 层弛豫时间
    layer_relaxation_times = {}
    for tw in t_w_list:
        layer_relaxation_times[tw] = {}
        for l in range(L-1):
            t_arr = np.array(layer_c[tw][l]['t'])
            c_arr = np.array(layer_c[tw][l]['c'])
            layer_relaxation_times[tw][l] = find_relaxation_time(t_arr, c_arr)
    
    print(f"\n完成，耗时: {time()-t0:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta, 'alpha': alpha, 'ln_alpha': ln_alpha, 'E0': E0,
        'time_points': np.array(time_pts), 'energy_ratio': np.array(E_ratio),
        't_w_list': t_w_list,
        # 全局相关函数
        'global_c': global_c,
        'global_C': global_C,
        'global_q': global_q,
        'global_Q': global_Q,
        'relaxation_times_c': relaxation_times_c,
        'relaxation_times_C': relaxation_times_C,
        # 层相关函数
        'layer_c': layer_c,
        'layer_q': layer_q,
        'layer_relaxation_times': layer_relaxation_times
    }


def plot_global_correlations(results, output_dir):
    """绘制全局相关函数补充图 (SI Sec. S2 风格)"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    t_w_list = results['t_w_list']
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    
    # (a) 全局自旋自相关 c(t, t_w)
    ax = axes[0, 0]
    for i, tw in enumerate(t_w_list):
        d = results['global_c'][tw]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=tw_colors[i], label=f"t_w={tw}")
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('c(t, t_w)', fontsize=11)
    ax.set_title('(a) Global spin autocorrelation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (b) 全局权重自相关 C(t, t_w)
    ax = axes[0, 1]
    for i, tw in enumerate(t_w_list):
        d = results['global_C'][tw]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['C'], 'o-', ms=3, color=tw_colors[i], label=f"t_w={tw}")
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('C(t, t_w)', fontsize=11)
    ax.set_title('(b) Global weight autocorrelation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (c) 全局自旋重叠 q(t, t_w)
    ax = axes[1, 0]
    for i, tw in enumerate(t_w_list):
        d = results['global_q'][tw]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['q'], 'o-', ms=3, color=tw_colors[i], label=f"t_w={tw}")
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('q(t, t_w)', fontsize=11)
    ax.set_title('(c) Global spin replica overlap', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    
    # (d) 全局权重重叠 Q(t, t_w)
    ax = axes[1, 1]
    for i, tw in enumerate(t_w_list):
        d = results['global_Q'][tw]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['Q'], 'o-', ms=3, color=tw_colors[i], label=f"t_w={tw}")
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('Q(t, t_w)', fontsize=11)
    ax.set_title('(d) Global weight replica overlap', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.suptitle(f'Global Correlation Functions (SI Sec. S2)\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'global_correlations_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n全局相关函数图保存到: {fp}")
    return fp


def plot_layer_correlations(results, output_dir):
    """绘制层相关函数补充图 (SI Sec. S3 风格)"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    L = results['L']
    t_w_list = results['t_w_list']
    
    # 选择一个 t_w 值展示所有层
    tw = t_w_list[-1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    
    # (a) 层自旋自相关 c_l(t, t_w) - 所有层
    ax = axes[0, 0]
    for l in range(L-1):
        d = results['layer_c'][tw][l]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=layer_colors[l], label=f"l={l+1}")
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$c_l(t, t_w)$', fontsize=11)
    ax.set_title(f'(a) Layer spin autocorrelation (t_w={tw})', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (b) 层副本重叠 q_l(t, t_w) - 所有层
    ax = axes[0, 1]
    for l in range(L-1):
        d = results['layer_q'][tw][l]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['q'], 'o-', ms=3, color=layer_colors[l], label=f"l={l+1}")
    ax.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.7, label='1/e (phase boundary)')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$q_l(t, t_w)$', fontsize=11)
    ax.set_title(f'(b) Layer replica overlap (t_w={tw})', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    
    # (c) 层自相关 - 不同 t_w 比较（选择边界层和中心层）
    ax = axes[1, 0]
    liquid_layer = (L - 1) // 2
    solid_layer = 0
    
    for i, tw in enumerate(t_w_list):
        # 液态层（实线）
        d = results['layer_c'][tw][liquid_layer]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], '-', lw=2, color=tw_colors[i], 
                       label=f"l={liquid_layer+1} (liquid), t_w={tw}")
        
        # 固态层（虚线）
        d = results['layer_c'][tw][solid_layer]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['c'], '--', lw=2, color=tw_colors[i], alpha=0.7,
                       label=f"l={solid_layer+1} (solid), t_w={tw}")
    
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$c_l(t, t_w)$', fontsize=11)
    ax.set_title('(c) Liquid vs Solid layer autocorrelation', fontsize=12)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (d) 最终层重叠分布
    ax = axes[1, 1]
    layers = np.arange(1, L)
    
    q_l_final = []
    for l in range(L-1):
        d = results['layer_q'][tw][l]
        if len(d['q']) > 0:
            q_l_final.append(d['q'][-1])
        else:
            q_l_final.append(np.nan)
    
    bars = ax.bar(layers, q_l_final, color=layer_colors, edgecolor='black', alpha=0.8)
    ax.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2, label='1/e (phase boundary)')
    
    # 标注固态和液态
    for i, (l, q) in enumerate(zip(layers, q_l_final)):
        if not np.isnan(q):
            if q > 1/np.e:
                ax.annotate('S', (l, q + 0.02), ha='center', fontsize=10, fontweight='bold', color='blue')
            else:
                ax.annotate('L', (l, q + 0.02), ha='center', fontsize=10, fontweight='bold', color='red')
    
    ax.set_xlabel('Layer l', fontsize=11)
    ax.set_ylabel('$q_l^*$', fontsize=11)
    ax.set_title('(d) Final layer overlap profile', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, L - 0.5)
    ax.set_ylim(0, max(q_l_final) * 1.2 if q_l_final else 1.0)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Layer Correlation Functions (SI Sec. S3)\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'layer_correlations_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"层相关函数图保存到: {fp}")
    return fp


def plot_all_layer_autocorr_detail(results, output_dir):
    """绘制所有层的自相关函数详细图"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    L = results['L']
    t_w_list = results['t_w_list']
    
    # 创建 3x3 子图
    n_rows = (L - 1 + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    
    for l in range(L-1):
        ax = axes[l]
        for i, tw in enumerate(t_w_list):
            d = results['layer_c'][tw][l]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=tw_colors[i], label=f"t_w={tw}")
        
        ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('$c_l(t, t_w)$', fontsize=10)
        ax.set_title(f'Layer {l+1}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    # 隐藏多余的子图
    for l in range(L-1, len(axes)):
        axes[l].set_visible(False)
    
    plt.suptitle(f'Layer Autocorrelation Functions - All Layers\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'all_layer_autocorr_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"所有层自相关函数图保存到: {fp}")
    return fp


def main():
    print("=" * 70)
    print("相关函数补充实验")
    print("SI Sec. S2: 全局相关函数补充")
    print("SI Sec. S3: 层相关函数补充")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'correlation_supplement')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载 MNIST 数据...")
    X_train, y_train, X_test, y_test = load_mnist_binary(train_size=200, test_size=40, seed=42)
    print(f"训练集: {X_train.shape}")
    
    # 参数设置
    L = 8
    N = 10  # ln α ≈ 3.0
    beta = 1e5
    mc_steps = 2500
    t_w_list = [100, 300, 600, 1000]  # 更多 t_w 值
    log_interval = 25
    
    print(f"\n参数: L={L}, N={N}, β={beta:.0e}, MC steps={mc_steps}")
    print(f"t_w = {t_w_list}")
    print(f"ln α = {np.log(X_train.shape[0] / N):.2f}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行实验
    results = run_correlation_experiment(X_train, y_train, N, L, beta, 
                                         mc_steps, t_w_list, log_interval, seed=42)
    
    # 绘图
    print("\n生成图表...")
    fp1 = plot_global_correlations(results, output_dir)
    fp2 = plot_layer_correlations(results, output_dir)
    fp3 = plot_all_layer_autocorr_detail(results, output_dir)
    
    # 保存数据
    data_fp = os.path.join(output_dir, f'correlation_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(data_fp, results=results)
    print(f"数据保存到: {data_fp}")
    
    # 结果摘要
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)
    
    print(f"\n配置: ln α = {results['ln_alpha']:.2f}, L = {L}")
    
    print("\n全局弛豫时间:")
    print("-" * 50)
    for tw in t_w_list:
        tau_c = results['relaxation_times_c'][tw]
        tau_C = results['relaxation_times_C'][tw]
        tau_c_str = f"{tau_c:.1f}" if not np.isnan(tau_c) else "N/A"
        tau_C_str = f"{tau_C:.1f}" if not np.isnan(tau_C) else "N/A"
        print(f"  t_w={tw}: τ^c = {tau_c_str}, τ^C = {tau_C_str}")
    
    print("\n最终相关函数值:")
    print("-" * 50)
    tw = t_w_list[-1]
    c_final = results['global_c'][tw]['c'][-1] if results['global_c'][tw]['c'] else np.nan
    C_final = results['global_C'][tw]['C'][-1] if results['global_C'][tw]['C'] else np.nan
    q_final = results['global_q'][tw]['q'][-1] if results['global_q'][tw]['q'] else np.nan
    Q_final = results['global_Q'][tw]['Q'][-1] if results['global_Q'][tw]['Q'] else np.nan
    print(f"  c* = {c_final:.4f}")
    print(f"  C* = {C_final:.4f}")
    print(f"  q* = {q_final:.4f}")
    print(f"  Q* = {Q_final:.4f}")
    
    print("\n最终层重叠 q_l*:")
    print("-" * 50)
    solid_layers = []
    liquid_layers = []
    for l in range(L-1):
        d = results['layer_q'][tw][l]
        q_final = d['q'][-1] if len(d['q']) > 0 else np.nan
        if q_final > 1/np.e:
            phase = "固态"
            solid_layers.append(l+1)
        else:
            phase = "液态"
            liquid_layers.append(l+1)
        print(f"  层 {l+1}: q_l* = {q_final:.4f} ({phase})")
    
    print(f"\n固态层: {solid_layers}")
    print(f"液态层: {liquid_layers}")
    
    return results


if __name__ == "__main__":
    main()
