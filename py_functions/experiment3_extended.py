#!/usr/bin/env python3
"""
experiment3_extended.py - 实验三扩展版：更长时间的层依赖训练动态

使用更多 MC 步数以观察完整的弛豫行为。
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


def compute_layer_autocorr(S_t, S_tw, L):
    c_l = np.zeros(L - 1)
    for l in range(L - 1):
        c_l[l] = np.mean(S_t[:, l, :] * S_tw[:, l, :])
    return c_l


def compute_layer_overlap(S_a, S_b, L):
    q_l = np.zeros(L - 1)
    for l in range(L - 1):
        q_l[l] = np.mean(S_a[:, l, :] * S_b[:, l, :])
    return q_l


def find_relaxation_time(t_array, c_array, threshold=1/np.e):
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


def run_experiment(X_train, y_train, N, L, beta, mc_steps, t_w_list, log_interval=20, seed=42):
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    
    np.random.seed(seed)
    
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    E0 = compute_total_energy(net_a)
    print(f"E(0)={E0:.2f}")
    
    time_pts = [0]
    E_ratio = [1.0]
    
    S_tw = {tw: None for tw in t_w_list}
    layer_autocorr = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    layer_overlap = {tw: {l: {'t': [], 'q': []} for l in range(L-1)} for tw in t_w_list}
    
    t0 = time()
    for step in range(1, mc_steps + 1):
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        for tw in t_w_list:
            if step == tw:
                S_tw[tw] = net_a.S.copy()
                print(f"  保存 t_w={tw} 配置")
        
        if step % log_interval == 0 or step == 1:
            time_pts.append(step)
            E_ratio.append(compute_total_energy(net_a) / E0)
            
            for tw in t_w_list:
                if S_tw[tw] is not None and step >= tw:
                    c_l = compute_layer_autocorr(net_a.S, S_tw[tw], L)
                    q_l = compute_layer_overlap(net_a.S, net_b.S, L)
                    for l in range(L-1):
                        layer_autocorr[tw][l]['t'].append(step)
                        layer_autocorr[tw][l]['c'].append(c_l[l])
                        layer_overlap[tw][l]['t'].append(step)
                        layer_overlap[tw][l]['q'].append(q_l[l])
        
        if step % max(1, mc_steps // 10) == 0:
            print(f"  Step {step}/{mc_steps}: E/E0={E_ratio[-1]:.4e}")
    
    # 计算弛豫时间
    relaxation_times = {}
    for tw in t_w_list:
        relaxation_times[tw] = {}
        for l in range(L-1):
            t_arr = np.array(layer_autocorr[tw][l]['t'])
            c_arr = np.array(layer_autocorr[tw][l]['c'])
            relaxation_times[tw][l] = find_relaxation_time(t_arr, c_arr)
    
    print(f"\n完成，耗时: {time()-t0:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta, 'alpha': alpha, 'ln_alpha': ln_alpha, 'E0': E0,
        'time_points': np.array(time_pts), 'energy_ratio': np.array(E_ratio),
        'layer_autocorr': layer_autocorr,
        'layer_overlap': layer_overlap,
        'relaxation_times': relaxation_times,
        't_w_list': t_w_list
    }


def plot_figure3_style(results, output_dir):
    """绘制论文 Figure 3 风格的图表"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    L = results['L']
    t_w_list = results['t_w_list']
    
    tw_colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(t_w_list)))
    layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
    
    # (a) 层自相关函数 - 选择代表性层
    ax = axes[0, 0]
    
    # 选择液态层（中心）和固态层（边界）
    liquid_layers = [L//2 - 1, L//2]  # 中心层
    solid_layers = [0, L-2]  # 边界层
    
    for i, tw in enumerate(t_w_list):
        # 液态层用实线
        for ll in liquid_layers:
            d = results['layer_autocorr'][tw][ll]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['c'], '-', lw=1.5, color=tw_colors[i], alpha=0.7,
                           label=f"l={ll+1}, t_w={tw}" if i == 0 else None)
        
        # 固态层用虚线
        for sl in solid_layers:
            d = results['layer_autocorr'][tw][sl]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['c'], '--', lw=1.5, color=tw_colors[i], alpha=0.7)
    
    ax.axhline(y=1/np.e, color='gray', linestyle=':', alpha=0.5, label='1/e')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$c_l(t, t_w)$', fontsize=11)
    ax.set_title('(a) Layer autocorrelation', fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (b) 弛豫时间 τ_l^c vs t_w
    ax = axes[0, 1]
    has_data = False
    for l in range(L-1):
        tau_vs_tw = []
        tw_vals = []
        for tw in t_w_list:
            tau = results['relaxation_times'][tw][l]
            if not np.isnan(tau):
                tau_vs_tw.append(tau)
                tw_vals.append(tw)
                has_data = True
        
        if len(tau_vs_tw) > 0:
            ax.loglog(tw_vals, tau_vs_tw, 'o-', ms=6, color=layer_colors[l], label=f"l={l+1}")
    
    if not has_data:
        ax.text(0.5, 0.5, '自相关函数未衰减到 1/e\n需要更多 MC 步数', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('$t_w$', fontsize=11)
    ax.set_ylabel('$τ_l^c$', fontsize=11)
    ax.set_title('(b) Relaxation time vs waiting time', fontsize=12)
    if has_data:
        ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # (c) 层副本重叠 q_l(t, t_w) - 所有层
    ax = axes[1, 0]
    tw = t_w_list[-1]
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
        if len(d['t']) > 0:
            ax.semilogx(d['t'], d['q'], 'o-', ms=3, color=layer_colors[l], label=f"l={l+1}")
    
    ax.axhline(y=1/np.e, color='red', linestyle='--', alpha=0.7, label='1/e (phase boundary)')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$q_l(t, t_w)$', fontsize=11)
    ax.set_title(f'(c) Layer replica overlap (t_w={tw})', fontsize=12)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)
    
    # (d) 最终层重叠 q_l* vs 层 l
    ax = axes[1, 1]
    layers = np.arange(1, L)
    
    q_l_final = []
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
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
                ax.annotate('S', (l, q + 0.03), ha='center', fontsize=10, fontweight='bold', color='blue')
            else:
                ax.annotate('L', (l, q + 0.03), ha='center', fontsize=10, fontweight='bold', color='red')
    
    ax.set_xlabel('Layer l', fontsize=11)
    ax.set_ylabel('$q_l^*$', fontsize=11)
    ax.set_title('(d) Final layer overlap profile (S=Solid, L=Liquid)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0.5, L - 0.5)
    ax.set_ylim(0, max(q_l_final) * 1.2 if q_l_final else 1.0)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment3_figure3_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表保存到: {fp}")
    return fp


def main():
    print("=" * 60)
    print("实验三扩展版：层依赖训练动态 (Figure 3)")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'experiment3_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据 - 使用较小的数据集以加速
    print("\n加载 MNIST 数据...")
    X_train, y_train, X_test, y_test = load_mnist_binary(train_size=200, test_size=40, seed=42)
    print(f"训练集: {X_train.shape}")
    
    # 参数设置 - 使用较小的网络以加速
    L = 8
    N = 10  # ln α ≈ 3.0
    beta = 1e5
    mc_steps = 3000  # 增加步数
    t_w_list = [200, 600, 1200]
    log_interval = 30
    
    print(f"\n参数: L={L}, N={N}, β={beta:.0e}, MC steps={mc_steps}")
    print(f"t_w = {t_w_list}")
    print(f"ln α = {np.log(X_train.shape[0] / N):.2f}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行实验
    results = run_experiment(X_train, y_train, N, L, beta, mc_steps, t_w_list, log_interval, seed=42)
    
    # 绘图
    print("\n生成图表...")
    fp = plot_figure3_style(results, output_dir)
    
    # 保存数据
    data_fp = os.path.join(output_dir, f'experiment3_extended_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(data_fp, results=results)
    print(f"数据保存到: {data_fp}")
    
    # 结果摘要
    print("\n" + "=" * 60)
    print("实验三结果摘要")
    print("=" * 60)
    
    print(f"\n配置: ln α = {results['ln_alpha']:.2f}, L = {L}")
    
    print("\n最终层重叠 q_l*:")
    print("-" * 50)
    tw = t_w_list[-1]
    solid_layers = []
    liquid_layers = []
    for l in range(L-1):
        d = results['layer_overlap'][tw][l]
        q_final = d['q'][-1] if len(d['q']) > 0 else np.nan
        if q_final > 1/np.e:
            phase = "固态 (Solid)"
            solid_layers.append(l+1)
        else:
            phase = "液态 (Liquid)"
            liquid_layers.append(l+1)
        print(f"  层 {l+1}: q_l* = {q_final:.4f} - {phase}")
    
    print(f"\n固态层: {solid_layers}")
    print(f"液态层: {liquid_layers}")
    
    # 计算穿透深度
    if solid_layers:
        xi_input = max([l for l in solid_layers if l <= L//2], default=0)
        xi_output = max([L - l for l in solid_layers if l > L//2], default=0)
        print(f"\n穿透深度: ξ_input = {xi_input}, ξ_output = {xi_output}")
    
    return results


if __name__ == "__main__":
    main()
