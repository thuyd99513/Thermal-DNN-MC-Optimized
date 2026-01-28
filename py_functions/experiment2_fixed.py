#!/usr/bin/env python3
"""
experiment2_fixed.py - 修复版实验二：全局训练动态 (Figure 2)

修复了训练准确率计算和层自相关函数记录的问题。
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
    # 输入层
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    E += compute_energy_numba(h_0 * net.S[:, 0, :])
    # 隐藏层
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        E += compute_energy_numba(h_l * net.S[:, l+1, :])
    # 输出层
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


def compute_train_accuracy_dnn(net, X_train, y_train):
    """使用 DNN 前向传播计算训练集准确率（更合理的定义）"""
    out = dnn_forward(X_train, net.J_in, net.J_hidden, net.J_out, net.SQRT_N_IN, net.SQRT_N)
    return np.sum(np.all(out == y_train, axis=1)) / len(y_train)


def compute_fraction_satisfied(net):
    """计算满足约束的样本比例"""
    correct = 0
    for mu in range(net.M):
        ok = True
        h = net.J_in @ net.S_in[mu] / net.SQRT_N_IN
        if np.any(h * net.S[mu, 0, :] < 0): ok = False
        if ok:
            for l in range(net.num_hidden_bond_layers):
                h = net.J_hidden[l] @ net.S[mu, l, :] / net.SQRT_N
                if np.any(h * net.S[mu, l+1, :] < 0): ok = False; break
        if ok:
            h = net.J_out @ net.S[mu, -1, :] / net.SQRT_N
            if np.any(h * net.S_out[mu] < 0): ok = False
        if ok: correct += 1
    return correct / net.M


@njit(cache=True, fastmath=True)
def autocorr_numba(S_t, S_tw):
    """计算自相关函数 c(t, t_w)"""
    return np.sum(S_t * S_tw) / S_t.size


@njit(cache=True, fastmath=True)
def overlap_numba(S_a, S_b):
    """计算副本重叠 q(t, t_w)"""
    return np.sum(S_a * S_b) / S_a.size


# ============================================================================
# 实验
# ============================================================================

def run_experiment(X_train, y_train, X_test, y_test, N, L, beta, 
                   mc_steps, t_w_list, log_interval=20, seed=42):
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    
    np.random.seed(seed)
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    net_b.J_in, net_b.J_hidden, net_b.J_out = net_a.J_in.copy(), net_a.J_hidden.copy(), net_a.J_out.copy()
    
    E0 = compute_total_energy(net_a)
    A_train_0 = compute_train_accuracy_dnn(net_a, X_train, y_train)
    A_test_0 = compute_test_accuracy(net_a, X_test, y_test)
    print(f"E(0)={E0:.2f}, A_train={A_train_0:.4f}, A_test={A_test_0:.4f}")
    
    # 数据存储
    time_pts = [0]
    E_ratio = [1.0]
    A_train = [A_train_0]
    A_test = [A_test_0]
    
    S_tw = {tw: None for tw in t_w_list}
    autocorr = {tw: {'t': [], 'c': []} for tw in t_w_list}
    overlap = {tw: {'t': [], 'q': []} for tw in t_w_list}
    layer_autocorr = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    layer_overlap = {l: {'t': [], 'q': []} for l in range(L-1)}
    
    t0 = time()
    for step in range(1, mc_steps + 1):
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        net_b.J_in, net_b.J_hidden, net_b.J_out = net_a.J_in.copy(), net_a.J_hidden.copy(), net_a.J_out.copy()
        
        # 保存 t_w 配置
        for tw in t_w_list:
            if step == tw:
                S_tw[tw] = net_a.S.copy()
                print(f"  保存 t_w={tw} 配置")
        
        if step % log_interval == 0 or step == 1:
            time_pts.append(step)
            E_ratio.append(compute_total_energy(net_a) / E0)
            A_train.append(compute_train_accuracy_dnn(net_a, X_train, y_train))
            A_test.append(compute_test_accuracy(net_a, X_test, y_test))
            
            # 自相关和重叠
            for tw in t_w_list:
                if S_tw[tw] is not None and step >= tw:
                    autocorr[tw]['t'].append(step)
                    autocorr[tw]['c'].append(autocorr_numba(net_a.S, S_tw[tw]))
                    overlap[tw]['t'].append(step)
                    overlap[tw]['q'].append(overlap_numba(net_a.S, net_b.S))
                    
                    # 层自相关
                    for l in range(L-1):
                        c_l = np.mean(net_a.S[:, l, :] * S_tw[tw][:, l, :])
                        layer_autocorr[tw][l]['t'].append(step)
                        layer_autocorr[tw][l]['c'].append(c_l)
            
            # 层重叠
            for l in range(L-1):
                q_l = np.mean(net_a.S[:, l, :] * net_b.S[:, l, :])
                layer_overlap[l]['t'].append(step)
                layer_overlap[l]['q'].append(q_l)
        
        if step % max(1, mc_steps // 5) == 0:
            print(f"  Step {step}/{mc_steps}: E/E0={E_ratio[-1]:.4e}, A_train={A_train[-1]:.4f}, A_test={A_test[-1]:.4f}")
    
    print(f"完成，耗时: {time()-t0:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta, 'alpha': alpha, 'ln_alpha': ln_alpha, 'E0': E0,
        'time_points': np.array(time_pts), 'energy_ratio': np.array(E_ratio),
        'accuracy_train': np.array(A_train), 'accuracy_test': np.array(A_test),
        'autocorr_data': autocorr, 'overlap_data': overlap,
        'layer_autocorr_data': layer_autocorr, 'layer_overlap_data': layer_overlap,
        't_w_list': t_w_list
    }


def plot_results(results_list, output_dir):
    """绘制 Figure 2 风格的结果图"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # (a) E(t)/E(0)
    ax = axes[0, 0]
    for i, r in enumerate(results_list):
        t = r['time_points']
        ax.loglog(t[t > 0], r['energy_ratio'][t > 0], 'o-', ms=3, color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('E(t)/E(0)', fontsize=11)
    ax.set_title('(a) Energy decay', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (b) A_train
    ax = axes[0, 1]
    for i, r in enumerate(results_list):
        ax.semilogx(r['time_points'], r['accuracy_train'], 'o-', ms=3, color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('A_train(t)', fontsize=11)
    ax.set_title('(b) Training accuracy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (c) A_test
    ax = axes[0, 2]
    for i, r in enumerate(results_list):
        ax.semilogx(r['time_points'], r['accuracy_test'], 'o-', ms=3, color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('A_test(t)', fontsize=11)
    ax.set_title('(c) Test accuracy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (d) Final values
    ax = axes[0, 3]
    ln_a = [r['ln_alpha'] for r in results_list]
    E_s = [r['energy_ratio'][-1] for r in results_list]
    A_s = [r['accuracy_test'][-1] for r in results_list]
    ax2 = ax.twinx()
    l1, = ax.semilogy(ln_a, E_s, 'bo-', ms=8, label='E*/E(0)')
    l2, = ax2.plot(ln_a, A_s, 'rs-', ms=8, label='A*_test')
    ax.set_xlabel('ln α', fontsize=11)
    ax.set_ylabel('E*/E(0)', color='b', fontsize=11)
    ax2.set_ylabel('A*_test', color='r', fontsize=11)
    ax.set_title('(d) Final values vs ln α', fontsize=12)
    ax.legend(handles=[l1, l2], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (e) c(t, t_w)
    ax = axes[1, 0]
    if results_list:
        r = results_list[0]
        tw_c = plt.cm.plasma(np.linspace(0, 1, len(r['t_w_list'])))
        for i, tw in enumerate(r['t_w_list']):
            d = r['autocorr_data'][tw]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['c'], 'o-', ms=3, color=tw_c[i], label=f"t_w={tw}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('c(t, t_w)', fontsize=11)
    ax.set_title('(e) Spin autocorrelation', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # (f) q(t, t_w)
    ax = axes[1, 1]
    if results_list:
        r = results_list[0]
        for i, tw in enumerate(r['t_w_list']):
            d = r['overlap_data'][tw]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['q'], 'o-', ms=3, color=tw_c[i], label=f"t_w={tw}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('q(t, t_w)', fontsize=11)
    ax.set_title('(f) Replica overlap', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # (g) Layer autocorrelation
    ax = axes[1, 2]
    if results_list:
        r = results_list[0]
        tw = r['t_w_list'][0] if r['t_w_list'] else None
        if tw:
            L = r['L']
            lc = plt.cm.coolwarm(np.linspace(0, 1, L-1))
            for l in range(L-1):
                d = r['layer_autocorr_data'][tw][l]
                if len(d['t']) > 0:
                    ax.semilogx(d['t'], d['c'], 'o-', ms=2, color=lc[l], label=f"l={l+1}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('c_l(t, t_w)', fontsize=11)
    ax.set_title(f'(g) Layer autocorrelation (t_w={r["t_w_list"][0] if r["t_w_list"] else "N/A"})', fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # (h) Layer overlap
    ax = axes[1, 3]
    if results_list:
        r = results_list[0]
        L = r['L']
        lc = plt.cm.coolwarm(np.linspace(0, 1, L-1))
        for l in range(L-1):
            d = r['layer_overlap_data'][l]
            if len(d['t']) > 0:
                ax.semilogx(d['t'], d['q'], 'o-', ms=2, color=lc[l], label=f"l={l+1}")
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('q_l(t)', fontsize=11)
    ax.set_title('(h) Layer overlap', fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment2_figure2_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表保存到: {fp}")
    return fp


def main():
    print("=" * 60)
    print("实验二：全局训练动态 (Figure 2) - 修复版")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'experiment2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_binary(train_size=300, test_size=60, seed=42)
    
    # 参数
    L, beta, mc_steps = 8, 1e5, 1000
    t_w_list = [100, 300, 600]  # 确保 t_w 在 log_interval 的倍数上
    log_interval = 20
    
    configs = [{'N': 30}, {'N': 15}, {'N': 8}]  # ln α ≈ 2.3, 3.0, 3.6
    
    print(f"\n参数: L={L}, β={beta:.0e}, MC steps={mc_steps}")
    print(f"t_w = {t_w_list}")
    
    # 预热
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行
    results_list = []
    for i, cfg in enumerate(configs):
        print(f"\n配置 {i+1}/{len(configs)}: N={cfg['N']}")
        results = run_experiment(X_train, y_train, X_test, y_test, cfg['N'], L, beta, mc_steps, t_w_list, log_interval, 42+i)
        results_list.append(results)
    
    # 绘图
    print("\n生成图表...")
    fp = plot_results(results_list, output_dir)
    
    # 保存数据
    data_fp = os.path.join(output_dir, f'experiment2_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')
    np.savez(data_fp, results_list=results_list)
    print(f"数据保存到: {data_fp}")
    
    # 摘要
    print("\n" + "=" * 60)
    print("实验二结果摘要")
    print("=" * 60)
    print(f"{'ln α':>8} | {'α':>8} | {'E*/E(0)':>12} | {'A*_train':>10} | {'A*_test':>10}")
    print("-" * 60)
    for r in results_list:
        print(f"{r['ln_alpha']:>8.2f} | {r['alpha']:>8.1f} | {r['energy_ratio'][-1]:>12.4e} | {r['accuracy_train'][-1]:>10.4f} | {r['accuracy_test'][-1]:>10.4f}")
    
    return results_list, fp


if __name__ == "__main__":
    main()
