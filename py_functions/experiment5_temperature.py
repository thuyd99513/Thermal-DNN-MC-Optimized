#!/usr/bin/env python3
"""
experiment5_temperature.py - 实验五：温度依赖性 (Figure S1)

目的：研究温度 T 对训练动态的影响

测量内容：
- 不同温度下的能量衰减 E(t)/E(0)
- 不同温度下的准确率 A(t)

参数范围：T = 10^-2, 2×10^-3, 10^-3, 5×10^-4, 10^-5
对应 β = 1/T = 100, 500, 1000, 2000, 100000
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
        
        return X_train, y_train, X_test, y_test, y_train_labels, y_test_labels
        
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


def compute_train_accuracy(net):
    """计算训练准确率（基于约束满足比例）"""
    total_satisfied = 0
    total_constraints = 0
    
    # 输入层约束
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    gaps_0 = h_0 * net.S[:, 0, :]
    total_satisfied += np.sum(gaps_0 >= 0)
    total_constraints += gaps_0.size
    
    # 隐藏层约束
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        gaps_l = h_l * net.S[:, l+1, :]
        total_satisfied += np.sum(gaps_l >= 0)
        total_constraints += gaps_l.size
    
    # 输出层约束
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N
    gaps_out = h_out * net.S_out
    total_satisfied += np.sum(gaps_out >= 0)
    total_constraints += gaps_out.size
    
    return total_satisfied / total_constraints


def forward_pass(net, X):
    """标准 DNN 前向传播"""
    M = X.shape[0]
    
    # 输入层
    h = (net.J_in @ X.T).T / net.SQRT_N_IN
    S = np.sign(h)
    S[S == 0] = 1
    
    # 隐藏层
    for l in range(net.num_hidden_bond_layers):
        h = (net.J_hidden[l] @ S.T).T / net.SQRT_N
        S = np.sign(h)
        S[S == 0] = 1
    
    # 输出层
    h_out = (net.J_out @ S.T).T / net.SQRT_N
    
    return h_out


def compute_test_accuracy(net, X_test, y_test_labels):
    """计算测试集准确率"""
    h_out = forward_pass(net, X_test)
    predictions = np.argmax(h_out, axis=1)
    return np.mean(predictions == y_test_labels)


# ============================================================================
# 实验
# ============================================================================

def run_temperature_experiment(X_train, y_train, X_test, y_test_labels, 
                               N, L, temperatures, mc_steps, log_interval=20, seed=42):
    """运行温度依赖性实验"""
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    print(f"温度范围: {temperatures}")
    
    results = {}
    
    for T in temperatures:
        beta = 1.0 / T
        print(f"\n{'='*50}")
        print(f"温度 T = {T:.0e}, β = {beta:.0e}")
        print(f"{'='*50}")
        
        np.random.seed(seed)
        net = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
        
        E0 = compute_total_energy(net)
        print(f"E(0) = {E0:.2f}")
        
        time_pts = [0]
        E_ratio = [1.0]
        A_train = [compute_train_accuracy(net)]
        A_test = [compute_test_accuracy(net, X_test, y_test_labels)]
        
        t0 = time()
        for step in range(1, mc_steps + 1):
            net.mc_step_vectorized()
            
            if step % log_interval == 0 or step == 1:
                time_pts.append(step)
                E_ratio.append(compute_total_energy(net) / E0)
                A_train.append(compute_train_accuracy(net))
                A_test.append(compute_test_accuracy(net, X_test, y_test_labels))
            
            if step % max(1, mc_steps // 5) == 0:
                print(f"  Step {step}/{mc_steps}: E/E0={E_ratio[-1]:.4e}, A_train={A_train[-1]:.3f}, A_test={A_test[-1]:.3f}")
        
        print(f"  完成，耗时: {time()-t0:.1f}s")
        
        results[T] = {
            'beta': beta,
            'time_points': np.array(time_pts),
            'energy_ratio': np.array(E_ratio),
            'train_accuracy': np.array(A_train),
            'test_accuracy': np.array(A_test),
            'E0': E0
        }
    
    return {
        'M': M, 'N': N, 'L': L, 'alpha': alpha, 'ln_alpha': ln_alpha,
        'temperatures': temperatures, 'mc_steps': mc_steps,
        'temperature_results': results
    }


def plot_figure_s1(results, output_dir):
    """绘制 Figure S1 风格的图表"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    temperatures = results['temperatures']
    temp_results = results['temperature_results']
    
    # 颜色映射
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
    
    # (a) 能量衰减 E(t)/E(0)
    ax1 = axes[0]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax1.loglog(data['time_points'][1:], data['energy_ratio'][1:], 
                   'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    
    ax1.set_xlabel('t (MC steps)', fontsize=12)
    ax1.set_ylabel('E(t)/E(0)', fontsize=12)
    ax1.set_title('(a) Energy decay', fontsize=13)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    
    # (b) 训练准确率 A_train(t)
    ax2 = axes[1]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax2.semilogx(data['time_points'], data['train_accuracy'], 
                     'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    
    ax2.set_xlabel('t (MC steps)', fontsize=12)
    ax2.set_ylabel('$A_{train}(t)$', fontsize=12)
    ax2.set_title('(b) Training accuracy', fontsize=13)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)
    
    plt.suptitle(f'Temperature Dependence (Figure S1)\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment5_figure_s1_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure S1 保存到: {fp}")
    return fp


def plot_detailed_temperature(results, output_dir):
    """绘制详细的温度依赖性图表"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    temperatures = results['temperatures']
    temp_results = results['temperature_results']
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(temperatures)))
    
    # (a) 能量衰减 - 线性坐标
    ax = axes[0, 0]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax.plot(data['time_points'], data['energy_ratio'], 
                'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('E(t)/E(0)', fontsize=11)
    ax.set_title('(a) Energy decay (linear scale)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # (b) 能量衰减 - 对数坐标
    ax = axes[0, 1]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax.loglog(data['time_points'][1:], data['energy_ratio'][1:], 
                  'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('E(t)/E(0)', fontsize=11)
    ax.set_title('(b) Energy decay (log-log scale)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    # (c) 训练准确率
    ax = axes[1, 0]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax.semilogx(data['time_points'], data['train_accuracy'], 
                    'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$A_{train}(t)$', fontsize=11)
    ax.set_title('(c) Training accuracy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    # (d) 测试准确率
    ax = axes[1, 1]
    for i, T in enumerate(temperatures):
        data = temp_results[T]
        ax.semilogx(data['time_points'], data['test_accuracy'], 
                    'o-', ms=3, color=colors[i], label=f'T={T:.0e}')
    ax.set_xlabel('t (MC steps)', fontsize=11)
    ax.set_ylabel('$A_{test}(t)$', fontsize=11)
    ax.set_title('(d) Test accuracy', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    
    plt.suptitle(f'Temperature Dependence - Detailed Analysis\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment5_detailed_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"详细图表保存到: {fp}")
    return fp


def plot_final_values(results, output_dir):
    """绘制最终值随温度的变化"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    temperatures = results['temperatures']
    temp_results = results['temperature_results']
    
    T_arr = np.array(temperatures)
    E_final = [temp_results[T]['energy_ratio'][-1] for T in temperatures]
    A_train_final = [temp_results[T]['train_accuracy'][-1] for T in temperatures]
    A_test_final = [temp_results[T]['test_accuracy'][-1] for T in temperatures]
    
    # (a) 最终能量
    ax = axes[0]
    ax.semilogx(T_arr, E_final, 'o-', ms=8, color='blue')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('$E^*/E(0)$', fontsize=11)
    ax.set_title('(a) Final energy ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    # (b) 最终训练准确率
    ax = axes[1]
    ax.semilogx(T_arr, A_train_final, 'o-', ms=8, color='green')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('$A^*_{train}$', fontsize=11)
    ax.set_title('(b) Final training accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0.5, 1.0)
    
    # (c) 最终测试准确率
    ax = axes[2]
    ax.semilogx(T_arr, A_test_final, 'o-', ms=8, color='red')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('$A^*_{test}$', fontsize=11)
    ax.set_title('(c) Final test accuracy', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_ylim(0.4, 1.0)
    
    plt.suptitle(f'Final Values vs Temperature\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment5_final_values_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"最终值图表保存到: {fp}")
    return fp


def main():
    print("=" * 70)
    print("实验五：温度依赖性 (Figure S1)")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              '..', 'reports', 'experiment5_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载 MNIST 数据...")
    X_train, y_train, X_test, y_test, y_train_labels, y_test_labels = load_mnist_binary(
        train_size=200, test_size=40, seed=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 参数设置
    L = 8
    N = 10
    mc_steps = 1500
    log_interval = 25
    
    # 温度范围（论文中的值）
    temperatures = [1e-2, 2e-3, 1e-3, 5e-4, 1e-5]
    
    print(f"\n参数: L={L}, N={N}, MC steps={mc_steps}")
    print(f"温度: {temperatures}")
    print(f"ln α = {np.log(X_train.shape[0] / N):.2f}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=1e3, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行实验
    results = run_temperature_experiment(
        X_train, y_train, X_test, y_test_labels,
        N, L, temperatures, mc_steps, log_interval, seed=42
    )
    
    # 绘图
    print("\n生成图表...")
    fp1 = plot_figure_s1(results, output_dir)
    fp2 = plot_detailed_temperature(results, output_dir)
    fp3 = plot_final_values(results, output_dir)
    
    # 结果摘要
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)
    
    print(f"\n配置: ln α = {results['ln_alpha']:.2f}, L = {L}")
    
    print("\n最终值:")
    print("-" * 60)
    print(f"{'T':<12} {'β':<12} {'E*/E(0)':<12} {'A*_train':<12} {'A*_test':<12}")
    print("-" * 60)
    
    for T in temperatures:
        data = results['temperature_results'][T]
        print(f"{T:<12.0e} {data['beta']:<12.0e} {data['energy_ratio'][-1]:<12.4e} "
              f"{data['train_accuracy'][-1]:<12.3f} {data['test_accuracy'][-1]:<12.3f}")
    
    return results


if __name__ == "__main__":
    main()
