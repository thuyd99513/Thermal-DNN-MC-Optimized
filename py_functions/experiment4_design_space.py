#!/usr/bin/env python3
"""
experiment4_design_space.py - 实验四：设计空间结构 (Figure 4)

目的：分析液态层和固态层的参数空间结构

测量内容：
1. 重叠参数分布 P(Q_l^ab) - 层 l 的副本权重重叠分布
2. 重叠矩阵热图 - 配置间的相似性矩阵
3. 层次聚类树状图 - 使用 Complete Linkage 算法

关键物理量：
Q_l^ab = (1/N_l·N_{l-1}) Σ_{i,j} J_{l,i,j}^a · J_{l,i,j}^b
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from datetime import datetime
from time import time
from numba import njit
import urllib.request
import gzip
import struct
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

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

def compute_layer_weight_overlap(J_a, J_b):
    """计算层权重重叠 Q_l^ab = (1/N_l·N_{l-1}) Σ_{i,j} J_{l,i,j}^a · J_{l,i,j}^b"""
    return np.mean(J_a * J_b)


def compute_all_pairwise_overlaps(weight_configs, layer_idx):
    """计算所有配置对之间的层权重重叠"""
    n_configs = len(weight_configs)
    Q_matrix = np.zeros((n_configs, n_configs))
    
    for i in range(n_configs):
        for j in range(n_configs):
            if layer_idx == 0:  # 输入层
                Q_matrix[i, j] = compute_layer_weight_overlap(
                    weight_configs[i]['J_in'], weight_configs[j]['J_in']
                )
            elif layer_idx == -1:  # 输出层
                Q_matrix[i, j] = compute_layer_weight_overlap(
                    weight_configs[i]['J_out'], weight_configs[j]['J_out']
                )
            else:  # 隐藏层
                Q_matrix[i, j] = compute_layer_weight_overlap(
                    weight_configs[i]['J_hidden'][layer_idx - 1],
                    weight_configs[j]['J_hidden'][layer_idx - 1]
                )
    
    return Q_matrix


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


# ============================================================================
# 实验
# ============================================================================

def run_experiment4(X_train, y_train, N, L, beta, mc_steps, 
                    n_replicas=10, save_interval=100, seed=42):
    """运行实验四：设计空间结构"""
    M = X_train.shape[0]
    alpha, ln_alpha = M / N, np.log(M / N)
    
    print(f"\n配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    print(f"副本数: {n_replicas}, MC 步数: {mc_steps}")
    
    # 创建多个副本
    replicas = []
    for r in range(n_replicas):
        net = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed + r * 1000)
        replicas.append(net)
    
    # 共享初始权重
    for r in range(1, n_replicas):
        replicas[r].J_in = replicas[0].J_in.copy()
        replicas[r].J_hidden = replicas[0].J_hidden.copy()
        replicas[r].J_out = replicas[0].J_out.copy()
    
    E0 = compute_total_energy(replicas[0])
    print(f"E(0) = {E0:.2f}")
    
    # 存储配置
    saved_configs = []
    energy_history = []
    
    t0 = time()
    for step in range(1, mc_steps + 1):
        # 所有副本执行 MC 步
        for net in replicas:
            net.mc_step_vectorized()
        
        # 同步权重（所有副本使用相同权重）
        for r in range(1, n_replicas):
            replicas[r].J_in = replicas[0].J_in.copy()
            replicas[r].J_hidden = replicas[0].J_hidden.copy()
            replicas[r].J_out = replicas[0].J_out.copy()
        
        # 保存配置
        if step % save_interval == 0:
            for r, net in enumerate(replicas):
                saved_configs.append({
                    'step': step,
                    'replica': r,
                    'J_in': net.J_in.copy(),
                    'J_hidden': net.J_hidden.copy(),
                    'J_out': net.J_out.copy()
                })
            
            E = compute_total_energy(replicas[0])
            energy_history.append((step, E / E0))
        
        if step % max(1, mc_steps // 10) == 0:
            E = compute_total_energy(replicas[0])
            print(f"  Step {step}/{mc_steps}: E/E0 = {E/E0:.4e}")
    
    print(f"\n完成，耗时: {time()-t0:.1f}s")
    print(f"保存了 {len(saved_configs)} 个配置")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta, 'alpha': alpha, 'ln_alpha': ln_alpha,
        'n_replicas': n_replicas, 'mc_steps': mc_steps,
        'saved_configs': saved_configs,
        'energy_history': energy_history
    }


def analyze_design_space(results, liquid_layer, solid_layer):
    """分析设计空间结构"""
    configs = results['saved_configs']
    L = results['L']
    
    print(f"\n分析设计空间结构...")
    print(f"液态层: l={liquid_layer}, 固态层: l={solid_layer}")
    
    # 计算液态层和固态层的重叠矩阵
    Q_liquid = compute_all_pairwise_overlaps(configs, liquid_layer)
    Q_solid = compute_all_pairwise_overlaps(configs, solid_layer)
    
    # 提取非对角元素用于分布分析
    n = len(configs)
    Q_liquid_offdiag = []
    Q_solid_offdiag = []
    
    for i in range(n):
        for j in range(i+1, n):
            Q_liquid_offdiag.append(Q_liquid[i, j])
            Q_solid_offdiag.append(Q_solid[i, j])
    
    Q_liquid_offdiag = np.array(Q_liquid_offdiag)
    Q_solid_offdiag = np.array(Q_solid_offdiag)
    
    # 层次聚类
    # 将重叠矩阵转换为距离矩阵
    D_liquid = 1 - Q_liquid
    D_solid = 1 - Q_solid
    
    # 确保对角线为 0
    np.fill_diagonal(D_liquid, 0)
    np.fill_diagonal(D_solid, 0)
    
    # 确保对称性
    D_liquid = (D_liquid + D_liquid.T) / 2
    D_solid = (D_solid + D_solid.T) / 2
    
    # 确保距离非负（重叠可能大于1）
    D_liquid = np.maximum(D_liquid, 0)
    D_solid = np.maximum(D_solid, 0)
    
    # 转换为压缩距离矩阵
    D_liquid_condensed = squareform(D_liquid)
    D_solid_condensed = squareform(D_solid)
    
    # Complete Linkage 聚类
    Z_liquid = linkage(D_liquid_condensed, method='complete')
    Z_solid = linkage(D_solid_condensed, method='complete')
    
    return {
        'Q_liquid': Q_liquid,
        'Q_solid': Q_solid,
        'Q_liquid_offdiag': Q_liquid_offdiag,
        'Q_solid_offdiag': Q_solid_offdiag,
        'Z_liquid': Z_liquid,
        'Z_solid': Z_solid,
        'liquid_layer': liquid_layer,
        'solid_layer': solid_layer
    }


def plot_figure4(results, analysis, output_dir):
    """绘制 Figure 4 风格的图表"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    liquid_layer = analysis['liquid_layer']
    solid_layer = analysis['solid_layer']
    
    # (a) 液态层重叠分布 P(Q_l^ab)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(analysis['Q_liquid_offdiag'], bins=30, density=True, 
             alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=np.mean(analysis['Q_liquid_offdiag']), color='red', 
                linestyle='--', label=f'mean={np.mean(analysis["Q_liquid_offdiag"]):.3f}')
    ax1.set_xlabel('$Q_l^{ab}$', fontsize=12)
    ax1.set_ylabel('$P(Q_l^{ab})$', fontsize=12)
    ax1.set_title(f'(a) Liquid layer (l={liquid_layer})\nOverlap distribution', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # (b) 固态层重叠分布 P(Q_l^ab)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(analysis['Q_solid_offdiag'], bins=30, density=True, 
             alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=np.mean(analysis['Q_solid_offdiag']), color='blue', 
                linestyle='--', label=f'mean={np.mean(analysis["Q_solid_offdiag"]):.3f}')
    ax2.set_xlabel('$Q_l^{ab}$', fontsize=12)
    ax2.set_ylabel('$P(Q_l^{ab})$', fontsize=12)
    ax2.set_title(f'(b) Solid layer (l={solid_layer})\nOverlap distribution', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # (c) 两层分布比较
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(analysis['Q_liquid_offdiag'], bins=30, density=True, 
             alpha=0.5, color='blue', label=f'Liquid (l={liquid_layer})')
    ax3.hist(analysis['Q_solid_offdiag'], bins=30, density=True, 
             alpha=0.5, color='red', label=f'Solid (l={solid_layer})')
    ax3.set_xlabel('$Q_l^{ab}$', fontsize=12)
    ax3.set_ylabel('$P(Q_l^{ab})$', fontsize=12)
    ax3.set_title('(c) Comparison', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # (d) 液态层重叠矩阵热图
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(analysis['Q_liquid'], cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xlabel('Configuration index', fontsize=11)
    ax4.set_ylabel('Configuration index', fontsize=11)
    ax4.set_title(f'(d) Liquid layer (l={liquid_layer})\nOverlap matrix', fontsize=11)
    plt.colorbar(im4, ax=ax4, label='$Q_l^{ab}$')
    
    # (e) 固态层重叠矩阵热图
    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(analysis['Q_solid'], cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_xlabel('Configuration index', fontsize=11)
    ax5.set_ylabel('Configuration index', fontsize=11)
    ax5.set_title(f'(e) Solid layer (l={solid_layer})\nOverlap matrix', fontsize=11)
    plt.colorbar(im5, ax=ax5, label='$Q_l^{ab}$')
    
    # (f) 层次聚类树状图比较
    ax6 = fig.add_subplot(gs[1, 2])
    
    # 绘制固态层的树状图
    dendrogram(analysis['Z_solid'], ax=ax6, color_threshold=0.5,
               leaf_font_size=8, no_labels=True)
    ax6.set_xlabel('Configuration', fontsize=11)
    ax6.set_ylabel('Distance (1 - Q)', fontsize=11)
    ax6.set_title(f'(f) Solid layer (l={solid_layer})\nHierarchical clustering', fontsize=11)
    
    plt.suptitle(f'Design Space Structure (Figure 4)\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment4_figure4_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure 4 保存到: {fp}")
    return fp


def plot_dendrograms(analysis, output_dir, results):
    """绘制详细的层次聚类树状图"""
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 液态层树状图
    ax1 = axes[0]
    dendrogram(analysis['Z_liquid'], ax=ax1, color_threshold=0.3,
               leaf_font_size=8, no_labels=True)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Distance (1 - Q)', fontsize=11)
    ax1.set_title(f'Liquid layer (l={analysis["liquid_layer"]})\nComplete Linkage Clustering', fontsize=12)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # 固态层树状图
    ax2 = axes[1]
    dendrogram(analysis['Z_solid'], ax=ax2, color_threshold=0.3,
               leaf_font_size=8, no_labels=True)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Distance (1 - Q)', fontsize=11)
    ax2.set_title(f'Solid layer (l={analysis["solid_layer"]})\nComplete Linkage Clustering', fontsize=12)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Hierarchical Clustering Dendrograms\nln α = {results["ln_alpha"]:.2f}', fontsize=14)
    plt.tight_layout()
    
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fp = os.path.join(output_dir, f'experiment4_dendrograms_{ts}.png')
    plt.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"树状图保存到: {fp}")
    return fp


def main():
    print("=" * 70)
    print("实验四：设计空间结构 (Figure 4)")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              '..', 'reports', 'experiment4_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print("\n加载 MNIST 数据...")
    X_train, y_train, X_test, y_test = load_mnist_binary(train_size=200, test_size=40, seed=42)
    print(f"训练集: {X_train.shape}")
    
    # 参数设置
    L = 8
    N = 10
    beta = 1e5
    mc_steps = 2000
    n_replicas = 8
    save_interval = 100
    
    print(f"\n参数: L={L}, N={N}, β={beta:.0e}")
    print(f"MC steps={mc_steps}, 副本数={n_replicas}")
    print(f"ln α = {np.log(X_train.shape[0] / N):.2f}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3): warmup.mc_step_vectorized()
    print("完成")
    
    # 运行实验
    results = run_experiment4(X_train, y_train, N, L, beta, mc_steps, 
                              n_replicas, save_interval, seed=42)
    
    # 分析设计空间
    # 液态层选择中心层，固态层选择输入边界层
    liquid_layer = L // 2  # 中心层
    solid_layer = 1  # 输入边界层（第一个隐藏层）
    
    analysis = analyze_design_space(results, liquid_layer, solid_layer)
    
    # 绘图
    print("\n生成图表...")
    fp1 = plot_figure4(results, analysis, output_dir)
    fp2 = plot_dendrograms(analysis, output_dir, results)
    
    # 结果摘要
    print("\n" + "=" * 70)
    print("实验结果摘要")
    print("=" * 70)
    
    print(f"\n配置: ln α = {results['ln_alpha']:.2f}, L = {L}")
    print(f"保存配置数: {len(results['saved_configs'])}")
    
    print(f"\n液态层 (l={liquid_layer}):")
    print(f"  Q_l^ab 均值: {np.mean(analysis['Q_liquid_offdiag']):.4f}")
    print(f"  Q_l^ab 标准差: {np.std(analysis['Q_liquid_offdiag']):.4f}")
    
    print(f"\n固态层 (l={solid_layer}):")
    print(f"  Q_l^ab 均值: {np.mean(analysis['Q_solid_offdiag']):.4f}")
    print(f"  Q_l^ab 标准差: {np.std(analysis['Q_solid_offdiag']):.4f}")
    
    return results, analysis


if __name__ == "__main__":
    main()
