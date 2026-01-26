"""
benchmark_performance.py - 性能基准测试脚本

对比 Network.py (原始实现) 和 Network_optimized.py (优化实现) 的性能差异。

测试内容：
1. 在不同网络规模 (M, N, L) 下测试
2. 记录每个 MC 步的耗时
3. 生成性能对比图表

作者：Manus AI
日期：2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
import time
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
OUTPUT_DIR = '/home/ubuntu/benchmark_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 模拟原始实现的核心函数 (从 Network.py 提取关键逻辑)
# ============================================================================

def soft_core_potential(h):
    """软核势能函数"""
    return np.heaviside(-h, 1.0) * np.power(h, 2)

def calc_ener(r):
    """计算能量"""
    return soft_core_potential(r).sum()


class NetworkOriginal:
    """
    原始实现的简化版本 - 串行逐个更新自旋
    """
    
    def __init__(self, M, N, L, N_in=784, N_out=2, beta=66.7):
        self.M = M
        self.N = N
        self.L = L
        self.N_in = N_in
        self.N_out = N_out
        self.beta = beta
        
        self.num_hidden_node_layers = L - 1
        self.num_hidden_bond_layers = L - 2
        
        self.SQRT_N = np.sqrt(N)
        self.SQRT_N_IN = np.sqrt(N_in)
        self.EPS = 0.000001
        
        # 初始化网络状态
        np.random.seed(42)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.int8)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        """计算中间层自旋翻转前的 part_gap"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
        """计算中间层自旋翻转后的 part_gap"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        return part_gap
    
    def update_S_serial(self, num_updates):
        """串行更新自旋 (原始方法)"""
        for _ in range(num_updates):
            # 随机选择一个自旋
            mu = np.random.randint(0, self.M)
            l_s = np.random.randint(1, self.num_hidden_node_layers - 1)
            n = np.random.randint(0, self.N)
            
            # 计算能量差
            part_gap_before = self.part_gap_hidden_before_flip(mu, l_s, n)
            part_gap_after = self.part_gap_hidden_after_flip(mu, l_s, n)
            
            delta_E = calc_ener(part_gap_after) - calc_ener(part_gap_before)
            
            # Metropolis 接受/拒绝
            if delta_E < self.EPS:
                self.S[mu, l_s, n] = -self.S[mu, l_s, n]
                self.H += delta_E
            elif np.random.random() < np.exp(-delta_E * self.beta):
                self.S[mu, l_s, n] = -self.S[mu, l_s, n]
                self.H += delta_E


class NetworkOptimizedSimple:
    """
    优化实现的简化版本 - 按层并行向量化更新
    """
    
    def __init__(self, M, N, L, N_in=784, N_out=2, beta=66.7):
        self.M = M
        self.N = N
        self.L = L
        self.N_in = N_in
        self.N_out = N_out
        self.beta = beta
        
        self.num_hidden_node_layers = L - 1
        self.num_hidden_bond_layers = L - 2
        
        self.SQRT_N = np.sqrt(N)
        self.SQRT_N_IN = np.sqrt(N_in)
        self.EPS = 0.000001
        
        # 初始化网络状态 (使用相同的随机种子)
        np.random.seed(42)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.int8)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    def part_gap_hidden_before_flip_layer(self, l_s):
        """向量化计算第 l_s 层所有自旋翻转前的 part_gap"""
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * self.S[:, l_s, n]
        
        for mu in range(M):
            J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
            gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
            part_gap[mu, :, 1:] = gap_next
        
        return part_gap
    
    def part_gap_hidden_after_flip_layer(self, l_s):
        """向量化计算第 l_s 层所有自旋翻转后的 part_gap"""
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            S_flipped_n = -self.S[:, l_s, n]
            
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * S_flipped_n
            
            for mu in range(M):
                S_layer_flipped = self.S[mu, l_s, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_hidden_next = self.J_hidden[l_s, :, :] @ S_layer_flipped
                gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
                part_gap[mu, n, 1:] = gap_next
        
        return part_gap
    
    def update_S_middle_layer_vectorized(self, l_s):
        """向量化更新中间层的所有自旋"""
        M, N = self.M, self.N
        
        part_gap_before = self.part_gap_hidden_before_flip_layer(l_s)
        part_gap_after = self.part_gap_hidden_after_flip_layer(l_s)
        
        E_before = soft_core_potential(part_gap_before).sum(axis=2)
        E_after = soft_core_potential(part_gap_after).sum(axis=2)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((M, N))
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
        
        self.S[:, l_s, :] = np.where(accept_mask, -self.S[:, l_s, :], self.S[:, l_s, :])
        self.H += np.sum(delta_E * accept_mask)
    
    def update_all_S_vectorized(self):
        """向量化更新所有隐藏层自旋"""
        for l_s in range(1, self.num_hidden_node_layers - 1):
            self.update_S_middle_layer_vectorized(l_s)


def benchmark_single_config(M, N, L, num_mc_steps=10, num_updates_per_step=None):
    """
    对单个配置进行基准测试
    
    Args:
        M: 训练样本数
        N: 每层隐藏节点数
        L: 网络层数
        num_mc_steps: MC 步数
        num_updates_per_step: 每步更新次数 (原始方法)，默认为 M * N * (L-3)
    
    Returns:
        dict: 包含测试结果的字典
    """
    if num_updates_per_step is None:
        num_updates_per_step = M * N * (L - 3)  # 中间层总自旋数
    
    print(f"\n配置: M={M}, N={N}, L={L}")
    print(f"  中间层数: {L-3}, 每层自旋数: {M}×{N}={M*N}")
    print(f"  每 MC 步更新次数: {num_updates_per_step}")
    
    # 测试原始实现
    print("  测试原始实现 (串行)...")
    net_original = NetworkOriginal(M, N, L)
    
    original_times = []
    for step in range(num_mc_steps):
        start = time.time()
        net_original.update_S_serial(num_updates_per_step)
        end = time.time()
        original_times.append(end - start)
    
    original_avg = np.mean(original_times)
    original_std = np.std(original_times)
    print(f"    平均耗时: {original_avg:.4f}s ± {original_std:.4f}s")
    
    # 测试优化实现
    print("  测试优化实现 (向量化)...")
    net_optimized = NetworkOptimizedSimple(M, N, L)
    
    optimized_times = []
    for step in range(num_mc_steps):
        start = time.time()
        net_optimized.update_all_S_vectorized()
        end = time.time()
        optimized_times.append(end - start)
    
    optimized_avg = np.mean(optimized_times)
    optimized_std = np.std(optimized_times)
    print(f"    平均耗时: {optimized_avg:.4f}s ± {optimized_std:.4f}s")
    
    # 计算加速比
    speedup = original_avg / optimized_avg if optimized_avg > 0 else 0
    print(f"  加速比: {speedup:.2f}x")
    
    return {
        'M': M,
        'N': N,
        'L': L,
        'num_updates_per_step': num_updates_per_step,
        'original_times': original_times,
        'original_avg': original_avg,
        'original_std': original_std,
        'optimized_times': optimized_times,
        'optimized_avg': optimized_avg,
        'optimized_std': optimized_std,
        'speedup': speedup
    }


def run_benchmark_suite():
    """运行完整的基准测试套件"""
    print("=" * 70)
    print("Thermal-DNN-MC-Optimized 性能基准测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 测试配置
    # 配置格式: (M, N, L)
    test_configs = [
        # 小规模测试
        (30, 3, 5),
        (60, 3, 5),
        (120, 3, 5),
        
        # 中等规模测试
        (30, 3, 10),
        (60, 3, 10),
        (120, 3, 10),
        
        # 较大规模测试
        (120, 5, 10),
        (240, 3, 10),
        
        # 默认配置 (论文配置)
        (120, 3, 10),  # L=10, M=120, N=3
    ]
    
    # 去重
    test_configs = list(set(test_configs))
    test_configs.sort(key=lambda x: (x[2], x[0], x[1]))
    
    results = []
    
    for M, N, L in test_configs:
        try:
            result = benchmark_single_config(M, N, L, num_mc_steps=5)
            results.append(result)
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    return results


def generate_visualizations(results):
    """生成可视化图表"""
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)
    
    # 准备数据
    configs = [f"M={r['M']}\nN={r['N']}\nL={r['L']}" for r in results]
    original_times = [r['original_avg'] for r in results]
    optimized_times = [r['optimized_avg'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # 图1: 耗时对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, original_times, width, label='Original (Serial)', color='coral', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, optimized_times, width, label='Optimized (Vectorized)', color='steelblue', alpha=0.8)
    
    axes[0].set_xlabel('Network Configuration', fontsize=12)
    axes[0].set_ylabel('Time per MC Step (seconds)', fontsize=12)
    axes[0].set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, fontsize=9)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 图2: 加速比
    colors = ['green' if s > 5 else 'orange' if s > 2 else 'red' for s in speedups]
    bars = axes[1].bar(x, speedups, color=colors, alpha=0.8, edgecolor='black')
    
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline (1x)')
    axes[1].axhline(y=5, color='green', linestyle=':', alpha=0.7, label='5x Speedup')
    axes[1].axhline(y=10, color='blue', linestyle=':', alpha=0.7, label='10x Speedup')
    
    # 添加数值标签
    for bar, speedup in zip(bars, speedups):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    axes[1].set_xlabel('Network Configuration', fontsize=12)
    axes[1].set_ylabel('Speedup Factor', fontsize=12)
    axes[1].set_title('Speedup: Optimized vs Original', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, fontsize=9)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 图3: 网络规模 vs 加速比
    total_spins = [r['M'] * r['N'] * (r['L'] - 1) for r in results]
    
    scatter = axes[2].scatter(total_spins, speedups, s=150, c=speedups, cmap='RdYlGn', 
                              edgecolor='black', linewidth=1, alpha=0.8)
    
    # 添加趋势线
    z = np.polyfit(total_spins, speedups, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(total_spins), max(total_spins), 100)
    axes[2].plot(x_line, p(x_line), 'r--', alpha=0.5, label='Trend')
    
    axes[2].set_xlabel('Total Hidden Spins (M × N × (L-1))', fontsize=12)
    axes[2].set_ylabel('Speedup Factor', fontsize=12)
    axes[2].set_title('Speedup vs Network Size', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.colorbar(scatter, ax=axes[2], label='Speedup')
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'benchmark_comparison.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")
    
    # 图4: 详细时间分布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 选择几个代表性配置
    selected_indices = [0, len(results)//3, 2*len(results)//3, -1]
    selected_indices = [i % len(results) for i in selected_indices]
    
    for idx, ax_idx in enumerate(range(4)):
        if idx >= len(results):
            break
        
        r = results[selected_indices[idx]]
        ax = axes[ax_idx // 2, ax_idx % 2]
        
        steps = range(1, len(r['original_times']) + 1)
        ax.plot(steps, r['original_times'], 'o-', color='coral', label='Original', linewidth=2, markersize=8)
        ax.plot(steps, r['optimized_times'], 's-', color='steelblue', label='Optimized', linewidth=2, markersize=8)
        
        ax.axhline(y=r['original_avg'], color='coral', linestyle='--', alpha=0.5)
        ax.axhline(y=r['optimized_avg'], color='steelblue', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('MC Step', fontsize=11)
        ax.set_ylabel('Time (seconds)', fontsize=11)
        ax.set_title(f"M={r['M']}, N={r['N']}, L={r['L']} (Speedup: {r['speedup']:.1f}x)", 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Time per MC Step Distribution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'benchmark_time_distribution.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")
    
    # 图5: 汇总图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_speedup = np.mean(speedups)
    max_speedup = max(speedups)
    min_speedup = min(speedups)
    
    # 创建汇总柱状图
    categories = ['Min Speedup', 'Avg Speedup', 'Max Speedup']
    values = [min_speedup, avg_speedup, max_speedup]
    colors = ['orange', 'steelblue', 'green']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'{val:.2f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Baseline')
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Performance Benchmark Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息文本框
    stats_text = f"Configurations Tested: {len(results)}\n"
    stats_text += f"Average Speedup: {avg_speedup:.2f}x\n"
    stats_text += f"Range: {min_speedup:.2f}x - {max_speedup:.2f}x"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'benchmark_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")


def save_results(results):
    """保存测试结果"""
    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    
    # 转换 numpy 类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    serializable_results = []
    for r in results:
        sr = {k: convert_to_serializable(v) for k, v in r.items()}
        serializable_results.append(sr)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n结果已保存: {json_path}")


def main():
    """主函数"""
    # 运行基准测试
    results = run_benchmark_suite()
    
    if not results:
        print("没有测试结果，退出")
        return
    
    # 生成可视化
    generate_visualizations(results)
    
    # 保存结果
    save_results(results)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("基准测试完成!")
    print("=" * 70)
    
    speedups = [r['speedup'] for r in results]
    print(f"\n测试配置数: {len(results)}")
    print(f"平均加速比: {np.mean(speedups):.2f}x")
    print(f"最小加速比: {min(speedups):.2f}x")
    print(f"最大加速比: {max(speedups):.2f}x")
    
    print(f"\n结果保存目录: {OUTPUT_DIR}")
    print("  - benchmark_comparison.png: 性能对比图")
    print("  - benchmark_time_distribution.png: 时间分布图")
    print("  - benchmark_summary.png: 汇总图")
    print("  - benchmark_results.json: 原始数据")


if __name__ == "__main__":
    main()
