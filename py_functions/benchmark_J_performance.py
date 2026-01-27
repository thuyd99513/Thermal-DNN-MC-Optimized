"""
benchmark_J_performance.py - J 更新性能基准测试

本脚本对比 J 更新的三个版本的性能：
1. 原始串行实现
2. V1 简单向量化实现
3. V3 完全向量化 + Numba JIT 实现

作者：Manus AI
日期：2026-01-27
"""

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Network_optimized_v3 import NetworkOptimizedV3, warmup_jit


# ============================================================================
# 原始串行 J 更新实现
# ============================================================================

class NetworkSerialJ:
    """原始串行 J 更新实现"""
    
    def __init__(self, M, N, L, N_in=784, N_out=2, beta=66.7, seed=42):
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
        self.EPS = 1e-6
        self.RAT = 0.1
        self.RESCALE_J = 1.0 / np.sqrt(1 + self.RAT ** 2)
        
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float64)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float64)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float64)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float64)
        self.J_in = np.random.randn(N, N_in).astype(np.float64)
        self.J_out = np.random.randn(N_out, N).astype(np.float64)
        
        # 归一化
        self._normalize_weights()
        
        self.H = 0.0
    
    def _normalize_weights(self):
        for n in range(self.N):
            norm = np.sqrt(np.sum(self.J_in[n] ** 2))
            self.J_in[n] *= np.sqrt(self.N_in) / norm
        for l in range(self.num_hidden_bond_layers):
            for n in range(self.N):
                norm = np.sqrt(np.sum(self.J_hidden[l, n] ** 2))
                self.J_hidden[l, n] *= np.sqrt(self.N) / norm
        for n in range(self.N_out):
            norm = np.sqrt(np.sum(self.J_out[n] ** 2))
            self.J_out[n] *= np.sqrt(self.N) / norm
    
    def soft_core_potential(self, h):
        return np.where(h < 0, h ** 2, 0.0)
    
    def calc_ener(self, r):
        return np.sum(self.soft_core_potential(r))
    
    def part_gap_hidden_shift(self, l, n2, J_row):
        gap = (J_row @ self.S[:, l, :].T / self.SQRT_N) * self.S[:, l + 1, n2]
        return gap
    
    def part_gap_in_shift(self, n2, J_row):
        gap = (J_row @ self.S_in.T / self.SQRT_N_IN) * self.S[:, 0, n2]
        return gap
    
    def part_gap_out_shift(self, n2, J_row):
        gap = (J_row @ self.S[:, -1, :].T / self.SQRT_N) * self.S_out[:, n2]
        return gap
    
    def update_J_hidden_serial(self, l, n2, n1, x):
        """串行更新单个隐藏层权重"""
        new_J_row = self.J_hidden[l, n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
        
        norm = np.sqrt(np.sum(new_J_row ** 2))
        new_J_row *= np.sqrt(self.N) / norm
        
        gap_before = self.part_gap_hidden_shift(l, n2, self.J_hidden[l, n2])
        gap_after = self.part_gap_hidden_shift(l, n2, new_J_row)
        delta_E = self.calc_ener(gap_after) - self.calc_ener(gap_before)
        
        rand = np.random.random()
        if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
            self.J_hidden[l, n2] = new_J_row
            self.H += delta_E
    
    def update_J_in_serial(self, n2, n1, x):
        """串行更新单个输入层权重"""
        new_J_row = self.J_in[n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
        
        norm = np.sqrt(np.sum(new_J_row ** 2))
        new_J_row *= np.sqrt(self.N_in) / norm
        
        gap_before = self.part_gap_in_shift(n2, self.J_in[n2])
        gap_after = self.part_gap_in_shift(n2, new_J_row)
        delta_E = self.calc_ener(gap_after) - self.calc_ener(gap_before)
        
        rand = np.random.random()
        if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
            self.J_in[n2] = new_J_row
            self.H += delta_E
    
    def update_J_out_serial(self, n2, n1, x):
        """串行更新单个输出层权重"""
        new_J_row = self.J_out[n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
        
        norm = np.sqrt(np.sum(new_J_row ** 2))
        new_J_row *= np.sqrt(self.N) / norm
        
        gap_before = self.part_gap_out_shift(n2, self.J_out[n2])
        gap_after = self.part_gap_out_shift(n2, new_J_row)
        delta_E = self.calc_ener(gap_after) - self.calc_ener(gap_before)
        
        rand = np.random.random()
        if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
            self.J_out[n2] = new_J_row
            self.H += delta_E
    
    def update_all_J_serial(self):
        """串行更新所有权重（每行更新一次）"""
        # 更新 J_in
        for n2 in range(self.N):
            n1 = np.random.randint(0, self.N_in)
            x = np.random.normal()
            self.update_J_in_serial(n2, n1, x)
        
        # 更新 J_hidden
        for l in range(self.num_hidden_bond_layers):
            for n2 in range(self.N):
                n1 = np.random.randint(0, self.N)
                x = np.random.normal()
                self.update_J_hidden_serial(l, n2, n1, x)
        
        # 更新 J_out
        for n2 in range(self.N_out):
            n1 = np.random.randint(0, self.N)
            x = np.random.normal()
            self.update_J_out_serial(n2, n1, x)


# ============================================================================
# 性能测试函数
# ============================================================================

def benchmark_J_update(configs, num_steps=5, num_warmup=2):
    """
    对不同配置进行 J 更新性能测试
    
    Args:
        configs: 配置列表，每个配置为 (M, N, L) 元组
        num_steps: 测试步数
        num_warmup: 预热步数
    
    Returns:
        results: 测试结果字典
    """
    results = []
    
    for M, N, L in configs:
        print(f"\n测试配置: M={M}, N={N}, L={L}")
        print("-" * 40)
        
        # 串行版本
        net_serial = NetworkSerialJ(M, N, L, seed=42)
        
        # 预热
        for _ in range(num_warmup):
            net_serial.update_all_J_serial()
        
        # 测试
        start = time.time()
        for _ in range(num_steps):
            net_serial.update_all_J_serial()
        serial_time = (time.time() - start) / num_steps * 1000  # ms
        
        print(f"  串行版本: {serial_time:.2f} ms/step")
        
        # V3 向量化版本
        net_v3 = NetworkOptimizedV3(M, N, L, seed=42)
        
        # 预热
        for _ in range(num_warmup):
            net_v3.update_all_J_vectorized()
        
        # 测试
        start = time.time()
        for _ in range(num_steps):
            net_v3.update_all_J_vectorized()
        v3_time = (time.time() - start) / num_steps * 1000  # ms
        
        print(f"  V3 向量化: {v3_time:.2f} ms/step")
        
        # 计算加速比
        speedup = serial_time / v3_time
        print(f"  加速比: {speedup:.1f}x")
        
        results.append({
            'M': M,
            'N': N,
            'L': L,
            'serial_time': serial_time,
            'v3_time': v3_time,
            'speedup': speedup,
            'total_weights': N * 784 + (L - 2) * N * N + 2 * N  # 近似权重数
        })
    
    return results


def plot_benchmark_results(results, output_dir):
    """绘制基准测试结果图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Noto Sans CJK SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取数据
    configs = [f"M={r['M']}\nN={r['N']}\nL={r['L']}" for r in results]
    serial_times = [r['serial_time'] for r in results]
    v3_times = [r['v3_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 图1: 执行时间对比
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, serial_times, width, label='Serial', color='#ff7f0e')
    bars2 = ax1.bar(x + width/2, v3_times, width, label='V3 Vectorized', color='#1f77b4')
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms/step)')
    ax1.set_title('J Update: Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.legend()
    ax1.set_yscale('log')
    
    # 图2: 加速比
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(np.array(speedups) / max(speedups))
    bars = ax2.bar(x, speedups, color=colors)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('J Update: Speedup (V3 vs Serial)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # 在柱子上标注数值
    for bar, speedup in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9)
    
    # 图3: 加速比 vs 网络规模
    ax3 = axes[2]
    total_weights = [r['total_weights'] for r in results]
    
    scatter = ax3.scatter(total_weights, speedups, c=speedups, cmap='RdYlGn', 
                          s=100, edgecolors='black', linewidth=1)
    
    # 添加趋势线
    z = np.polyfit(total_weights, speedups, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(total_weights), max(total_weights), 100)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, label='Trend')
    
    ax3.set_xlabel('Total Weights (approx)')
    ax3.set_ylabel('Speedup (x)')
    ax3.set_title('Speedup vs Network Scale')
    ax3.legend()
    
    plt.colorbar(scatter, ax=ax3, label='Speedup')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'J_benchmark_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存到: {output_dir}/J_benchmark_comparison.png")


def run_benchmark():
    """运行完整的性能基准测试"""
    print("=" * 60)
    print("J 更新性能基准测试")
    print("=" * 60)
    
    # 预热 JIT
    warmup_jit()
    
    # 测试配置
    configs = [
        (30, 3, 5),    # 小规模
        (60, 3, 5),    # 小规模 2
        (120, 3, 5),   # 小规模 3
        (30, 3, 10),   # 中等规模
        (60, 3, 10),   # 中等规模 2
        (120, 3, 10),  # 默认配置
        (120, 5, 10),  # 大规模 1
        (240, 3, 10),  # 大规模 2
    ]
    
    # 运行测试
    results = benchmark_J_update(configs, num_steps=5, num_warmup=2)
    
    # 绘制图表
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              '..', 'reports', 'J_benchmark_results')
    plot_benchmark_results(results, output_dir)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("性能测试汇总")
    print("=" * 60)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    max_speedup = max([r['speedup'] for r in results])
    min_speedup = min([r['speedup'] for r in results])
    
    print(f"  平均加速比: {avg_speedup:.1f}x")
    print(f"  最大加速比: {max_speedup:.1f}x")
    print(f"  最小加速比: {min_speedup:.1f}x")
    
    return results


if __name__ == "__main__":
    run_benchmark()
