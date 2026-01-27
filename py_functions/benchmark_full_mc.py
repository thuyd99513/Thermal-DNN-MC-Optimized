"""
benchmark_full_mc.py - 完整 MC 步（S + J 更新）性能基准测试

本脚本对比完整 MC 步的三个版本的性能：
1. 原始串行实现
2. V2 版本（仅 S 优化）
3. V3 版本（S + J 优化）

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
from Network_optimized_v2 import NetworkOptimizedV2


# ============================================================================
# 原始串行实现
# ============================================================================

class NetworkSerialFull:
    """原始串行 S + J 更新实现"""
    
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
    
    # S 更新相关
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float64)
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float64)
        
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        return part_gap
    
    def update_S_hidden_serial(self, mu, l_s, n):
        """串行更新单个隐藏层自旋"""
        gap_before = self.part_gap_hidden_before_flip(mu, l_s, n)
        gap_after = self.part_gap_hidden_after_flip(mu, l_s, n)
        delta_E = self.calc_ener(gap_after) - self.calc_ener(gap_before)
        
        rand = np.random.random()
        if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
            self.S[mu, l_s, n] = -self.S[mu, l_s, n]
            self.H += delta_E
    
    def update_all_S_serial(self):
        """串行更新所有自旋"""
        for l_s in range(1, self.num_hidden_node_layers - 1):
            for mu in range(self.M):
                for n in range(self.N):
                    self.update_S_hidden_serial(mu, l_s, n)
    
    # J 更新相关
    def part_gap_hidden_shift(self, l, n2, J_row):
        gap = (J_row @ self.S[:, l, :].T / self.SQRT_N) * self.S[:, l + 1, n2]
        return gap
    
    def update_J_hidden_serial(self, l, n2, n1, x):
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
    
    def update_all_J_serial(self):
        """串行更新所有权重"""
        for l in range(self.num_hidden_bond_layers):
            for n2 in range(self.N):
                n1 = np.random.randint(0, self.N)
                x = np.random.normal()
                self.update_J_hidden_serial(l, n2, n1, x)
    
    def mc_step_serial(self):
        """完整 MC 步"""
        self.update_all_S_serial()
        self.update_all_J_serial()


# ============================================================================
# 性能测试函数
# ============================================================================

def benchmark_full_mc(configs, num_steps=5, num_warmup=2):
    """
    对不同配置进行完整 MC 步性能测试
    """
    results = []
    
    for M, N, L in configs:
        print(f"\n测试配置: M={M}, N={N}, L={L}")
        print("-" * 50)
        
        # 串行版本
        net_serial = NetworkSerialFull(M, N, L, seed=42)
        
        for _ in range(num_warmup):
            net_serial.mc_step_serial()
        
        start = time.time()
        for _ in range(num_steps):
            net_serial.mc_step_serial()
        serial_time = (time.time() - start) / num_steps * 1000
        
        print(f"  串行版本: {serial_time:.2f} ms/step")
        
        # V2 版本（仅 S 优化）
        net_v2 = NetworkOptimizedV2(M, N, L, seed=42)
        
        for _ in range(num_warmup):
            net_v2.update_all_S_vectorized()
        
        start = time.time()
        for _ in range(num_steps):
            net_v2.update_all_S_vectorized()
        v2_time = (time.time() - start) / num_steps * 1000
        
        print(f"  V2 (仅 S 优化): {v2_time:.2f} ms/step")
        
        # V3 版本（S + J 优化）
        net_v3 = NetworkOptimizedV3(M, N, L, seed=42)
        
        for _ in range(num_warmup):
            net_v3.mc_step_vectorized()
        
        start = time.time()
        for _ in range(num_steps):
            net_v3.mc_step_vectorized()
        v3_time = (time.time() - start) / num_steps * 1000
        
        print(f"  V3 (S + J 优化): {v3_time:.2f} ms/step")
        
        # 计算加速比
        speedup_v2 = serial_time / v2_time
        speedup_v3 = serial_time / v3_time
        
        print(f"  V2 加速比: {speedup_v2:.1f}x")
        print(f"  V3 加速比: {speedup_v3:.1f}x")
        
        results.append({
            'M': M,
            'N': N,
            'L': L,
            'serial_time': serial_time,
            'v2_time': v2_time,
            'v3_time': v3_time,
            'speedup_v2': speedup_v2,
            'speedup_v3': speedup_v3,
        })
    
    return results


def plot_full_mc_results(results, output_dir):
    """绘制完整 MC 步基准测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    configs = [f"M={r['M']}\nN={r['N']}\nL={r['L']}" for r in results]
    serial_times = [r['serial_time'] for r in results]
    v2_times = [r['v2_time'] for r in results]
    v3_times = [r['v3_time'] for r in results]
    speedups_v2 = [r['speedup_v2'] for r in results]
    speedups_v3 = [r['speedup_v3'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 执行时间对比
    ax1 = axes[0]
    x = np.arange(len(configs))
    width = 0.25
    
    bars1 = ax1.bar(x - width, serial_times, width, label='Serial', color='#ff7f0e')
    bars2 = ax1.bar(x, v2_times, width, label='V2 (S only)', color='#2ca02c')
    bars3 = ax1.bar(x + width, v3_times, width, label='V3 (S + J)', color='#1f77b4')
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms/step)')
    ax1.set_title('Full MC Step: Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=8)
    ax1.legend()
    ax1.set_yscale('log')
    
    # 图2: 加速比对比
    ax2 = axes[1]
    
    bars1 = ax2.bar(x - width/2, speedups_v2, width, label='V2 (S only)', color='#2ca02c')
    bars2 = ax2.bar(x + width/2, speedups_v3, width, label='V3 (S + J)', color='#1f77b4')
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Full MC Step: Speedup vs Serial')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=8)
    ax2.legend()
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    # 标注数值
    for bar, speedup in zip(bars1, speedups_v2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{speedup:.0f}x', ha='center', va='bottom', fontsize=8)
    for bar, speedup in zip(bars2, speedups_v3):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{speedup:.0f}x', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'full_mc_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n图表已保存到: {output_dir}/full_mc_benchmark.png")


def run_benchmark():
    """运行完整的性能基准测试"""
    print("=" * 60)
    print("完整 MC 步（S + J）性能基准测试")
    print("=" * 60)
    
    # 预热 JIT
    warmup_jit()
    
    # 测试配置
    configs = [
        (30, 3, 5),
        (60, 3, 5),
        (120, 3, 5),
        (30, 3, 10),
        (60, 3, 10),
        (120, 3, 10),
        (120, 5, 10),
        (240, 3, 10),
    ]
    
    results = benchmark_full_mc(configs, num_steps=5, num_warmup=2)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              '..', 'reports', 'full_mc_benchmark_results')
    plot_full_mc_results(results, output_dir)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("性能测试汇总")
    print("=" * 60)
    
    avg_speedup_v2 = np.mean([r['speedup_v2'] for r in results])
    avg_speedup_v3 = np.mean([r['speedup_v3'] for r in results])
    max_speedup_v3 = max([r['speedup_v3'] for r in results])
    
    print(f"  V2 (仅 S 优化) 平均加速比: {avg_speedup_v2:.1f}x")
    print(f"  V3 (S + J 优化) 平均加速比: {avg_speedup_v3:.1f}x")
    print(f"  V3 最大加速比: {max_speedup_v3:.1f}x")
    
    return results


if __name__ == "__main__":
    run_benchmark()
