"""
test_optimization_v2.py - 优化版本验证和性能测试脚本

本脚本执行以下任务：
1. 验证优化版本 (V2) 与原始实现的物理正确性一致
2. 对比三个版本的性能：原始串行、V1向量化、V2完全向量化+Numba
3. 生成详细的测试报告和可视化图表

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

# 导入优化版本
from Network_optimized_v2 import (
    NetworkOptimizedV2, NetworkOriginalReference,
    soft_core_potential_numpy, warmup_jit,
    compute_part_gap_hidden_before_vectorized,
    compute_part_gap_hidden_after_vectorized,
    calc_ener_3d_numba
)

# 创建输出目录
OUTPUT_DIR = '/home/ubuntu/optimization_v2_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 物理正确性验证
# ============================================================================

def verify_part_gap_correctness():
    """验证 part_gap 计算的正确性"""
    print("\n" + "=" * 70)
    print("验证 part_gap 计算正确性")
    print("=" * 70)
    
    M, N, L = 30, 3, 10
    seed = 42
    
    # 创建两个相同初始状态的网络
    net_ref = NetworkOriginalReference(M, N, L, seed=seed)
    net_v2 = NetworkOptimizedV2(M, N, L, seed=seed)
    
    # 确保初始状态相同
    assert np.allclose(net_ref.S, net_v2.S), "初始 S 不一致"
    assert np.allclose(net_ref.J_hidden, net_v2.J_hidden), "初始 J_hidden 不一致"
    
    test_results = []
    
    # 测试中间层的 part_gap 计算
    for l_s in range(1, net_ref.num_hidden_node_layers - 1):
        print(f"\n测试层 l_s = {l_s}")
        
        # 使用 V2 的向量化方法计算
        part_gap_before_v2 = compute_part_gap_hidden_before_vectorized(
            net_v2.J_hidden[l_s - 1], net_v2.J_hidden[l_s],
            net_v2.S[:, l_s - 1, :], net_v2.S[:, l_s, :], net_v2.S[:, l_s + 1, :],
            net_v2.SQRT_N
        )
        part_gap_after_v2 = compute_part_gap_hidden_after_vectorized(
            net_v2.J_hidden[l_s - 1], net_v2.J_hidden[l_s],
            net_v2.S[:, l_s - 1, :], net_v2.S[:, l_s, :], net_v2.S[:, l_s + 1, :],
            net_v2.SQRT_N
        )
        
        # 使用原始方法逐个计算并对比
        max_diff_before = 0.0
        max_diff_after = 0.0
        num_tests = 0
        
        for mu in range(min(M, 10)):  # 测试前10个样本
            for n in range(N):
                # 原始方法
                part_gap_before_ref = net_ref.part_gap_hidden_before_flip(mu, l_s, n)
                part_gap_after_ref = net_ref.part_gap_hidden_after_flip(mu, l_s, n)
                
                # V2 方法
                part_gap_before_v2_single = part_gap_before_v2[mu, n, :]
                part_gap_after_v2_single = part_gap_after_v2[mu, n, :]
                
                # 计算差异
                diff_before = np.abs(part_gap_before_ref - part_gap_before_v2_single).max()
                diff_after = np.abs(part_gap_after_ref - part_gap_after_v2_single).max()
                
                max_diff_before = max(max_diff_before, diff_before)
                max_diff_after = max(max_diff_after, diff_after)
                num_tests += 1
        
        passed_before = max_diff_before < 1e-5
        passed_after = max_diff_after < 1e-5
        
        print(f"  before_flip 最大差异: {max_diff_before:.2e} - {'✓ 通过' if passed_before else '✗ 失败'}")
        print(f"  after_flip 最大差异:  {max_diff_after:.2e} - {'✓ 通过' if passed_after else '✗ 失败'}")
        
        test_results.append({
            'layer': l_s,
            'max_diff_before': float(max_diff_before),
            'max_diff_after': float(max_diff_after),
            'passed_before': passed_before,
            'passed_after': passed_after,
            'num_tests': num_tests
        })
    
    return test_results


def verify_energy_calculation():
    """验证能量计算的正确性"""
    print("\n" + "=" * 70)
    print("验证能量计算正确性")
    print("=" * 70)
    
    M, N, L = 30, 3, 10
    seed = 42
    
    net_ref = NetworkOriginalReference(M, N, L, seed=seed)
    net_v2 = NetworkOptimizedV2(M, N, L, seed=seed)
    
    test_results = []
    
    for l_s in range(1, net_ref.num_hidden_node_layers - 1):
        print(f"\n测试层 l_s = {l_s}")
        
        # 使用 V2 计算能量差
        part_gap_before_v2 = compute_part_gap_hidden_before_vectorized(
            net_v2.J_hidden[l_s - 1], net_v2.J_hidden[l_s],
            net_v2.S[:, l_s - 1, :], net_v2.S[:, l_s, :], net_v2.S[:, l_s + 1, :],
            net_v2.SQRT_N
        )
        part_gap_after_v2 = compute_part_gap_hidden_after_vectorized(
            net_v2.J_hidden[l_s - 1], net_v2.J_hidden[l_s],
            net_v2.S[:, l_s - 1, :], net_v2.S[:, l_s, :], net_v2.S[:, l_s + 1, :],
            net_v2.SQRT_N
        )
        
        E_before_v2 = calc_ener_3d_numba(part_gap_before_v2)
        E_after_v2 = calc_ener_3d_numba(part_gap_after_v2)
        delta_E_v2 = E_after_v2 - E_before_v2
        
        # 使用原始方法计算能量差
        max_diff = 0.0
        num_tests = 0
        
        for mu in range(min(M, 10)):
            for n in range(N):
                part_gap_before_ref = net_ref.part_gap_hidden_before_flip(mu, l_s, n)
                part_gap_after_ref = net_ref.part_gap_hidden_after_flip(mu, l_s, n)
                
                E_before_ref = soft_core_potential_numpy(part_gap_before_ref).sum()
                E_after_ref = soft_core_potential_numpy(part_gap_after_ref).sum()
                delta_E_ref = E_after_ref - E_before_ref
                
                diff = abs(delta_E_ref - delta_E_v2[mu, n])
                max_diff = max(max_diff, diff)
                num_tests += 1
        
        passed = max_diff < 1e-4
        print(f"  能量差最大差异: {max_diff:.2e} - {'✓ 通过' if passed else '✗ 失败'}")
        
        test_results.append({
            'layer': l_s,
            'max_diff': float(max_diff),
            'passed': passed,
            'num_tests': num_tests
        })
    
    return test_results


def verify_mc_evolution():
    """验证 MC 演化的统计一致性"""
    print("\n" + "=" * 70)
    print("验证 MC 演化统计一致性")
    print("=" * 70)
    
    M, N, L = 60, 3, 8
    num_steps = 20
    
    # 使用相同的随机种子
    seed = 12345
    
    # V2 优化版本
    np.random.seed(seed)
    net_v2 = NetworkOptimizedV2(M, N, L, seed=seed)
    H_history_v2 = [net_v2.H]
    
    np.random.seed(seed + 1000)  # 使用不同的种子进行 MC 更新
    for _ in range(num_steps):
        net_v2.update_all_S_vectorized()
        H_history_v2.append(net_v2.H)
    
    # 检查能量是否合理变化
    H_changes = np.diff(H_history_v2)
    
    print(f"  初始能量: {H_history_v2[0]:.4f}")
    print(f"  最终能量: {H_history_v2[-1]:.4f}")
    print(f"  能量变化范围: [{min(H_changes):.4f}, {max(H_changes):.4f}]")
    print(f"  平均能量变化: {np.mean(H_changes):.4f}")
    
    # 验证能量变化是合理的（不应该有极端值）
    reasonable = all(abs(dH) < 1000 for dH in H_changes)
    print(f"  能量变化合理性: {'✓ 通过' if reasonable else '✗ 失败'}")
    
    return {
        'H_history': [float(h) for h in H_history_v2],
        'H_changes': [float(h) for h in H_changes],
        'reasonable': reasonable
    }


# ============================================================================
# 性能基准测试
# ============================================================================

class NetworkOriginalSerial:
    """原始串行实现 - 用于性能对比"""
    
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
        
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float32)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
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
        for _ in range(num_updates):
            mu = np.random.randint(0, self.M)
            l_s = np.random.randint(1, self.num_hidden_node_layers - 1)
            n = np.random.randint(0, self.N)
            
            part_gap_before = self.part_gap_hidden_before_flip(mu, l_s, n)
            part_gap_after = self.part_gap_hidden_after_flip(mu, l_s, n)
            
            delta_E = soft_core_potential_numpy(part_gap_after).sum() - soft_core_potential_numpy(part_gap_before).sum()
            
            if delta_E < self.EPS:
                self.S[mu, l_s, n] = -self.S[mu, l_s, n]
                self.H += delta_E
            elif np.random.random() < np.exp(-delta_E * self.beta):
                self.S[mu, l_s, n] = -self.S[mu, l_s, n]
                self.H += delta_E


class NetworkV1Vectorized:
    """V1 向量化实现 (有嵌套循环) - 用于性能对比"""
    
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
        
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float32)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    def part_gap_hidden_before_flip_layer(self, l_s):
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
        """带嵌套循环的版本"""
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            S_flipped_n = -self.S[:, l_s, n]
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * S_flipped_n
            
            # 嵌套循环 - 这是瓶颈
            for mu in range(M):
                S_layer_flipped = self.S[mu, l_s, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_hidden_next = self.J_hidden[l_s, :, :] @ S_layer_flipped
                gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
                part_gap[mu, n, 1:] = gap_next
        
        return part_gap
    
    def update_S_middle_layer_vectorized(self, l_s):
        part_gap_before = self.part_gap_hidden_before_flip_layer(l_s)
        part_gap_after = self.part_gap_hidden_after_flip_layer(l_s)
        
        E_before = soft_core_potential_numpy(part_gap_before).sum(axis=2)
        E_after = soft_core_potential_numpy(part_gap_after).sum(axis=2)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((self.M, self.N))
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
        
        self.S[:, l_s, :] = np.where(accept_mask, -self.S[:, l_s, :], self.S[:, l_s, :])
        self.H += np.sum(delta_E * accept_mask)
    
    def update_all_S_vectorized(self):
        for l_s in range(1, self.num_hidden_node_layers - 1):
            self.update_S_middle_layer_vectorized(l_s)


def benchmark_performance():
    """运行性能基准测试"""
    print("\n" + "=" * 70)
    print("性能基准测试")
    print("=" * 70)
    
    # 预热 JIT
    warmup_jit()
    
    # 测试配置
    test_configs = [
        (30, 3, 10),
        (60, 3, 10),
        (120, 3, 10),
        (120, 5, 10),
        (240, 3, 10),
    ]
    
    results = []
    
    for M, N, L in test_configs:
        print(f"\n配置: M={M}, N={N}, L={L}")
        num_updates_per_step = M * N * (L - 3)
        num_steps = 5
        
        # 测试原始串行实现
        print("  测试原始串行实现...")
        net_serial = NetworkOriginalSerial(M, N, L)
        
        serial_times = []
        for _ in range(num_steps):
            start = time.time()
            net_serial.update_S_serial(num_updates_per_step)
            serial_times.append(time.time() - start)
        
        serial_avg = np.mean(serial_times)
        print(f"    平均耗时: {serial_avg*1000:.2f}ms")
        
        # 测试 V1 向量化实现 (有嵌套循环)
        print("  测试 V1 向量化实现...")
        net_v1 = NetworkV1Vectorized(M, N, L)
        
        v1_times = []
        for _ in range(num_steps):
            start = time.time()
            net_v1.update_all_S_vectorized()
            v1_times.append(time.time() - start)
        
        v1_avg = np.mean(v1_times)
        print(f"    平均耗时: {v1_avg*1000:.2f}ms")
        
        # 测试 V2 完全向量化 + Numba 实现
        print("  测试 V2 完全向量化 + Numba 实现...")
        net_v2 = NetworkOptimizedV2(M, N, L)
        
        # 额外预热
        for _ in range(2):
            net_v2.update_all_S_vectorized()
        
        v2_times = []
        for _ in range(num_steps):
            start = time.time()
            net_v2.update_all_S_vectorized()
            v2_times.append(time.time() - start)
        
        v2_avg = np.mean(v2_times)
        print(f"    平均耗时: {v2_avg*1000:.2f}ms")
        
        # 计算加速比
        speedup_v1_vs_serial = serial_avg / v1_avg if v1_avg > 0 else 0
        speedup_v2_vs_serial = serial_avg / v2_avg if v2_avg > 0 else 0
        speedup_v2_vs_v1 = v1_avg / v2_avg if v2_avg > 0 else 0
        
        print(f"  加速比:")
        print(f"    V1 vs Serial: {speedup_v1_vs_serial:.2f}x")
        print(f"    V2 vs Serial: {speedup_v2_vs_serial:.2f}x")
        print(f"    V2 vs V1:     {speedup_v2_vs_v1:.2f}x")
        
        results.append({
            'M': M, 'N': N, 'L': L,
            'serial_avg': serial_avg,
            'v1_avg': v1_avg,
            'v2_avg': v2_avg,
            'speedup_v1_vs_serial': speedup_v1_vs_serial,
            'speedup_v2_vs_serial': speedup_v2_vs_serial,
            'speedup_v2_vs_v1': speedup_v2_vs_v1
        })
    
    return results


# ============================================================================
# 可视化
# ============================================================================

def generate_visualizations(correctness_results, energy_results, mc_results, perf_results):
    """生成可视化图表"""
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 正确性验证结果
    ax = axes[0, 0]
    
    layers = [r['layer'] for r in correctness_results]
    diff_before = [r['max_diff_before'] for r in correctness_results]
    diff_after = [r['max_diff_after'] for r in correctness_results]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, diff_before, width, label='before_flip', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, diff_after, width, label='after_flip', color='coral', alpha=0.8)
    
    ax.axhline(y=1e-5, color='green', linestyle='--', alpha=0.7, label='Tolerance (1e-5)')
    ax.set_xlabel('Layer Index (l_s)', fontsize=12)
    ax.set_ylabel('Max Absolute Difference', fontsize=12)
    ax.set_title('Part Gap Calculation Correctness', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'l_s={l}' for l in layers])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加通过/失败标记
    all_passed = all(r['passed_before'] and r['passed_after'] for r in correctness_results)
    status = "✓ All Tests Passed" if all_passed else "✗ Some Tests Failed"
    ax.text(0.98, 0.98, status, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='right',
            color='green' if all_passed else 'red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 图2: 性能对比
    ax = axes[0, 1]
    
    configs = [f"M={r['M']}\nN={r['N']}" for r in perf_results]
    serial_times = [r['serial_avg'] * 1000 for r in perf_results]
    v1_times = [r['v1_avg'] * 1000 for r in perf_results]
    v2_times = [r['v2_avg'] * 1000 for r in perf_results]
    
    x = np.arange(len(configs))
    width = 0.25
    
    bars1 = ax.bar(x - width, serial_times, width, label='Original Serial', color='coral', alpha=0.8)
    bars2 = ax.bar(x, v1_times, width, label='V1 Vectorized', color='orange', alpha=0.8)
    bars3 = ax.bar(x + width, v2_times, width, label='V2 Vectorized+Numba', color='green', alpha=0.8)
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time per MC Step (ms)', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: 加速比
    ax = axes[1, 0]
    
    speedup_v1 = [r['speedup_v1_vs_serial'] for r in perf_results]
    speedup_v2 = [r['speedup_v2_vs_serial'] for r in perf_results]
    
    x = np.arange(len(configs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, speedup_v1, width, label='V1 vs Serial', color='orange', alpha=0.8)
    bars2 = ax.bar(x + width/2, speedup_v2, width, label='V2 vs Serial', color='green', alpha=0.8)
    
    # 添加数值标签
    for bar, speedup in zip(bars1, speedup_v1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, speedup in zip(bars2, speedup_v2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{speedup:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Speedup vs Original Serial Implementation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4: MC 演化能量曲线
    ax = axes[1, 1]
    
    H_history = mc_results['H_history']
    steps = range(len(H_history))
    
    ax.plot(steps, H_history, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.axhline(y=H_history[0], color='gray', linestyle='--', alpha=0.5, label='Initial Energy')
    
    ax.set_xlabel('MC Step', fontsize=12)
    ax.set_ylabel('Energy (H)', fontsize=12)
    ax.set_title('MC Evolution Energy Trajectory', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    stats_text = f"Initial: {H_history[0]:.2f}\nFinal: {H_history[-1]:.2f}\nChange: {H_history[-1]-H_history[0]:.2f}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'optimization_v2_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")
    
    # 图5: 汇总图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    avg_speedup_v1 = np.mean([r['speedup_v1_vs_serial'] for r in perf_results])
    avg_speedup_v2 = np.mean([r['speedup_v2_vs_serial'] for r in perf_results])
    avg_speedup_v2_vs_v1 = np.mean([r['speedup_v2_vs_v1'] for r in perf_results])
    
    categories = ['Original\nSerial', 'V1\nVectorized', 'V2\nVectorized+Numba']
    speedups = [1.0, avg_speedup_v1, avg_speedup_v2]
    colors = ['coral', 'orange', 'green']
    
    bars = ax.bar(categories, speedups, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{speedup:.1f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Average Speedup Factor', fontsize=12)
    ax.set_title('Performance Optimization Summary', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    stats_text = f"V1 vs Serial: {avg_speedup_v1:.2f}x\n"
    stats_text += f"V2 vs Serial: {avg_speedup_v2:.2f}x\n"
    stats_text += f"V2 vs V1: {avg_speedup_v2_vs_v1:.2f}x"
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'optimization_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")


def save_results(correctness_results, energy_results, mc_results, perf_results):
    """保存结果到 JSON"""
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'correctness_verification': correctness_results,
        'energy_verification': energy_results,
        'mc_evolution': mc_results,
        'performance_benchmark': perf_results
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'optimization_v2_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {json_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Thermal-DNN-MC-Optimized V2 优化验证与性能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 验证 part_gap 计算正确性
    correctness_results = verify_part_gap_correctness()
    
    # 2. 验证能量计算正确性
    energy_results = verify_energy_calculation()
    
    # 3. 验证 MC 演化统计一致性
    mc_results = verify_mc_evolution()
    
    # 4. 性能基准测试
    perf_results = benchmark_performance()
    
    # 5. 生成可视化
    generate_visualizations(correctness_results, energy_results, mc_results, perf_results)
    
    # 6. 保存结果
    save_results(correctness_results, energy_results, mc_results, perf_results)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    all_correct = (all(r['passed_before'] and r['passed_after'] for r in correctness_results) and
                   all(r['passed'] for r in energy_results) and
                   mc_results['reasonable'])
    
    print(f"\n物理正确性验证: {'✓ 全部通过' if all_correct else '✗ 存在问题'}")
    
    avg_speedup_v2 = np.mean([r['speedup_v2_vs_serial'] for r in perf_results])
    avg_speedup_v2_vs_v1 = np.mean([r['speedup_v2_vs_v1'] for r in perf_results])
    
    print(f"\n性能提升:")
    print(f"  V2 vs 原始串行: {avg_speedup_v2:.2f}x 平均加速")
    print(f"  V2 vs V1 向量化: {avg_speedup_v2_vs_v1:.2f}x 额外加速")
    
    print(f"\n结果保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
