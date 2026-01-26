"""
profile_bottleneck.py - 性能瓶颈深度剖析脚本

本脚本使用 cProfile 和自定义计时器对 Network_optimized.py 中的各个函数进行
详细的性能剖析，找出最大的性能瓶颈。

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
import cProfile
import pstats
import io
from datetime import datetime
from functools import wraps
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
OUTPUT_DIR = '/home/ubuntu/profile_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 软核势能函数
# ============================================================================

def soft_core_potential(h):
    """软核势能函数"""
    return np.heaviside(-h, 1.0) * np.power(h, 2)

def calc_ener(r):
    """计算能量"""
    return soft_core_potential(r).sum()

# ============================================================================
# 详细计时装饰器
# ============================================================================

detailed_timing = {}

def detailed_timethis(func):
    """详细计时装饰器，记录调用次数和总时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        name = func.__name__
        if name not in detailed_timing:
            detailed_timing[name] = {'calls': 0, 'total_time': 0.0, 'times': []}
        
        detailed_timing[name]['calls'] += 1
        detailed_timing[name]['total_time'] += (end - start)
        detailed_timing[name]['times'].append(end - start)
        
        return result
    return wrapper


# ============================================================================
# 带详细计时的优化网络类
# ============================================================================

class NetworkProfiled:
    """
    用于性能剖析的网络类
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
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float32)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    # ========================================================================
    # Part Gap 计算函数 (带详细计时)
    # ========================================================================
    
    @detailed_timethis
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
    
    @detailed_timethis
    def part_gap_hidden_after_flip_layer(self, l_s):
        """向量化计算第 l_s 层所有自旋翻转后的 part_gap - 这是主要瓶颈"""
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            S_flipped_n = -self.S[:, l_s, n]
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * S_flipped_n
            
            # 这个嵌套循环是主要瓶颈！
            for mu in range(M):
                S_layer_flipped = self.S[mu, l_s, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_hidden_next = self.J_hidden[l_s, :, :] @ S_layer_flipped
                gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
                part_gap[mu, n, 1:] = gap_next
        
        return part_gap
    
    @detailed_timethis
    def compute_energy_diff(self, part_gap_before, part_gap_after):
        """计算能量差"""
        E_before = soft_core_potential(part_gap_before).sum(axis=2)
        E_after = soft_core_potential(part_gap_after).sum(axis=2)
        return E_after - E_before
    
    @detailed_timethis
    def metropolis_decision(self, delta_E):
        """Metropolis 接受/拒绝决策"""
        M, N = delta_E.shape
        rand_array = np.random.random((M, N))
        return (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
    
    @detailed_timethis
    def apply_updates(self, l_s, accept_mask, delta_E):
        """应用更新"""
        self.S[:, l_s, :] = np.where(accept_mask, -self.S[:, l_s, :], self.S[:, l_s, :])
        self.H += np.sum(delta_E * accept_mask)
    
    def update_S_middle_layer_vectorized(self, l_s):
        """向量化更新中间隐藏层的所有自旋"""
        part_gap_before = self.part_gap_hidden_before_flip_layer(l_s)
        part_gap_after = self.part_gap_hidden_after_flip_layer(l_s)
        
        delta_E = self.compute_energy_diff(part_gap_before, part_gap_after)
        accept_mask = self.metropolis_decision(delta_E)
        self.apply_updates(l_s, accept_mask, delta_E)
    
    def update_all_S_vectorized(self):
        """更新所有隐藏层的自旋"""
        for l_s in range(1, self.num_hidden_node_layers - 1):
            self.update_S_middle_layer_vectorized(l_s)


# ============================================================================
# 分解计时测试
# ============================================================================

def profile_breakdown(M, N, L, num_steps=3):
    """
    分解各个操作的耗时
    """
    print(f"\n配置: M={M}, N={N}, L={L}")
    
    # 重置计时器
    detailed_timing.clear()
    
    net = NetworkProfiled(M, N, L)
    
    # 运行多个 MC 步
    for step in range(num_steps):
        net.update_all_S_vectorized()
    
    # 计算统计
    results = {}
    total_time = sum(v['total_time'] for v in detailed_timing.values())
    
    for name, data in detailed_timing.items():
        avg_time = data['total_time'] / data['calls'] if data['calls'] > 0 else 0
        pct = 100 * data['total_time'] / total_time if total_time > 0 else 0
        results[name] = {
            'calls': data['calls'],
            'total_time': data['total_time'],
            'avg_time': avg_time,
            'percentage': pct
        }
    
    return results, total_time


def profile_nested_loop_impact():
    """
    专门分析嵌套循环的影响
    """
    print("\n" + "=" * 70)
    print("分析嵌套循环对性能的影响")
    print("=" * 70)
    
    # 测试配置
    configs = [
        (30, 3, 10),
        (60, 3, 10),
        (120, 3, 10),
        (120, 5, 10),
        (240, 3, 10),
    ]
    
    results = []
    
    for M, N, L in configs:
        print(f"\n配置: M={M}, N={N}, L={L}")
        
        net = NetworkProfiled(M, N, L)
        l_s = 2  # 测试中间层
        
        # 测试 before_flip (无嵌套循环)
        times_before = []
        for _ in range(5):
            start = time.perf_counter()
            _ = net.part_gap_hidden_before_flip_layer(l_s)
            times_before.append(time.perf_counter() - start)
        
        # 测试 after_flip (有嵌套循环)
        times_after = []
        for _ in range(5):
            start = time.perf_counter()
            _ = net.part_gap_hidden_after_flip_layer(l_s)
            times_after.append(time.perf_counter() - start)
        
        avg_before = np.mean(times_before)
        avg_after = np.mean(times_after)
        ratio = avg_after / avg_before if avg_before > 0 else 0
        
        print(f"  before_flip (无嵌套): {avg_before*1000:.3f} ms")
        print(f"  after_flip (有嵌套):  {avg_after*1000:.3f} ms")
        print(f"  比率: {ratio:.2f}x")
        
        results.append({
            'M': M, 'N': N, 'L': L,
            'time_before': avg_before,
            'time_after': avg_after,
            'ratio': ratio,
            'total_spins': M * N * (L - 1)
        })
    
    return results


def profile_operation_breakdown():
    """
    分析各个操作的耗时占比
    """
    print("\n" + "=" * 70)
    print("分析各操作耗时占比")
    print("=" * 70)
    
    # 使用默认配置
    M, N, L = 120, 3, 10
    
    results, total_time = profile_breakdown(M, N, L, num_steps=5)
    
    print(f"\n总耗时: {total_time:.4f}s")
    print("\n各函数耗时:")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['total_time'])
    
    for name, data in sorted_results:
        print(f"  {name:40s}: {data['total_time']:.4f}s ({data['percentage']:5.1f}%) "
              f"[{data['calls']} calls, {data['avg_time']*1000:.3f}ms/call]")
    
    return results


def profile_memory_operations():
    """
    分析内存操作的影响
    """
    print("\n" + "=" * 70)
    print("分析内存操作的影响")
    print("=" * 70)
    
    M, N, L = 120, 3, 10
    
    net = NetworkProfiled(M, N, L)
    l_s = 2
    
    # 测试 .copy() 操作的影响
    S_layer = net.S[:, l_s, :].copy()
    
    num_iterations = 1000
    
    # 测试 1: 使用 .copy()
    start = time.perf_counter()
    for _ in range(num_iterations):
        for n in range(N):
            temp = S_layer.copy()
            temp[:, n] = -temp[:, n]
    time_with_copy = time.perf_counter() - start
    
    # 测试 2: 使用预分配数组
    temp_array = np.zeros_like(S_layer)
    start = time.perf_counter()
    for _ in range(num_iterations):
        for n in range(N):
            np.copyto(temp_array, S_layer)
            temp_array[:, n] = -temp_array[:, n]
    time_with_preallocate = time.perf_counter() - start
    
    # 测试 3: 使用数学技巧避免复制
    flip_mask = np.ones((M, N), dtype=np.float32)
    start = time.perf_counter()
    for _ in range(num_iterations):
        for n in range(N):
            flip_mask.fill(1)
            flip_mask[:, n] = -1
            temp = S_layer * flip_mask
    time_with_math = time.perf_counter() - start
    
    print(f"  使用 .copy(): {time_with_copy*1000:.2f} ms")
    print(f"  使用预分配:  {time_with_preallocate*1000:.2f} ms")
    print(f"  使用数学技巧: {time_with_math*1000:.2f} ms")
    
    return {
        'copy': time_with_copy,
        'preallocate': time_with_preallocate,
        'math': time_with_math
    }


def generate_visualizations(breakdown_results, loop_results, memory_results):
    """生成可视化图表"""
    print("\n" + "=" * 70)
    print("生成可视化图表")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 各函数耗时占比 (饼图)
    ax = axes[0, 0]
    labels = []
    sizes = []
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    sorted_breakdown = sorted(breakdown_results.items(), key=lambda x: -x[1]['percentage'])
    for name, data in sorted_breakdown:
        if data['percentage'] > 1:  # 只显示占比 > 1% 的
            labels.append(f"{name}\n({data['percentage']:.1f}%)")
            sizes.append(data['percentage'])
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                       colors=colors[:len(sizes)], startangle=90)
    ax.set_title('Function Time Distribution', fontsize=14, fontweight='bold')
    
    # 图2: 嵌套循环影响 (柱状图)
    ax = axes[0, 1]
    x = np.arange(len(loop_results))
    width = 0.35
    
    configs = [f"M={r['M']}\nN={r['N']}" for r in loop_results]
    times_before = [r['time_before'] * 1000 for r in loop_results]
    times_after = [r['time_after'] * 1000 for r in loop_results]
    
    bars1 = ax.bar(x - width/2, times_before, width, label='before_flip (No nested loop)', color='steelblue')
    bars2 = ax.bar(x + width/2, times_after, width, label='after_flip (With nested loop)', color='coral')
    
    # 添加比率标签
    for i, r in enumerate(loop_results):
        ax.annotate(f'{r["ratio"]:.1f}x', xy=(x[i], times_after[i]), 
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Nested Loop Impact on Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: 内存操作对比
    ax = axes[1, 0]
    methods = ['Using .copy()', 'Pre-allocated', 'Math Trick']
    times = [memory_results['copy'] * 1000, 
             memory_results['preallocate'] * 1000, 
             memory_results['math'] * 1000]
    colors = ['coral', 'steelblue', 'green']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{t:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Memory Operation Comparison (1000 iterations)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4: 瓶颈分析总结
    ax = axes[1, 1]
    
    # 计算瓶颈占比
    total_pct = sum(v['percentage'] for v in breakdown_results.values())
    after_flip_pct = breakdown_results.get('part_gap_hidden_after_flip_layer', {}).get('percentage', 0)
    before_flip_pct = breakdown_results.get('part_gap_hidden_before_flip_layer', {}).get('percentage', 0)
    other_pct = total_pct - after_flip_pct - before_flip_pct
    
    categories = ['part_gap_after_flip\n(Main Bottleneck)', 'part_gap_before_flip', 'Other Operations']
    values = [after_flip_pct, before_flip_pct, other_pct]
    colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
    
    bars = ax.barh(categories, values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}%', ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Percentage of Total Time (%)', fontsize=12)
    ax.set_title('Bottleneck Analysis Summary', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'bottleneck_analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")
    
    # 图5: 改进建议图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 创建改进建议的可视化
    improvements = [
        ('Current Implementation', 1.0, 'coral'),
        ('Eliminate Nested Loop\n(Vectorize after_flip)', 2.5, 'orange'),
        ('Add Numba JIT\n(@njit decorator)', 5.0, 'gold'),
        ('GPU Acceleration\n(PyTorch/CuPy)', 50.0, 'green'),
    ]
    
    names = [i[0] for i in improvements]
    speedups = [i[1] for i in improvements]
    colors = [i[2] for i in improvements]
    
    bars = ax.bar(names, speedups, color=colors, alpha=0.8, edgecolor='black', width=0.6)
    
    for bar, speedup in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{speedup:.1f}x', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Expected Speedup Factor', fontsize=12)
    ax.set_title('Potential Performance Improvements', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加说明文本
    ax.text(0.98, 0.98, 
            "Improvement Strategy:\n"
            "1. Vectorize nested loop in after_flip\n"
            "2. Apply Numba JIT compilation\n"
            "3. Migrate to GPU (PyTorch/CuPy)",
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'improvement_potential.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {fig_path}")


def main():
    """主函数"""
    print("=" * 70)
    print("Thermal-DNN-MC-Optimized 性能瓶颈深度剖析")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. 分析各操作耗时占比
    breakdown_results = profile_operation_breakdown()
    
    # 2. 分析嵌套循环的影响
    loop_results = profile_nested_loop_impact()
    
    # 3. 分析内存操作的影响
    memory_results = profile_memory_operations()
    
    # 4. 生成可视化
    generate_visualizations(breakdown_results, loop_results, memory_results)
    
    # 5. 保存结果
    results_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'breakdown': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                         for kk, vv in v.items()} 
                     for k, v in breakdown_results.items()},
        'loop_impact': [{k: float(v) if isinstance(v, (np.floating, float)) else v 
                        for k, v in r.items()} for r in loop_results],
        'memory_operations': {k: float(v) for k, v in memory_results.items()}
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'profile_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n结果已保存: {json_path}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("瓶颈分析总结")
    print("=" * 70)
    
    after_flip_pct = breakdown_results.get('part_gap_hidden_after_flip_layer', {}).get('percentage', 0)
    print(f"\n主要瓶颈: part_gap_hidden_after_flip_layer")
    print(f"  占总耗时: {after_flip_pct:.1f}%")
    print(f"\n瓶颈原因:")
    print(f"  1. 嵌套循环 (for n in range(N): for mu in range(M):)")
    print(f"  2. 每次迭代都调用 .copy() 创建临时数组")
    print(f"  3. 无法利用 NumPy 的向量化优势")
    
    print(f"\n改进建议:")
    print(f"  1. 消除嵌套循环，使用纯向量化计算")
    print(f"  2. 使用 Numba @njit 加速剩余循环")
    print(f"  3. 迁移到 GPU (PyTorch/CuPy)")
    
    print(f"\n结果保存目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
