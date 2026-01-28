#!/usr/bin/env python3
"""
reproduce_experiment2.py - 复现论文实验二：全局训练动态

本脚本复现论文 Figure 2 的结果，测量：
1. 能量 E(t)/E(0) 随时间的衰减
2. 准确率 A(t) 随时间的演化
3. 自相关函数 c(t, t_w)
4. 副本间重叠 q(t, t_w)

作者：Manus AI
日期：2026-01-28
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from datetime import datetime
from time import time

# 导入优化后的网络类
from Network_optimized_v3 import NetworkOptimizedV3

# ============================================================================
# 物理量计算函数
# ============================================================================

def compute_gap_all_layers(net):
    """
    计算所有层的 gap 值
    
    Returns:
        gap_list: 每层的 gap 数组列表
    """
    gap_list = []
    
    # 第一层 (输入层到第一隐藏层)
    # gap = (J_in @ S_in.T) / sqrt(N_in) * S_0
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN  # (M, N)
    gap_0 = h_0 * net.S[:, 0, :]  # (M, N)
    gap_list.append(gap_0)
    
    # 中间层
    for l in range(net.num_hidden_bond_layers):
        # gap = (J_hidden[l] @ S[:, l, :].T) / sqrt(N) * S[:, l+1, :]
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N  # (M, N)
        gap_l = h_l * net.S[:, l+1, :]  # (M, N)
        gap_list.append(gap_l)
    
    # 输出层
    # gap = (J_out @ S[:, -1, :].T) / sqrt(N) * S_out
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N  # (M, N_out)
    gap_out = h_out * net.S_out  # (M, N_out)
    gap_list.append(gap_out)
    
    return gap_list


def compute_energy(net):
    """
    计算总能量 E = Σ V(gap)，其中 V(x) = x² if x < 0 else 0
    """
    gap_list = compute_gap_all_layers(net)
    total_energy = 0.0
    
    for gap in gap_list:
        # 软核势能：只惩罚负 gap
        negative_gaps = gap[gap < 0]
        total_energy += np.sum(negative_gaps ** 2)
    
    return total_energy


def compute_accuracy(net):
    """
    计算训练集准确率
    A = (1/M) Σ_μ Θ(min_i gap_i^μ)
    
    对于每个样本，如果所有层的最小 gap > 0，则认为正确分类
    """
    gap_list = compute_gap_all_layers(net)
    
    # 对每个样本，计算所有层的最小 gap
    min_gaps = np.full(net.M, np.inf)
    
    for gap in gap_list:
        # gap shape: (M, N) 或 (M, N_out)
        sample_min = np.min(gap, axis=1)  # (M,)
        min_gaps = np.minimum(min_gaps, sample_min)
    
    # 计算准确率
    correct = np.sum(min_gaps > 0)
    accuracy = correct / net.M
    
    return accuracy


def compute_spin_autocorrelation(S_t, S_tw):
    """
    计算自旋自相关函数
    c(t, t_w) = (1/N_dof) Σ S_i(t) S_i(t_w)
    
    Args:
        S_t: 时间 t 的自旋配置 (M, L-1, N)
        S_tw: 时间 t_w 的自旋配置 (M, L-1, N)
    
    Returns:
        自相关函数值
    """
    N_dof = S_t.size
    return np.sum(S_t * S_tw) / N_dof


def compute_replica_overlap(S_a, S_b):
    """
    计算副本间重叠
    q(t) = (1/N_dof) Σ S_i^a(t) S_i^b(t)
    
    Args:
        S_a: 副本 a 的自旋配置 (M, L-1, N)
        S_b: 副本 b 的自旋配置 (M, L-1, N)
    
    Returns:
        重叠参数值
    """
    N_dof = S_a.size
    return np.sum(S_a * S_b) / N_dof


def compute_layer_overlap(S_a, S_b, layer):
    """
    计算特定层的副本间重叠
    q_l(t) = (1/N) Σ_n (1/M) Σ_μ S_l,n^a(μ,t) S_l,n^b(μ,t)
    """
    S_a_layer = S_a[:, layer, :]  # (M, N)
    S_b_layer = S_b[:, layer, :]  # (M, N)
    return np.mean(S_a_layer * S_b_layer)


# ============================================================================
# 实验二主函数
# ============================================================================

def run_experiment2(M, N, L, beta, mc_steps, t_w_list, log_interval=100, seed=42):
    """
    运行实验二：全局训练动态
    
    Args:
        M: 样本数
        N: 每层神经元数
        L: 层数
        beta: 逆温度
        mc_steps: MC 步数
        t_w_list: 等待时间列表
        log_interval: 记录间隔
        seed: 随机种子
    
    Returns:
        results: 包含所有测量结果的字典
    """
    print(f"\n{'='*60}")
    print(f"实验二：全局训练动态")
    print(f"M={M}, N={N}, L={L}, β={beta:.0e}")
    print(f"MC steps={mc_steps}, t_w={t_w_list}")
    print(f"{'='*60}")
    
    # 初始化两个副本（用于计算副本重叠）
    np.random.seed(seed)
    net_a = NetworkOptimizedV3(M, N, L, beta=beta, seed=seed)
    net_b = NetworkOptimizedV3(M, N, L, beta=beta, seed=seed+1000)
    
    # 共享权重（副本只在自旋配置上不同）
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    net_b.S_in = net_a.S_in.copy()
    net_b.S_out = net_a.S_out.copy()
    
    # 计算初始能量
    E0 = compute_energy(net_a)
    print(f"初始能量 E(0) = {E0:.4f}")
    
    # 存储结果
    time_points = []
    energy_ratio = []
    accuracy = []
    
    # 存储用于自相关的配置
    S_tw_configs = {tw: None for tw in t_w_list}
    autocorr_data = {tw: {'t': [], 'c': []} for tw in t_w_list}
    overlap_data = {tw: {'t': [], 'q': []} for tw in t_w_list}
    
    # 层重叠数据
    layer_overlap_data = []
    
    start_time = time()
    
    for step in range(1, mc_steps + 1):
        # MC 步：更新两个副本
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        # 记录数据
        if step % log_interval == 0 or step == 1:
            t = step
            time_points.append(t)
            
            # 计算能量和准确率
            E_t = compute_energy(net_a)
            A_t = compute_accuracy(net_a)
            energy_ratio.append(E_t / E0 if E0 > 0 else 1.0)
            accuracy.append(A_t)
            
            # 保存 t_w 时刻的配置
            for tw in t_w_list:
                if step == tw:
                    S_tw_configs[tw] = net_a.S.copy()
                    print(f"  保存 t_w={tw} 时刻的配置")
            
            # 计算自相关和副本重叠
            for tw in t_w_list:
                if S_tw_configs[tw] is not None and step >= tw:
                    c_t = compute_spin_autocorrelation(net_a.S, S_tw_configs[tw])
                    q_t = compute_replica_overlap(net_a.S, net_b.S)
                    
                    autocorr_data[tw]['t'].append(t)
                    autocorr_data[tw]['c'].append(c_t)
                    overlap_data[tw]['t'].append(t)
                    overlap_data[tw]['q'].append(q_t)
            
            # 计算层重叠
            layer_q = []
            for l in range(L - 1):
                q_l = compute_layer_overlap(net_a.S, net_b.S, l)
                layer_q.append(q_l)
            layer_overlap_data.append({'t': t, 'q_l': layer_q})
        
        # 进度输出
        if step % (mc_steps // 10) == 0:
            elapsed = time() - start_time
            eta = elapsed / step * (mc_steps - step)
            E_t = compute_energy(net_a)
            A_t = compute_accuracy(net_a)
            print(f"  Step {step}/{mc_steps} ({100*step/mc_steps:.0f}%), "
                  f"E/E0={E_t/E0:.4f}, A={A_t:.4f}, "
                  f"ETA: {eta/60:.1f}min", flush=True)
    
    total_time = time() - start_time
    print(f"\n模拟完成，总耗时: {total_time/60:.1f} 分钟")
    
    # 整理结果
    results = {
        'M': M, 'N': N, 'L': L, 'beta': beta,
        'ln_alpha': np.log(M / N),
        'E0': E0,
        'time_points': np.array(time_points),
        'energy_ratio': np.array(energy_ratio),
        'accuracy': np.array(accuracy),
        'autocorr_data': autocorr_data,
        'overlap_data': overlap_data,
        'layer_overlap_data': layer_overlap_data,
        't_w_list': t_w_list
    }
    
    return results


def plot_experiment2_results(results_list, output_dir):
    """
    绘制实验二结果图表
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # (a) E(t)/E(0) vs t
    ax = axes[0, 0]
    for results in results_list:
        ln_alpha = results['ln_alpha']
        ax.loglog(results['time_points'], results['energy_ratio'], 
                  'o-', markersize=3, label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('E(t)/E(0)')
    ax.set_title('(a) Energy decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) A(t) vs t
    ax = axes[0, 1]
    for results in results_list:
        ln_alpha = results['ln_alpha']
        ax.semilogx(results['time_points'], results['accuracy'],
                    'o-', markersize=3, label=f"ln α = {ln_alpha:.1f}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('A(t)')
    ax.set_title('(b) Training accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # (c) E* vs ln α
    ax = axes[0, 2]
    ln_alphas = [r['ln_alpha'] for r in results_list]
    E_stars = [r['energy_ratio'][-1] for r in results_list]
    ax.semilogy(ln_alphas, E_stars, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('ln α')
    ax.set_ylabel('E*/E(0)')
    ax.set_title('(c) Final energy vs ln α')
    ax.grid(True, alpha=0.3)
    
    # (d) A* vs ln α
    ax = axes[1, 0]
    A_stars = [r['accuracy'][-1] for r in results_list]
    ax.plot(ln_alphas, A_stars, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('ln α')
    ax.set_ylabel('A*')
    ax.set_title('(d) Final accuracy vs ln α')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # (e) c(t, t_w) vs t
    ax = axes[1, 1]
    # 使用第一个结果的自相关数据
    if results_list:
        results = results_list[0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(results['t_w_list'])))
        for i, tw in enumerate(results['t_w_list']):
            data = results['autocorr_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['c'], 'o-', markersize=3,
                            color=colors[i], label=f"t_w = {tw}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('c(t, t_w)')
    ax.set_title('(e) Spin autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (f) q(t, t_w) vs t
    ax = axes[1, 2]
    if results_list:
        results = results_list[0]
        for i, tw in enumerate(results['t_w_list']):
            data = results['overlap_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['q'], 'o-', markersize=3,
                            color=colors[i], label=f"t_w = {tw}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('q(t, t_w)')
    ax.set_title('(f) Replica overlap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'experiment2_results_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表保存到: {filepath}")
    return filepath


def main():
    """主函数"""
    print("=" * 70)
    print("实验二：全局训练动态 (Figure 2)")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'reports', 'experiment2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 实验参数
    L = 10
    beta = 1e5
    mc_steps = 50000
    t_w_list = [100, 1000, 5000]
    log_interval = 100
    
    # 不同的 ln α 配置
    configs = [
        {'M': 50, 'N': 5},    # ln α ≈ 2.3
        {'M': 100, 'N': 5},   # ln α ≈ 3.0
        {'M': 200, 'N': 5},   # ln α ≈ 3.7
        {'M': 100, 'N': 10},  # ln α ≈ 2.3
    ]
    
    print(f"\n实验配置:")
    print(f"  L = {L}, β = {beta:.0e}")
    print(f"  MC steps = {mc_steps}")
    print(f"  t_w = {t_w_list}")
    print(f"  配置数: {len(configs)}")
    
    # 预热 JIT
    print("\n预热 JIT 编译...")
    warmup_net = NetworkOptimizedV3(10, 3, 5, beta=beta, seed=0)
    for _ in range(10):
        warmup_net.mc_step_vectorized()
    print("JIT 预热完成")
    
    # 运行实验
    results_list = []
    for i, config in enumerate(configs):
        M, N = config['M'], config['N']
        ln_alpha = np.log(M / N)
        print(f"\n{'='*60}")
        print(f"配置 {i+1}/{len(configs)}: M={M}, N={N}, ln α={ln_alpha:.2f}")
        print(f"{'='*60}")
        
        results = run_experiment2(M, N, L, beta, mc_steps, t_w_list, 
                                  log_interval=log_interval, seed=42+i)
        results_list.append(results)
    
    # 绘制结果
    print("\n生成图表...")
    filepath = plot_experiment2_results(results_list, output_dir)
    
    # 打印结果摘要
    print("\n" + "=" * 70)
    print("实验二结果摘要")
    print("=" * 70)
    
    print("\n最终值 (t* = {})".format(mc_steps))
    print("-" * 50)
    print(f"{'ln α':>8} | {'E*/E(0)':>12} | {'A*':>8}")
    print("-" * 50)
    
    for results in results_list:
        ln_alpha = results['ln_alpha']
        E_star = results['energy_ratio'][-1]
        A_star = results['accuracy'][-1]
        print(f"{ln_alpha:>8.2f} | {E_star:>12.4e} | {A_star:>8.4f}")
    
    print("-" * 50)
    
    return results_list, filepath


if __name__ == "__main__":
    main()
