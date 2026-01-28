#!/usr/bin/env python3
"""
run_experiment1_v2.py - 实验一复现（修复版本）

使用优化的参数在合理时间内完成实验一的复现。
添加了输出刷新以实时显示进度。

作者：Manus AI
日期：2026-01-28
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import os
import sys
from datetime import datetime

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Network_optimized_v3 import NetworkOptimizedV3, warmup_jit


def compute_replica_overlap(S1, S2):
    """计算两个副本之间的自旋重叠参数 q_l"""
    M, L_minus_1, N = S1.shape
    q_l = np.zeros(L_minus_1)
    
    for l in range(L_minus_1):
        overlap = np.sum(S1[:, l, :] * S2[:, l, :])
        q_l[l] = overlap / (M * N)
    
    return q_l


def run_replica_simulation(M, N, L, beta, mc_steps, num_replicas=2, 
                           N_in=784, N_out=2, seed=None):
    """运行多副本模拟并计算重叠参数"""
    ln_alpha = np.log(M / N)
    
    print(f"\n{'='*60}", flush=True)
    print(f"副本模拟: M={M}, N={N}, L={L}, ln(α)={ln_alpha:.2f}", flush=True)
    print(f"MC steps: {mc_steps}, 副本数: {num_replicas}", flush=True)
    print(f"{'='*60}", flush=True)
    
    if seed is not None:
        np.random.seed(seed)
    
    # 生成共享的边界条件
    S_in_shared = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float64)
    S_out_shared = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float64)
    
    # 创建副本
    replicas = []
    for r in range(num_replicas):
        net = NetworkOptimizedV3(M, N, L, N_in, N_out, beta)
        net.S_in = S_in_shared.copy()
        net.S_out = S_out_shared.copy()
        replicas.append(net)
        print(f"  副本 {r+1} 初始化完成", flush=True)
    
    # 运行模拟
    start_time = time.time()
    report_interval = max(1, mc_steps // 10)
    
    for step in range(mc_steps + 1):
        # MC 步
        if step < mc_steps:
            for net in replicas:
                net.mc_step_vectorized()
        
        # 进度报告
        if step > 0 and step % report_interval == 0:
            elapsed = time.time() - start_time
            rate = step / elapsed
            eta = (mc_steps - step) / rate if rate > 0 else 0
            
            # 计算当前重叠
            q_l = compute_replica_overlap(replicas[0].S, replicas[1].S)
            q_mean = np.mean(q_l)
            
            print(f"  Step {step}/{mc_steps} ({100*step/mc_steps:.0f}%), "
                  f"<q>={q_mean:.4f}, {rate:.1f} steps/s, ETA: {eta/60:.1f}min", flush=True)
    
    total_time = time.time() - start_time
    print(f"\n模拟完成，耗时: {total_time/60:.1f} 分钟", flush=True)
    
    # 计算最终重叠参数
    q_l_pairs = []
    for i in range(num_replicas):
        for j in range(i + 1, num_replicas):
            q_l = compute_replica_overlap(replicas[i].S, replicas[j].S)
            q_l_pairs.append(q_l)
    
    q_l_star = np.mean(q_l_pairs, axis=0)
    
    return q_l_star


def plot_phase_diagram(results, output_dir):
    """绘制相图"""
    L = 10
    num_layers = L - 1
    
    ln_alpha_values = [r['ln_alpha'] for r in results]
    q_l_matrix = np.array([r['q_l_star'] for r in results])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 图1: 热图
    ax1 = axes[0]
    colors = ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000']
    cmap = LinearSegmentedColormap.from_list('liquid_solid', colors)
    
    im = ax1.imshow(q_l_matrix.T, aspect='auto', cmap=cmap, 
                    vmin=0, vmax=1, origin='lower')
    
    ax1.set_xticks(range(len(ln_alpha_values)))
    ax1.set_xticklabels([f'{x:.1f}' for x in ln_alpha_values])
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels([f'{l+1}' for l in range(num_layers)])
    
    ax1.set_xlabel('ln α', fontsize=12)
    ax1.set_ylabel('Layer l', fontsize=12)
    ax1.set_title('Phase Diagram: q_l*', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('q_l*', fontsize=12)
    
    # 图2: q_l* vs ln α
    ax2 = axes[1]
    colors_layers = plt.cm.viridis(np.linspace(0, 1, num_layers))
    
    for l in range(num_layers):
        ax2.plot(ln_alpha_values, q_l_matrix[:, l], 'o-', 
                 color=colors_layers[l], linewidth=2, markersize=8,
                 label=f'Layer {l+1}')
    
    ax2.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2, 
                label=f'q* = 1/e')
    
    ax2.set_xlabel('ln α', fontsize=12)
    ax2.set_ylabel('q_l*', fontsize=12)
    ax2.set_title('Layer Overlap vs ln α', fontsize=14)
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # 图3: 层分布
    ax3 = axes[2]
    layers = np.arange(1, num_layers + 1)
    
    for i, r in enumerate(results):
        ax3.plot(layers, r['q_l_star'], 'o-', linewidth=2, markersize=8,
                 label=f"ln α = {r['ln_alpha']:.1f}")
    
    ax3.axhline(y=1/np.e, color='red', linestyle='--', linewidth=2,
                label=f'q* = 1/e')
    
    ax3.set_xlabel('Layer l', fontsize=12)
    ax3.set_ylabel('q_l*', fontsize=12)
    ax3.set_title('Layer Profile of q_l*', fontsize=14)
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xticks(layers)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'phase_diagram_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n相图保存到: {filepath}", flush=True)
    return filepath


def main():
    """主函数"""
    print("=" * 70, flush=True)
    print("实验一：液-固相图构建", flush=True)
    print("=" * 70, flush=True)
    
    # 预热 JIT
    warmup_jit()
    
    # 参数设置
    L = 10
    N_in = 784
    N_out = 2
    beta = 1e5
    num_replicas = 3
    
    # 使用优化的参数配置
    configs = [
        {'M': 100, 'N': 3},    # ln α ≈ 3.5
        {'M': 100, 'N': 5},    # ln α ≈ 3.0
        {'M': 100, 'N': 10},   # ln α ≈ 2.3
        {'M': 100, 'N': 20},   # ln α ≈ 1.6
    ]
    
    mc_steps = 50000  # 5万步，每个配置约3-5分钟
    
    print(f"\n实验配置:", flush=True)
    print(f"  L = {L}, β = {beta:.0e}", flush=True)
    print(f"  MC steps = {mc_steps:,}", flush=True)
    print(f"  副本数 = {num_replicas}", flush=True)
    print(f"\n参数组合:", flush=True)
    for cfg in configs:
        ln_alpha = np.log(cfg['M'] / cfg['N'])
        print(f"    M={cfg['M']}, N={cfg['N']}, ln(α)={ln_alpha:.2f}", flush=True)
    
    # 运行模拟
    results = []
    total_start = time.time()
    
    for i, cfg in enumerate(configs):
        M, N = cfg['M'], cfg['N']
        ln_alpha = np.log(M / N)
        
        print(f"\n{'='*60}", flush=True)
        print(f"配置 {i+1}/{len(configs)}: M={M}, N={N}, ln(α)={ln_alpha:.2f}", flush=True)
        print(f"{'='*60}", flush=True)
        
        q_l_star = run_replica_simulation(
            M=M, N=N, L=L, beta=beta,
            mc_steps=mc_steps,
            num_replicas=num_replicas,
            N_in=N_in, N_out=N_out,
            seed=42 + i
        )
        
        results.append({
            'M': M, 'N': N, 'L': L,
            'ln_alpha': ln_alpha,
            'q_l_star': q_l_star
        })
        
        # 打印当前结果
        print(f"\n层重叠参数 q_l* (ln α = {ln_alpha:.2f}):", flush=True)
        for l in range(L - 1):
            phase = "固态" if q_l_star[l] > 1/np.e else "液态"
            print(f"    层 {l+1}: q_l* = {q_l_star[l]:.4f} ({phase})", flush=True)
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}", flush=True)
    print(f"所有模拟完成，总耗时: {total_time/60:.1f} 分钟", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 绘制相图
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'reports', 'experiment1_results')
    os.makedirs(output_dir, exist_ok=True)
    
    plot_phase_diagram(results, output_dir)
    
    # 打印最终结果摘要
    print("\n" + "=" * 70, flush=True)
    print("实验一结果摘要", flush=True)
    print("=" * 70, flush=True)
    
    print("\n层重叠参数 q_l* (相变边界 q* = 1/e ≈ 0.368):", flush=True)
    print("-" * 70, flush=True)
    
    header = "ln α    | " + " | ".join([f"L{l+1:2d}" for l in range(L-1)])
    print(header, flush=True)
    print("-" * 70, flush=True)
    
    for r in results:
        row = f"{r['ln_alpha']:.2f}    | "
        row += " | ".join([f"{q:.2f}" for q in r['q_l_star']])
        print(row, flush=True)
    
    print("-" * 70, flush=True)
    print("\n相态判断 (q > 1/e = 固态, q < 1/e = 液态):", flush=True)
    
    for r in results:
        solid_layers = [l+1 for l, q in enumerate(r['q_l_star']) if q > 1/np.e]
        liquid_layers = [l+1 for l, q in enumerate(r['q_l_star']) if q <= 1/np.e]
        
        print(f"\nln α = {r['ln_alpha']:.2f}:", flush=True)
        print(f"  固态层: {solid_layers if solid_layers else '无'}", flush=True)
        print(f"  液态层: {liquid_layers if liquid_layers else '无'}", flush=True)
    
    return results


if __name__ == "__main__":
    main()
