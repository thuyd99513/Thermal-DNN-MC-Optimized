#!/usr/bin/env python3
"""
run_experiment1_full.py - 实验一完整复现（MC版本）

复现论文 "Liquid and solid layers in a thermal deep learning machine" 中的 Figure 1b。
使用与论文一致的参数规模进行蒙特卡洛模拟。

论文参数:
- L = 10 (层数)
- M = 2000 (样本数)
- N = 5, 10, 20, 40 (每层神经元数，对应 ln α = 5.99, 5.30, 4.61, 3.91)
- β = 10^5 (逆温度)
- t* = 5×10^5 (模拟时间，这里用 MC 步数近似)
- N_s = 8 (独立样本/初始配置)
- N_r = 8 (每个样本的副本数)

作者：Manus AI
日期：2026-01-29
"""

import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime
from time import time

# 添加 py_functions 到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'py_functions'))

from Network_optimized_v3 import NetworkOptimizedV3, warmup_jit


# ============================================================================
# 配置参数
# ============================================================================

# 论文参数配置
PAPER_CONFIG = {
    'L': 10,                    # 层数
    'M': 2000,                  # 样本数
    'N_values': [5, 10, 20, 40], # 每层神经元数 (对应不同 ln α)
    'N_in': 784,                # 输入层维度 (MNIST 28x28)
    'N_out': 2,                 # 输出层维度 (二分类)
    'beta': 1e5,                # 逆温度
    'mc_steps': 500000,         # MC 步数 (对应论文的 t* = 5×10^5)
    'num_samples': 8,           # 独立样本数 N_s
    'num_replicas': 8,          # 每个样本的副本数 N_r
}

# 可选的快速测试配置
QUICK_CONFIG = {
    'L': 10,
    'M': 500,
    'N_values': [5, 10, 20, 40],
    'N_in': 784,
    'N_out': 2,
    'beta': 1e5,
    'mc_steps': 50000,
    'num_samples': 2,
    'num_replicas': 4,
}

# 中等规模配置 (用于验证)
MEDIUM_CONFIG = {
    'L': 10,
    'M': 1000,
    'N_values': [5, 10, 20, 40],
    'N_in': 784,
    'N_out': 2,
    'beta': 1e5,
    'mc_steps': 200000,
    'num_samples': 4,
    'num_replicas': 6,
}


# ============================================================================
# 核心计算函数
# ============================================================================

def compute_layer_overlap(S1, S2):
    """
    计算两个副本之间的层重叠参数 q_l
    
    Args:
        S1, S2: shape (M, L-1, N) 的自旋配置
    
    Returns:
        q_l: shape (L-1,) 的层重叠参数
    """
    M, L_minus_1, N = S1.shape
    q_l = np.zeros(L_minus_1)
    
    for l in range(L_minus_1):
        # q_l = (1/MN) * sum_{mu,i} S1[mu,l,i] * S2[mu,l,i]
        overlap = np.sum(S1[:, l, :] * S2[:, l, :])
        q_l[l] = overlap / (M * N)
    
    return q_l


def run_single_sample(config, sample_id, output_dir, verbose=True):
    """
    运行单个样本的多副本模拟
    
    Args:
        config: 配置字典
        sample_id: 样本编号
        output_dir: 输出目录
        verbose: 是否打印详细信息
    
    Returns:
        results: 包含所有 N 值结果的字典
    """
    L = config['L']
    M = config['M']
    N_values = config['N_values']
    N_in = config['N_in']
    N_out = config['N_out']
    beta = config['beta']
    mc_steps = config['mc_steps']
    num_replicas = config['num_replicas']
    
    results = {}
    
    for N in N_values:
        ln_alpha = np.log(M / N)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"样本 {sample_id}, N={N}, ln(α)={ln_alpha:.2f}")
            print(f"M={M}, L={L}, β={beta:.0e}, MC步数={mc_steps:,}")
            print(f"副本数={num_replicas}")
            print(f"{'='*60}")
        
        # 设置随机种子（确保可重复性）
        base_seed = sample_id * 1000 + N
        np.random.seed(base_seed)
        
        # 生成共享的边界条件 (所有副本共享相同的输入/输出)
        S_in_shared = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float64)
        S_out_shared = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float64)
        
        # 创建副本
        replicas = []
        for r in range(num_replicas):
            np.random.seed(base_seed + r + 1)  # 每个副本不同的初始配置
            net = NetworkOptimizedV3(M, N, L, N_in, N_out, beta)
            net.S_in = S_in_shared.copy()
            net.S_out = S_out_shared.copy()
            replicas.append(net)
        
        if verbose:
            print(f"  {num_replicas} 个副本初始化完成")
        
        # 运行 MC 模拟
        start_time = time()
        report_interval = max(1, mc_steps // 20)  # 每 5% 报告一次
        
        # 记录能量历史
        energy_history = []
        
        for step in range(mc_steps + 1):
            # MC 步
            if step < mc_steps:
                for net in replicas:
                    net.mc_step_vectorized()
            
            # 进度报告
            if step > 0 and step % report_interval == 0:
                elapsed = time() - start_time
                rate = step / elapsed
                eta = (mc_steps - step) / rate if rate > 0 else 0
                
                # 计算当前平均能量
                avg_energy = np.mean([net.H for net in replicas])
                energy_history.append((step, avg_energy))
                
                # 计算当前平均重叠
                q_l_sum = np.zeros(L - 1)
                count = 0
                for i in range(num_replicas):
                    for j in range(i + 1, num_replicas):
                        q_l_sum += compute_layer_overlap(replicas[i].S, replicas[j].S)
                        count += 1
                q_l_avg = q_l_sum / count
                q_mean = np.mean(q_l_avg)
                
                if verbose:
                    print(f"  Step {step:>7,}/{mc_steps:,} ({100*step/mc_steps:>5.1f}%), "
                          f"E={avg_energy:.2f}, <q>={q_mean:.4f}, "
                          f"{rate:.1f} steps/s, ETA: {eta/60:.1f}min", flush=True)
        
        total_time = time() - start_time
        
        # 计算最终的层重叠参数 q_l*
        # 对所有副本对进行平均
        q_l_pairs = []
        for i in range(num_replicas):
            for j in range(i + 1, num_replicas):
                q_l = compute_layer_overlap(replicas[i].S, replicas[j].S)
                q_l_pairs.append(q_l)
        
        q_l_star = np.mean(q_l_pairs, axis=0)
        q_l_std = np.std(q_l_pairs, axis=0)
        
        # 保存结果
        results[N] = {
            'N': N,
            'ln_alpha': ln_alpha,
            'q_l_star': q_l_star.tolist(),
            'q_l_std': q_l_std.tolist(),
            'energy_history': energy_history,
            'final_energy': np.mean([net.H for net in replicas]),
            'simulation_time': total_time,
        }
        
        if verbose:
            print(f"\n  模拟完成，耗时: {total_time/60:.1f} 分钟")
            print(f"  最终能量: {results[N]['final_energy']:.2f}")
            print(f"  层重叠参数 q_l*:")
            for l in range(L - 1):
                phase = "固态" if q_l_star[l] > 1/np.e else "液态"
                print(f"    层 {l+1}: q_l* = {q_l_star[l]:.4f} ± {q_l_std[l]:.4f} ({phase})")
        
        # 保存中间结果
        sample_result_file = os.path.join(
            output_dir, f'sample_{sample_id:02d}_N_{N:02d}_result.json'
        )
        with open(sample_result_file, 'w') as f:
            json.dump(results[N], f, indent=2)
        
        if verbose:
            print(f"  结果保存到: {sample_result_file}")
    
    return results


def aggregate_results(all_results, config, output_dir):
    """
    汇总所有样本的结果
    
    Args:
        all_results: 所有样本结果的列表
        config: 配置字典
        output_dir: 输出目录
    
    Returns:
        aggregated: 汇总后的结果字典
    """
    L = config['L']
    N_values = config['N_values']
    M = config['M']
    
    aggregated = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'results': {}
    }
    
    for N in N_values:
        ln_alpha = np.log(M / N)
        
        # 收集所有样本的 q_l*
        q_l_all = []
        for sample_results in all_results:
            if N in sample_results:
                q_l_all.append(sample_results[N]['q_l_star'])
        
        q_l_all = np.array(q_l_all)
        
        # 计算平均值和标准误差
        q_l_mean = np.mean(q_l_all, axis=0)
        q_l_std = np.std(q_l_all, axis=0)
        q_l_sem = q_l_std / np.sqrt(len(q_l_all))  # 标准误差
        
        aggregated['results'][str(N)] = {
            'N': N,
            'ln_alpha': ln_alpha,
            'q_l_mean': q_l_mean.tolist(),
            'q_l_std': q_l_std.tolist(),
            'q_l_sem': q_l_sem.tolist(),
            'num_samples': len(q_l_all),
        }
    
    # 保存汇总结果
    aggregated_file = os.path.join(output_dir, 'aggregated_results.json')
    with open(aggregated_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\n汇总结果保存到: {aggregated_file}")
    
    return aggregated


def print_final_summary(aggregated):
    """打印最终结果摘要"""
    print("\n" + "=" * 70)
    print("实验一完整复现 - 最终结果摘要")
    print("=" * 70)
    
    config = aggregated['config']
    print(f"\n配置: M={config['M']}, L={config['L']}, β={config['beta']:.0e}")
    print(f"MC步数: {config['mc_steps']:,}")
    print(f"独立样本数: {config['num_samples']}, 副本数: {config['num_replicas']}")
    
    print("\n层重叠参数 q_l* (相变边界 q* = 1/e ≈ 0.368):")
    print("-" * 70)
    
    L = config['L']
    header = "ln α    | " + " | ".join([f"L{l+1:2d}" for l in range(L-1)])
    print(header)
    print("-" * 70)
    
    for N_str, data in sorted(aggregated['results'].items(), 
                               key=lambda x: -x[1]['ln_alpha']):
        row = f"{data['ln_alpha']:.2f}    | "
        row += " | ".join([f"{q:.2f}" for q in data['q_l_mean']])
        print(row)
    
    print("-" * 70)
    
    print("\n相态判断 (q > 1/e = 固态, q < 1/e = 液态):")
    for N_str, data in sorted(aggregated['results'].items(),
                               key=lambda x: -x[1]['ln_alpha']):
        solid_layers = [l+1 for l, q in enumerate(data['q_l_mean']) if q > 1/np.e]
        liquid_layers = [l+1 for l, q in enumerate(data['q_l_mean']) if q <= 1/np.e]
        
        print(f"\nln α = {data['ln_alpha']:.2f} (N={data['N']}):")
        print(f"  固态层: {solid_layers if solid_layers else '无'}")
        print(f"  液态层: {liquid_layers if liquid_layers else '无'}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='实验一完整复现 - 液固相图构建 (MC版本)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整论文参数 (需要大量计算资源)
  python run_experiment1_full.py --config paper
  
  # 中等规模 (用于验证)
  python run_experiment1_full.py --config medium
  
  # 快速测试
  python run_experiment1_full.py --config quick
  
  # 只运行特定样本 (用于并行计算)
  python run_experiment1_full.py --config paper --sample-id 0
  python run_experiment1_full.py --config paper --sample-id 1
  ...
  
  # 汇总已完成的样本结果
  python run_experiment1_full.py --aggregate-only --output-dir results/
        """
    )
    
    parser.add_argument('--config', type=str, default='paper',
                        choices=['paper', 'medium', 'quick'],
                        help='配置选择: paper(完整), medium(中等), quick(快速测试)')
    
    parser.add_argument('--sample-id', type=int, default=None,
                        help='只运行指定的样本ID (用于并行计算)')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录 (默认: results/experiment1_full_YYYYMMDD_HHMMSS)')
    
    parser.add_argument('--aggregate-only', action='store_true',
                        help='只汇总已有结果，不运行新模拟')
    
    parser.add_argument('--skip-warmup', action='store_true',
                        help='跳过 JIT 预热 (如果已经预热过)')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.config == 'paper':
        config = PAPER_CONFIG.copy()
    elif args.config == 'medium':
        config = MEDIUM_CONFIG.copy()
    else:
        config = QUICK_CONFIG.copy()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'results', f'experiment1_{args.config}_{timestamp}'
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=" * 70)
    print("实验一完整复现 - 液固相图构建 (MC版本)")
    print("=" * 70)
    print(f"\n配置: {args.config}")
    print(f"输出目录: {output_dir}")
    print(f"\n参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 如果只是汇总结果
    if args.aggregate_only:
        print("\n汇总已有结果...")
        all_results = []
        for sample_id in range(config['num_samples']):
            sample_results = {}
            for N in config['N_values']:
                result_file = os.path.join(
                    output_dir, f'sample_{sample_id:02d}_N_{N:02d}_result.json'
                )
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        sample_results[N] = json.load(f)
            if sample_results:
                all_results.append(sample_results)
        
        if all_results:
            aggregated = aggregate_results(all_results, config, output_dir)
            print_final_summary(aggregated)
        else:
            print("未找到任何结果文件!")
        return
    
    # JIT 预热
    if not args.skip_warmup:
        print("\n预热 JIT 编译...")
        warmup_jit()
        print("JIT 预热完成")
    
    # 确定要运行的样本
    if args.sample_id is not None:
        sample_ids = [args.sample_id]
    else:
        sample_ids = list(range(config['num_samples']))
    
    print(f"\n将运行样本: {sample_ids}")
    
    # 运行模拟
    all_results = []
    total_start = time()
    
    for sample_id in sample_ids:
        print(f"\n{'#'*70}")
        print(f"# 样本 {sample_id + 1}/{config['num_samples']}")
        print(f"{'#'*70}")
        
        sample_results = run_single_sample(config, sample_id, output_dir, verbose=True)
        all_results.append(sample_results)
    
    total_time = time() - total_start
    
    print(f"\n{'='*70}")
    print(f"所有模拟完成，总耗时: {total_time/3600:.2f} 小时")
    print(f"{'='*70}")
    
    # 如果运行了所有样本，进行汇总
    if args.sample_id is None:
        aggregated = aggregate_results(all_results, config, output_dir)
        print_final_summary(aggregated)
    else:
        print(f"\n样本 {args.sample_id} 完成。")
        print(f"运行所有样本后，使用 --aggregate-only 汇总结果。")


if __name__ == "__main__":
    main()
