#!/usr/bin/env python3
"""
benchmark_speedup.py - S和J优化加速比测试

对比优化前后的MC模拟速度差异。
使用 MCS=100 进行快速测试。

作者：Manus AI
日期：2026-01-30
"""

import numpy as np
import os
import sys
import json
import time
from datetime import datetime

# 添加 py_functions 到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'py_functions'))

# 尝试导入优化版本和原始版本
try:
    from Network_optimized_v3 import NetworkOptimizedV3
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    print("警告: 无法导入 NetworkOptimizedV3")

try:
    from Network import Network
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False
    print("警告: 无法导入原始 Network")


def benchmark_original(M, N, L, mc_steps, beta=1e5, seed=42):
    """测试原始实现的速度"""
    if not HAS_ORIGINAL:
        return None, None
    
    print(f"\n{'='*60}")
    print(f"测试原始实现 (Network)")
    print(f"M={M}, N={N}, L={L}, MCS={mc_steps}")
    print(f"{'='*60}")
    
    # 创建网络
    np.random.seed(seed)
    net = Network(M=M, N=N, L=L, N_in=784, N_out=2, beta=beta)
    
    # 预热（让 Python 解释器优化）
    print("预热中...")
    for _ in range(10):
        net.mc_update_J_or_S()
    
    # 正式测试
    print(f"运行 {mc_steps} 个 MC 步...")
    start_time = time.time()
    
    for step in range(mc_steps):
        net.mc_update_J_or_S()
        if (step + 1) % (mc_steps // 5) == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"  Step {step+1}/{mc_steps}, {rate:.2f} steps/s")
    
    total_time = time.time() - start_time
    rate = mc_steps / total_time
    
    print(f"\n原始实现结果:")
    print(f"  总时间: {total_time:.3f} 秒")
    print(f"  速度: {rate:.2f} MC步/秒")
    
    return total_time, rate


def benchmark_optimized(M, N, L, mc_steps, beta=1e5, seed=42, balanced=True):
    """测试优化实现的速度"""
    if not HAS_OPTIMIZED:
        return None, None
    
    mode_str = "平衡模式" if balanced else "非平衡模式"
    print(f"\n{'='*60}")
    print(f"测试优化实现 (NetworkOptimizedV3) - {mode_str}")
    print(f"M={M}, N={N}, L={L}, MCS={mc_steps}")
    print(f"{'='*60}")
    
    # 创建网络
    np.random.seed(seed)
    net = NetworkOptimizedV3(M=M, N=N, L=L, N_in=784, N_out=2, beta=beta, seed=seed)
    
    # 打印比例信息
    J_sweeps_ratio, num_S, num_J, S_per_sweep, J_per_sweep = net._compute_update_ratio()
    print(f"S/J 更新比例:")
    print(f"  num_S = {num_S:,}, num_J = {num_J:,}")
    print(f"  原始比例 num_S:num_J = {num_S/num_J:.2f}:1")
    print(f"  J_sweeps_per_S_sweep = {J_sweeps_ratio:.4f}")
    
    # 预热 JIT
    print("预热 JIT 编译中...")
    for _ in range(3):
        if balanced:
            net.mc_step_vectorized_balanced()
        else:
            net.mc_step_vectorized()
    
    # 重置累积器
    if hasattr(net, '_J_sweep_accumulator'):
        net._J_sweep_accumulator = 0.0
    
    # 正式测试
    print(f"运行 {mc_steps} 个 MC 步...")
    start_time = time.time()
    
    for step in range(mc_steps):
        if balanced:
            net.mc_step_vectorized_balanced()
        else:
            net.mc_step_vectorized()
        
        if (step + 1) % (mc_steps // 5) == 0:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            print(f"  Step {step+1}/{mc_steps}, {rate:.2f} steps/s")
    
    total_time = time.time() - start_time
    rate = mc_steps / total_time
    
    print(f"\n优化实现结果 ({mode_str}):")
    print(f"  总时间: {total_time:.3f} 秒")
    print(f"  速度: {rate:.2f} MC步/秒")
    
    return total_time, rate


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='S和J优化加速比测试')
    parser.add_argument('--mc-steps', type=int, default=100, help='MC步数 (默认: 100)')
    parser.add_argument('--M', type=int, default=2000, help='样本数 (默认: 2000)')
    parser.add_argument('--N', type=int, default=20, help='每层神经元数 (默认: 20)')
    parser.add_argument('--L', type=int, default=10, help='层数 (默认: 10)')
    parser.add_argument('--beta', type=float, default=1e5, help='逆温度 (默认: 1e5)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')
    parser.add_argument('--skip-original', action='store_true', help='跳过原始实现测试')
    parser.add_argument('--output-dir', type=str, default='.', help='输出目录')
    
    args = parser.parse_args()
    
    print("="*70)
    print("S 和 J 优化加速比测试")
    print("="*70)
    print(f"测试参数:")
    print(f"  MC 步数: {args.mc_steps}")
    print(f"  M (样本数): {args.M}")
    print(f"  N (神经元数): {args.N}")
    print(f"  L (层数): {args.L}")
    print(f"  β (逆温度): {args.beta:.0e}")
    print(f"  随机种子: {args.seed}")
    print("="*70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'mc_steps': args.mc_steps,
            'M': args.M,
            'N': args.N,
            'L': args.L,
            'beta': args.beta,
            'seed': args.seed
        },
        'results': {}
    }
    
    # 测试原始实现
    if not args.skip_original and HAS_ORIGINAL:
        orig_time, orig_rate = benchmark_original(
            args.M, args.N, args.L, args.mc_steps, args.beta, args.seed
        )
        if orig_time is not None:
            results['results']['original'] = {
                'time': orig_time,
                'rate': orig_rate
            }
    else:
        orig_rate = None
        print("\n跳过原始实现测试")
    
    # 测试优化实现（平衡模式）
    if HAS_OPTIMIZED:
        opt_balanced_time, opt_balanced_rate = benchmark_optimized(
            args.M, args.N, args.L, args.mc_steps, args.beta, args.seed, balanced=True
        )
        if opt_balanced_time is not None:
            results['results']['optimized_balanced'] = {
                'time': opt_balanced_time,
                'rate': opt_balanced_rate
            }
        
        # 测试优化实现（非平衡模式，仅供参考）
        opt_unbalanced_time, opt_unbalanced_rate = benchmark_optimized(
            args.M, args.N, args.L, args.mc_steps, args.beta, args.seed, balanced=False
        )
        if opt_unbalanced_time is not None:
            results['results']['optimized_unbalanced'] = {
                'time': opt_unbalanced_time,
                'rate': opt_unbalanced_rate
            }
    
    # 计算加速比
    print("\n" + "="*70)
    print("加速比总结")
    print("="*70)
    
    if orig_rate and opt_balanced_rate:
        speedup_balanced = opt_balanced_rate / orig_rate
        print(f"优化版本（平衡模式）vs 原始版本:")
        print(f"  原始速度: {orig_rate:.2f} MC步/秒")
        print(f"  优化速度: {opt_balanced_rate:.2f} MC步/秒")
        print(f"  加速比: {speedup_balanced:.2f}x")
        results['results']['speedup_balanced'] = speedup_balanced
    
    if orig_rate and opt_unbalanced_rate:
        speedup_unbalanced = opt_unbalanced_rate / orig_rate
        print(f"\n优化版本（非平衡模式）vs 原始版本:")
        print(f"  原始速度: {orig_rate:.2f} MC步/秒")
        print(f"  优化速度: {opt_unbalanced_rate:.2f} MC步/秒")
        print(f"  加速比: {speedup_unbalanced:.2f}x")
        results['results']['speedup_unbalanced'] = speedup_unbalanced
    
    if opt_balanced_rate and opt_unbalanced_rate:
        ratio = opt_unbalanced_rate / opt_balanced_rate
        print(f"\n平衡模式 vs 非平衡模式:")
        print(f"  平衡模式速度: {opt_balanced_rate:.2f} MC步/秒")
        print(f"  非平衡模式速度: {opt_unbalanced_rate:.2f} MC步/秒")
        print(f"  比值: {ratio:.2f}x (非平衡更快，但J更新频率不正确)")
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'benchmark_speedup_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)


if __name__ == '__main__':
    main()
