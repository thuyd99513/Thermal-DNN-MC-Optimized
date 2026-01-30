#!/usr/bin/env python3
"""
benchmark_S_J_speedup.py - S 和 J 更新优化加速比对比测试

本脚本对比 S（自旋）和 J（权重）更新在优化前后的加速比率：

1. S 更新：
   - 原始：串行逐个自旋更新
   - 优化：向量化 + Numba JIT 并行

2. J 更新：
   - 原始：串行逐个权重更新
   - 优化：向量化 + Numba JIT 并行

测试环境信息会自动从 SLURM 环境变量获取。

作者：Manus AI
日期：2026-01-30
"""

import numpy as np
import time
import sys
import os
import json
import platform
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# 获取环境信息
# ============================================================================

def get_environment_info():
    """获取运行环境信息"""
    info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
    }
    
    # 尝试获取 SLURM 环境变量
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_CPUS_PER_TASK', 
                  'SLURM_NNODES', 'SLURM_NTASKS', 'SLURM_JOB_PARTITION',
                  'HOSTNAME', 'SLURM_NODELIST']
    
    info['slurm'] = {}
    for var in slurm_vars:
        val = os.environ.get(var)
        if val:
            info['slurm'][var] = val
    
    # 尝试获取 NumPy 配置
    try:
        info['numpy_version'] = np.__version__
        info['numpy_config'] = str(np.show_config())
    except:
        pass
    
    # 尝试获取 Numba 信息
    try:
        import numba
        info['numba_version'] = numba.__version__
    except:
        info['numba_version'] = 'Not installed'
    
    return info


# ============================================================================
# 原始串行实现
# ============================================================================

def soft_core_potential(h):
    """软核势能函数"""
    return np.where(h < 0, h ** 2, 0.0)

def calc_ener(r):
    """计算能量"""
    return np.sum(soft_core_potential(r))


class NetworkSerial:
    """
    原始串行实现 - 逐个更新 S 和 J
    """
    
    def __init__(self, M, N, L, N_in=784, N_out=2, beta=1e5, seed=42):
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
        """归一化权重"""
        for n in range(self.N):
            norm = np.sqrt(np.sum(self.J_in[n] ** 2))
            if norm > 0:
                self.J_in[n] *= np.sqrt(self.N_in) / norm
        for l in range(self.num_hidden_bond_layers):
            for n in range(self.N):
                norm = np.sqrt(np.sum(self.J_hidden[l, n] ** 2))
                if norm > 0:
                    self.J_hidden[l, n] *= np.sqrt(self.N) / norm
        for n in range(self.N_out):
            norm = np.sqrt(np.sum(self.J_out[n] ** 2))
            if norm > 0:
                self.J_out[n] *= np.sqrt(self.N) / norm
    
    # -------------------------------------------------------------------------
    # S 更新 (串行)
    # -------------------------------------------------------------------------
    
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        """计算中间层自旋翻转前的 part_gap"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float64)
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
        """计算中间层自旋翻转后的 part_gap"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float64)
        
        S_flipped = -self.S[mu, l_s, n]
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * S_flipped
        
        S_layer_copy = self.S[mu, l_s, :].copy()
        S_layer_copy[n] = S_flipped
        J_hidden_next = self.J_hidden[l_s, :, :] @ S_layer_copy
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def update_S_serial(self, num_updates):
        """串行更新自旋"""
        for _ in range(num_updates):
            mu = np.random.randint(0, self.M)
            l_s = np.random.randint(1, self.num_hidden_node_layers - 1)
            n = np.random.randint(0, self.N)
            
            part_gap_before = self.part_gap_hidden_before_flip(mu, l_s, n)
            part_gap_after = self.part_gap_hidden_after_flip(mu, l_s, n)
            
            delta_E = calc_ener(part_gap_after) - calc_ener(part_gap_before)
            
            if delta_E < self.EPS or np.random.random() < np.exp(-delta_E * self.beta):
                self.S[mu, l_s, n] = -self.S[mu, l_s, n]
                self.H += delta_E
    
    # -------------------------------------------------------------------------
    # J 更新 (串行)
    # -------------------------------------------------------------------------
    
    def part_gap_hidden_shift(self, l, n2, J_row):
        """计算隐藏层 J 更新的 gap"""
        gap = (J_row @ self.S[:, l, :].T / self.SQRT_N) * self.S[:, l + 1, n2]
        return gap
    
    def part_gap_in_shift(self, n2, J_row):
        """计算输入层 J 更新的 gap"""
        gap = (J_row @ self.S_in.T / self.SQRT_N_IN) * self.S[:, 0, n2]
        return gap
    
    def part_gap_out_shift(self, n2, J_row):
        """计算输出层 J 更新的 gap"""
        gap = (J_row @ self.S[:, -1, :].T / self.SQRT_N) * self.S_out[:, n2]
        return gap
    
    def update_J_hidden_serial(self, l, n2, n1, x):
        """串行更新单个隐藏层权重"""
        new_J_row = self.J_hidden[l, n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
        
        norm = np.sqrt(np.sum(new_J_row ** 2))
        if norm > 0:
            new_J_row *= np.sqrt(self.N) / norm
        
        gap_before = self.part_gap_hidden_shift(l, n2, self.J_hidden[l, n2])
        gap_after = self.part_gap_hidden_shift(l, n2, new_J_row)
        delta_E = calc_ener(gap_after) - calc_ener(gap_before)
        
        if delta_E < self.EPS or np.random.random() < np.exp(-delta_E * self.beta):
            self.J_hidden[l, n2] = new_J_row
            self.H += delta_E
    
    def update_all_J_serial(self):
        """串行更新所有权重（每行更新一次）"""
        # 更新 J_in
        for n2 in range(self.N):
            n1 = np.random.randint(0, self.N_in)
            x = np.random.normal()
            new_J_row = self.J_in[n2].copy()
            new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
            norm = np.sqrt(np.sum(new_J_row ** 2))
            if norm > 0:
                new_J_row *= np.sqrt(self.N_in) / norm
            gap_before = self.part_gap_in_shift(n2, self.J_in[n2])
            gap_after = self.part_gap_in_shift(n2, new_J_row)
            delta_E = calc_ener(gap_after) - calc_ener(gap_before)
            if delta_E < self.EPS or np.random.random() < np.exp(-delta_E * self.beta):
                self.J_in[n2] = new_J_row
                self.H += delta_E
        
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
            new_J_row = self.J_out[n2].copy()
            new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
            norm = np.sqrt(np.sum(new_J_row ** 2))
            if norm > 0:
                new_J_row *= np.sqrt(self.N) / norm
            gap_before = self.part_gap_out_shift(n2, self.J_out[n2])
            gap_after = self.part_gap_out_shift(n2, new_J_row)
            delta_E = calc_ener(gap_after) - calc_ener(gap_before)
            if delta_E < self.EPS or np.random.random() < np.exp(-delta_E * self.beta):
                self.J_out[n2] = new_J_row
                self.H += delta_E


# ============================================================================
# 基准测试函数
# ============================================================================

def benchmark_S_update(M, N, L, num_updates, num_trials=3):
    """
    对比 S 更新的串行和优化实现
    
    Args:
        M: 样本数
        N: 每层神经元数
        L: 网络层数
        num_updates: 每次测试的更新次数
        num_trials: 重复测试次数
    
    Returns:
        dict: 包含测试结果
    """
    print(f"\n{'='*60}")
    print(f"S 更新基准测试: M={M}, N={N}, L={L}, updates={num_updates}")
    print(f"{'='*60}")
    
    # 测试串行实现
    print("\n测试串行实现...")
    serial_times = []
    for trial in range(num_trials):
        net = NetworkSerial(M, N, L, seed=42+trial)
        start = time.time()
        net.update_S_serial(num_updates)
        elapsed = time.time() - start
        serial_times.append(elapsed)
        print(f"  Trial {trial+1}: {elapsed:.4f}s")
    
    serial_mean = np.mean(serial_times)
    serial_std = np.std(serial_times)
    print(f"  平均: {serial_mean:.4f}s ± {serial_std:.4f}s")
    
    # 测试优化实现
    print("\n测试优化实现 (向量化 + Numba JIT)...")
    try:
        from Network_optimized_v3 import NetworkOptimizedV3, warmup_jit
        
        # 预热 JIT
        print("  预热 JIT 编译...")
        warmup_jit(M=min(M, 100), N=min(N, 5), L=L)
        
        optimized_times = []
        for trial in range(num_trials):
            net = NetworkOptimizedV3(M, N, L, seed=42+trial)
            
            # 计算等效的 MC sweeps
            # 每个 sweep 更新所有中间层的所有自旋
            middle_layers = L - 3
            spins_per_layer = M * N
            total_spins = middle_layers * spins_per_layer
            num_sweeps = max(1, num_updates // total_spins)
            
            start = time.time()
            for _ in range(num_sweeps):
                net.update_all_S_vectorized()
            elapsed = time.time() - start
            optimized_times.append(elapsed)
            print(f"  Trial {trial+1}: {elapsed:.4f}s ({num_sweeps} sweeps)")
        
        optimized_mean = np.mean(optimized_times)
        optimized_std = np.std(optimized_times)
        print(f"  平均: {optimized_mean:.4f}s ± {optimized_std:.4f}s")
        
        speedup = serial_mean / optimized_mean if optimized_mean > 0 else float('inf')
        print(f"\n  >>> S 更新加速比: {speedup:.2f}x <<<")
        
    except ImportError as e:
        print(f"  无法导入优化模块: {e}")
        optimized_mean = None
        optimized_std = None
        speedup = None
    
    return {
        'type': 'S_update',
        'config': {'M': M, 'N': N, 'L': L, 'num_updates': num_updates},
        'serial': {'mean': serial_mean, 'std': serial_std, 'times': serial_times},
        'optimized': {'mean': optimized_mean, 'std': optimized_std, 
                      'times': optimized_times if 'optimized_times' in dir() else None},
        'speedup': speedup
    }


def benchmark_J_update(M, N, L, num_sweeps, num_trials=3):
    """
    对比 J 更新的串行和优化实现
    
    Args:
        M: 样本数
        N: 每层神经元数
        L: 网络层数
        num_sweeps: MC sweeps 数量
        num_trials: 重复测试次数
    
    Returns:
        dict: 包含测试结果
    """
    print(f"\n{'='*60}")
    print(f"J 更新基准测试: M={M}, N={N}, L={L}, sweeps={num_sweeps}")
    print(f"{'='*60}")
    
    # 测试串行实现
    print("\n测试串行实现...")
    serial_times = []
    for trial in range(num_trials):
        net = NetworkSerial(M, N, L, seed=42+trial)
        start = time.time()
        for _ in range(num_sweeps):
            net.update_all_J_serial()
        elapsed = time.time() - start
        serial_times.append(elapsed)
        print(f"  Trial {trial+1}: {elapsed:.4f}s")
    
    serial_mean = np.mean(serial_times)
    serial_std = np.std(serial_times)
    print(f"  平均: {serial_mean:.4f}s ± {serial_std:.4f}s")
    
    # 测试优化实现
    print("\n测试优化实现 (向量化 + Numba JIT)...")
    try:
        from Network_optimized_v3 import NetworkOptimizedV3, warmup_jit
        
        # 预热 JIT
        print("  预热 JIT 编译...")
        warmup_jit(M=min(M, 100), N=min(N, 5), L=L)
        
        optimized_times = []
        for trial in range(num_trials):
            net = NetworkOptimizedV3(M, N, L, seed=42+trial)
            start = time.time()
            for _ in range(num_sweeps):
                net.update_all_J_vectorized()
            elapsed = time.time() - start
            optimized_times.append(elapsed)
            print(f"  Trial {trial+1}: {elapsed:.4f}s")
        
        optimized_mean = np.mean(optimized_times)
        optimized_std = np.std(optimized_times)
        print(f"  平均: {optimized_mean:.4f}s ± {optimized_std:.4f}s")
        
        speedup = serial_mean / optimized_mean if optimized_mean > 0 else float('inf')
        print(f"\n  >>> J 更新加速比: {speedup:.2f}x <<<")
        
    except ImportError as e:
        print(f"  无法导入优化模块: {e}")
        optimized_mean = None
        optimized_std = None
        speedup = None
    
    return {
        'type': 'J_update',
        'config': {'M': M, 'N': N, 'L': L, 'num_sweeps': num_sweeps},
        'serial': {'mean': serial_mean, 'std': serial_std, 'times': serial_times},
        'optimized': {'mean': optimized_mean, 'std': optimized_std,
                      'times': optimized_times if 'optimized_times' in dir() else None},
        'speedup': speedup
    }


def run_full_benchmark():
    """运行完整的基准测试"""
    
    print("="*70)
    print("S 和 J 更新优化加速比对比测试")
    print("="*70)
    
    # 获取环境信息
    env_info = get_environment_info()
    print(f"\n运行环境:")
    print(f"  平台: {env_info['platform']}")
    print(f"  Python: {env_info['python_version']}")
    print(f"  CPU 核心数: {env_info['cpu_count']}")
    print(f"  NumPy: {env_info.get('numpy_version', 'Unknown')}")
    print(f"  Numba: {env_info.get('numba_version', 'Unknown')}")
    
    if env_info['slurm']:
        print(f"\nSLURM 环境:")
        for k, v in env_info['slurm'].items():
            print(f"  {k}: {v}")
    
    # 测试配置
    test_configs = [
        # 小规模测试
        {'M': 100, 'N': 5, 'L': 10, 'S_updates': 5000, 'J_sweeps': 10},
        # 中等规模测试
        {'M': 500, 'N': 10, 'L': 10, 'S_updates': 10000, 'J_sweeps': 5},
        # 论文规模测试 (缩小版)
        {'M': 1000, 'N': 20, 'L': 10, 'S_updates': 20000, 'J_sweeps': 3},
    ]
    
    results = {
        'environment': env_info,
        'tests': []
    }
    
    for config in test_configs:
        M, N, L = config['M'], config['N'], config['L']
        
        # S 更新测试
        s_result = benchmark_S_update(M, N, L, config['S_updates'], num_trials=3)
        results['tests'].append(s_result)
        
        # J 更新测试
        j_result = benchmark_J_update(M, N, L, config['J_sweeps'], num_trials=3)
        results['tests'].append(j_result)
    
    # 生成汇总报告
    print("\n" + "="*70)
    print("汇总报告")
    print("="*70)
    
    print("\n{:<15} {:<10} {:<10} {:<10} {:<12} {:<12} {:<10}".format(
        "类型", "M", "N", "L", "串行(s)", "优化(s)", "加速比"))
    print("-"*70)
    
    for test in results['tests']:
        config = test['config']
        serial_time = test['serial']['mean']
        opt_time = test['optimized']['mean'] if test['optimized']['mean'] else 'N/A'
        speedup = test['speedup'] if test['speedup'] else 'N/A'
        
        if isinstance(opt_time, float):
            opt_str = f"{opt_time:.4f}"
            speedup_str = f"{speedup:.2f}x"
        else:
            opt_str = opt_time
            speedup_str = speedup
        
        print("{:<15} {:<10} {:<10} {:<10} {:<12.4f} {:<12} {:<10}".format(
            test['type'], config['M'], config['N'], config['L'],
            serial_time, opt_str, speedup_str))
    
    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, 'benchmark_S_J_results.json')
    
    # 转换 numpy 类型为 Python 原生类型
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    results_native = convert_to_native(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_native, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


# ============================================================================
# SLURM 提交脚本生成
# ============================================================================

def generate_slurm_script():
    """生成 SLURM 提交脚本"""
    script = '''#!/bin/bash
#SBATCH -p vip_33
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -J benchmark_SJ
#SBATCH -o logs/benchmark_SJ_%j.out
#SBATCH -e logs/benchmark_SJ_%j.err
#SBATCH --time=01:00:00

module purge
module load miniforge/24.11
source activate nnsm

mkdir -p logs

echo "============================================"
echo "S 和 J 更新优化加速比对比测试"
echo "开始时间: $(date)"
echo "节点: $HOSTNAME"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo "============================================"

cd py_functions
python3 benchmark_S_J_speedup.py

echo "============================================"
echo "测试完成"
echo "结束时间: $(date)"
echo "============================================"
'''
    
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               '..', 'submit_benchmark_SJ.sh')
    with open(script_path, 'w') as f:
        f.write(script)
    
    print(f"SLURM 脚本已生成: {script_path}")
    print("提交命令: sbatch submit_benchmark_SJ.sh")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='S 和 J 更新优化加速比对比测试')
    parser.add_argument('--generate-slurm', action='store_true',
                        help='生成 SLURM 提交脚本')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式（较小规模）')
    
    args = parser.parse_args()
    
    if args.generate_slurm:
        generate_slurm_script()
    else:
        run_full_benchmark()
