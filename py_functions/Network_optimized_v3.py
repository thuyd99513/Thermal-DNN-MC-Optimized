"""
Network_optimized_v3.py - S 和 J 更新的完全向量化 + Numba JIT 优化版本

本模块在 Network_optimized_v2.py 的基础上，增加了 J 更新的完全向量化优化：
1. S 更新：沿用 V2 版本的完全向量化 + Numba JIT
2. J 更新：新增完全向量化 + Numba JIT 实现

作者：Manus AI
日期：2026-01-27
"""

import numpy as np
from numba import njit, prange
from functools import wraps
from time import time
import os

# ============================================================================
# 计时装饰器
# ============================================================================

def timethis(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        # print(f'{func.__name__}: {(end - start) * 1000:.2f}ms')
        return result
    return wrapper


# ============================================================================
# Numba 加速的核心计算函数 (S 更新相关 - 沿用 V2)
# ============================================================================

@njit(cache=True, fastmath=True)
def soft_core_potential_numba(h):
    """
    软核势能函数 (Numba 加速版本)
    V(h) = h^2 if h < 0, else 0
    """
    result = np.zeros_like(h)
    for i in range(h.size):
        val = h.flat[i]
        if val < 0:
            result.flat[i] = val * val
    return result


@njit(cache=True, fastmath=True)
def calc_ener_numba(r):
    """计算能量 (Numba 加速版本)"""
    total = 0.0
    for i in range(r.size):
        val = r.flat[i]
        if val < 0:
            total += val * val
    return total


@njit(cache=True, fastmath=True)
def calc_ener_1d_numba(r):
    """
    计算 1D 数组的能量 (Numba 加速版本)
    
    Args:
        r: shape (M,) 的数组
    
    Returns:
        标量能量值
    """
    total = 0.0
    for i in range(r.shape[0]):
        val = r[i]
        if val < 0:
            total += val * val
    return total


@njit(cache=True, fastmath=True, parallel=True)
def calc_ener_2d_rows_numba(r):
    """
    计算 2D 数组每行的能量 (Numba 加速版本)
    
    Args:
        r: shape (N, M) 的数组
    
    Returns:
        shape (N,) 的能量数组
    """
    rows = r.shape[0]
    cols = r.shape[1]
    result = np.zeros(rows, dtype=np.float64)
    
    for i in prange(rows):
        total = 0.0
        for j in range(cols):
            val = r[i, j]
            if val < 0:
                total += val * val
        result[i] = total
    
    return result


@njit(cache=True, fastmath=True)
def calc_ener_3d_numba(r):
    """
    计算 3D 数组沿最后一个轴的能量 (Numba 加速版本)
    
    Args:
        r: shape (M, N, K) 的数组
    
    Returns:
        shape (M, N) 的能量数组
    """
    M = r.shape[0]
    N = r.shape[1]
    K = r.shape[2]
    result = np.zeros((M, N), dtype=np.float64)
    
    for m in range(M):
        for n in range(N):
            total = 0.0
            for k in range(K):
                val = r[m, n, k]
                if val < 0:
                    total += val * val
            result[m, n] = total
    
    return result


@njit(cache=True, fastmath=True, parallel=True)
def metropolis_accept_numba(delta_E, beta, EPS, rand_array):
    """
    Metropolis 接受准则 (Numba 加速版本)
    
    Args:
        delta_E: shape (M, N) 的能量差数组
        beta: 逆温度
        EPS: 容差阈值
        rand_array: shape (M, N) 的随机数数组
    
    Returns:
        shape (M, N) 的布尔接受掩码
    """
    M = delta_E.shape[0]
    N = delta_E.shape[1]
    accept = np.zeros((M, N), dtype=np.bool_)
    
    for m in prange(M):
        for n in range(N):
            dE = delta_E[m, n]
            if dE < EPS:
                accept[m, n] = True
            else:
                prob = np.exp(-dE * beta)
                if rand_array[m, n] < prob:
                    accept[m, n] = True
    
    return accept


@njit(cache=True, fastmath=True, parallel=True)
def metropolis_accept_1d_numba(delta_E, beta, EPS, rand_array):
    """
    Metropolis 接受准则 - 1D 版本 (Numba 加速版本)
    
    Args:
        delta_E: shape (N,) 的能量差数组
        beta: 逆温度
        EPS: 容差阈值
        rand_array: shape (N,) 的随机数数组
    
    Returns:
        shape (N,) 的布尔接受掩码
    """
    N = delta_E.shape[0]
    accept = np.zeros(N, dtype=np.bool_)
    
    for n in prange(N):
        dE = delta_E[n]
        if dE < EPS:
            accept[n] = True
        else:
            prob = np.exp(-dE * beta)
            if rand_array[n] < prob:
                accept[n] = True
    
    return accept


@njit(cache=True, fastmath=True, parallel=True)
def apply_spin_updates_numba(S_layer, accept_mask, delta_E):
    """
    应用自旋更新 (Numba 加速版本)
    
    Args:
        S_layer: shape (M, N) 的自旋数组 (会被原地修改)
        accept_mask: shape (M, N) 的布尔接受掩码
        delta_E: shape (M, N) 的能量差数组
    
    Returns:
        总能量变化
    """
    M = S_layer.shape[0]
    N = S_layer.shape[1]
    total_dE = 0.0
    
    for m in prange(M):
        for n in range(N):
            if accept_mask[m, n]:
                S_layer[m, n] = -S_layer[m, n]
                total_dE += delta_E[m, n]
    
    return total_dE


# ============================================================================
# Numba 加速的 S 更新 part_gap 计算函数 (沿用 V2)
# ============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_hidden_before_vectorized(J_hidden_lh, J_hidden_ls, S_lh, S_ls, S_ls_plus1, SQRT_N):
    """
    完全向量化计算中间层翻转前的 part_gap (Numba 加速版本)
    """
    M = S_ls.shape[0]
    N = S_ls.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            # part_gap[0]: 前一层到当前层的 gap
            dot_prev = 0.0
            for k in range(N):
                dot_prev += J_hidden_lh[n, k] * S_lh[m, k]
            part_gap[m, n, 0] = (dot_prev / SQRT_N) * S_ls[m, n]
            
            # part_gap[1:N+1]: 当前层到下一层的 N 个 gap
            for n2 in range(N):
                dot_next = 0.0
                for k in range(N):
                    dot_next += J_hidden_ls[n2, k] * S_ls[m, k]
                part_gap[m, n, 1 + n2] = (dot_next / SQRT_N) * S_ls_plus1[m, n2]
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_hidden_after_vectorized(J_hidden_lh, J_hidden_ls, S_lh, S_ls, S_ls_plus1, SQRT_N):
    """
    完全向量化计算中间层翻转后的 part_gap (Numba 加速版本)
    """
    M = S_ls.shape[0]
    N = S_ls.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            S_flipped = -S_ls[m, n]
            
            # part_gap[0]: 前一层到当前层的 gap (使用翻转后的值)
            dot_prev = 0.0
            for k in range(N):
                dot_prev += J_hidden_lh[n, k] * S_lh[m, k]
            part_gap[m, n, 0] = (dot_prev / SQRT_N) * S_flipped
            
            # part_gap[1:N+1]: 当前层到下一层的 N 个 gap (使用翻转后的 S_ls)
            for n2 in range(N):
                dot_next = 0.0
                for k in range(N):
                    if k == n:
                        dot_next += J_hidden_ls[n2, k] * S_flipped
                    else:
                        dot_next += J_hidden_ls[n2, k] * S_ls[m, k]
                part_gap[m, n, 1 + n2] = (dot_next / SQRT_N) * S_ls_plus1[m, n2]
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_in_before_vectorized(J_in, J_hidden_0, S_in, S_0, S_1, SQRT_N_IN, SQRT_N):
    """完全向量化计算第一层翻转前的 part_gap"""
    M = S_0.shape[0]
    N = S_0.shape[1]
    N_in = S_in.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            # part_gap[0]: 输入层到第一层的 gap
            dot_in = 0.0
            for k in range(N_in):
                dot_in += J_in[n, k] * S_in[m, k]
            part_gap[m, n, 0] = (dot_in / SQRT_N_IN) * S_0[m, n]
            
            # part_gap[1:N+1]: 第一层到第二层的 N 个 gap
            for n2 in range(N):
                dot_next = 0.0
                for k in range(N):
                    dot_next += J_hidden_0[n2, k] * S_0[m, k]
                part_gap[m, n, 1 + n2] = (dot_next / SQRT_N) * S_1[m, n2]
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_in_after_vectorized(J_in, J_hidden_0, S_in, S_0, S_1, SQRT_N_IN, SQRT_N):
    """完全向量化计算第一层翻转后的 part_gap"""
    M = S_0.shape[0]
    N = S_0.shape[1]
    N_in = S_in.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            S_flipped = -S_0[m, n]
            
            # part_gap[0]: 输入层到第一层的 gap
            dot_in = 0.0
            for k in range(N_in):
                dot_in += J_in[n, k] * S_in[m, k]
            part_gap[m, n, 0] = (dot_in / SQRT_N_IN) * S_flipped
            
            # part_gap[1:N+1]: 第一层到第二层的 N 个 gap
            for n2 in range(N):
                dot_next = 0.0
                for k in range(N):
                    if k == n:
                        dot_next += J_hidden_0[n2, k] * S_flipped
                    else:
                        dot_next += J_hidden_0[n2, k] * S_0[m, k]
                part_gap[m, n, 1 + n2] = (dot_next / SQRT_N) * S_1[m, n2]
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_out_before_vectorized(J_hidden_last, J_out, S_second_last, S_last, S_out, SQRT_N):
    """完全向量化计算最后层翻转前的 part_gap"""
    M = S_last.shape[0]
    N = S_last.shape[1]
    N_out = S_out.shape[1]
    part_gap = np.zeros((M, N, N_out + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            # part_gap[0]: 倒数第二层到最后层的 gap
            dot_prev = 0.0
            for k in range(N):
                dot_prev += J_hidden_last[n, k] * S_second_last[m, k]
            part_gap[m, n, 0] = (dot_prev / SQRT_N) * S_last[m, n]
            
            # part_gap[1:N_out+1]: 最后层到输出层的 N_out 个 gap
            for n2 in range(N_out):
                dot_out = 0.0
                for k in range(N):
                    dot_out += J_out[n2, k] * S_last[m, k]
                part_gap[m, n, 1 + n2] = (dot_out / SQRT_N) * S_out[m, n2]
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_out_after_vectorized(J_hidden_last, J_out, S_second_last, S_last, S_out, SQRT_N):
    """完全向量化计算最后层翻转后的 part_gap"""
    M = S_last.shape[0]
    N = S_last.shape[1]
    N_out = S_out.shape[1]
    part_gap = np.zeros((M, N, N_out + 1), dtype=np.float64)
    
    for m in prange(M):
        for n in range(N):
            S_flipped = -S_last[m, n]
            
            # part_gap[0]: 倒数第二层到最后层的 gap
            dot_prev = 0.0
            for k in range(N):
                dot_prev += J_hidden_last[n, k] * S_second_last[m, k]
            part_gap[m, n, 0] = (dot_prev / SQRT_N) * S_flipped
            
            # part_gap[1:N_out+1]: 最后层到输出层的 N_out 个 gap
            for n2 in range(N_out):
                dot_out = 0.0
                for k in range(N):
                    if k == n:
                        dot_out += J_out[n2, k] * S_flipped
                    else:
                        dot_out += J_out[n2, k] * S_last[m, k]
                part_gap[m, n, 1 + n2] = (dot_out / SQRT_N) * S_out[m, n2]
    
    return part_gap


# ============================================================================
# Numba 加速的 J 更新计算函数 (新增)
# ============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def compute_J_update_hidden_vectorized(J_hidden_l, S_l, S_l_plus1, n1_array, x_array, 
                                        RAT, RESCALE_J, SQRT_N, target_norm):
    """
    完全向量化计算隐藏层权重更新 (Numba 加速版本)
    
    对于第 l 层的权重矩阵 J_hidden[l]，一次性对所有 N 行进行更新尝试。
    
    Args:
        J_hidden_l: shape (N, N) 的权重矩阵
        S_l: shape (M, N) 的当前层自旋
        S_l_plus1: shape (M, N) 的下一层自旋
        n1_array: shape (N,) 每行要更新的列索引
        x_array: shape (N,) 每行的高斯扰动值
        RAT: 扰动比例
        RESCALE_J: 预缩放因子
        SQRT_N: sqrt(N)
        target_norm: 目标范数 (通常为 N)
    
    Returns:
        new_J: shape (N, N) 更新后的权重矩阵
        gap_before: shape (N, M) 更新前的 gap
        gap_after: shape (N, M) 更新后的 gap
    """
    N = J_hidden_l.shape[0]
    M = S_l.shape[0]
    
    # 创建新权重矩阵
    new_J = np.copy(J_hidden_l)
    
    # 1. 更新指定元素
    for n2 in prange(N):
        n1 = n1_array[n2]
        new_J[n2, n1] = (J_hidden_l[n2, n1] + x_array[n2] * RAT) * RESCALE_J
    
    # 2. 按行 rescale
    for n2 in prange(N):
        norm_sq = 0.0
        for k in range(N):
            norm_sq += new_J[n2, k] * new_J[n2, k]
        scale = np.sqrt(target_norm / norm_sq)
        for k in range(N):
            new_J[n2, k] *= scale
    
    # 3. 计算更新前后的 gap
    # gap[n2, mu] = (J[n2, :] @ S_l[mu, :] / SQRT_N) * S_l_plus1[mu, n2]
    gap_before = np.zeros((N, M), dtype=np.float64)
    gap_after = np.zeros((N, M), dtype=np.float64)
    
    for n2 in prange(N):
        for mu in range(M):
            dot_before = 0.0
            dot_after = 0.0
            for k in range(N):
                dot_before += J_hidden_l[n2, k] * S_l[mu, k]
                dot_after += new_J[n2, k] * S_l[mu, k]
            gap_before[n2, mu] = (dot_before / SQRT_N) * S_l_plus1[mu, n2]
            gap_after[n2, mu] = (dot_after / SQRT_N) * S_l_plus1[mu, n2]
    
    return new_J, gap_before, gap_after


@njit(cache=True, fastmath=True, parallel=True)
def compute_J_update_in_vectorized(J_in, S_in, S_0, n1_array, x_array,
                                    RAT, RESCALE_J, SQRT_N_IN, target_norm):
    """
    完全向量化计算输入层权重更新 (Numba 加速版本)
    
    Args:
        J_in: shape (N, N_in) 的输入权重矩阵
        S_in: shape (M, N_in) 的输入层自旋
        S_0: shape (M, N) 的第一隐藏层自旋
        n1_array: shape (N,) 每行要更新的列索引
        x_array: shape (N,) 每行的高斯扰动值
        RAT: 扰动比例
        RESCALE_J: 预缩放因子
        SQRT_N_IN: sqrt(N_in)
        target_norm: 目标范数 (通常为 N_in)
    
    Returns:
        new_J: shape (N, N_in) 更新后的权重矩阵
        gap_before: shape (N, M) 更新前的 gap
        gap_after: shape (N, M) 更新后的 gap
    """
    N = J_in.shape[0]
    N_in = J_in.shape[1]
    M = S_in.shape[0]
    
    # 创建新权重矩阵
    new_J = np.copy(J_in)
    
    # 1. 更新指定元素
    for n2 in prange(N):
        n1 = n1_array[n2]
        new_J[n2, n1] = (J_in[n2, n1] + x_array[n2] * RAT) * RESCALE_J
    
    # 2. 按行 rescale
    for n2 in prange(N):
        norm_sq = 0.0
        for k in range(N_in):
            norm_sq += new_J[n2, k] * new_J[n2, k]
        scale = np.sqrt(target_norm / norm_sq)
        for k in range(N_in):
            new_J[n2, k] *= scale
    
    # 3. 计算更新前后的 gap
    gap_before = np.zeros((N, M), dtype=np.float64)
    gap_after = np.zeros((N, M), dtype=np.float64)
    
    for n2 in prange(N):
        for mu in range(M):
            dot_before = 0.0
            dot_after = 0.0
            for k in range(N_in):
                dot_before += J_in[n2, k] * S_in[mu, k]
                dot_after += new_J[n2, k] * S_in[mu, k]
            gap_before[n2, mu] = (dot_before / SQRT_N_IN) * S_0[mu, n2]
            gap_after[n2, mu] = (dot_after / SQRT_N_IN) * S_0[mu, n2]
    
    return new_J, gap_before, gap_after


@njit(cache=True, fastmath=True, parallel=True)
def compute_J_update_out_vectorized(J_out, S_last, S_out, n1_array, x_array,
                                     RAT, RESCALE_J, SQRT_N, target_norm):
    """
    完全向量化计算输出层权重更新 (Numba 加速版本)
    
    Args:
        J_out: shape (N_out, N) 的输出权重矩阵
        S_last: shape (M, N) 的最后隐藏层自旋
        S_out: shape (M, N_out) 的输出层自旋
        n1_array: shape (N_out,) 每行要更新的列索引
        x_array: shape (N_out,) 每行的高斯扰动值
        RAT: 扰动比例
        RESCALE_J: 预缩放因子
        SQRT_N: sqrt(N)
        target_norm: 目标范数 (通常为 N)
    
    Returns:
        new_J: shape (N_out, N) 更新后的权重矩阵
        gap_before: shape (N_out, M) 更新前的 gap
        gap_after: shape (N_out, M) 更新后的 gap
    """
    N_out = J_out.shape[0]
    N = J_out.shape[1]
    M = S_last.shape[0]
    
    # 创建新权重矩阵
    new_J = np.copy(J_out)
    
    # 1. 更新指定元素
    for n2 in prange(N_out):
        n1 = n1_array[n2]
        new_J[n2, n1] = (J_out[n2, n1] + x_array[n2] * RAT) * RESCALE_J
    
    # 2. 按行 rescale
    for n2 in prange(N_out):
        norm_sq = 0.0
        for k in range(N):
            norm_sq += new_J[n2, k] * new_J[n2, k]
        scale = np.sqrt(target_norm / norm_sq)
        for k in range(N):
            new_J[n2, k] *= scale
    
    # 3. 计算更新前后的 gap
    gap_before = np.zeros((N_out, M), dtype=np.float64)
    gap_after = np.zeros((N_out, M), dtype=np.float64)
    
    for n2 in prange(N_out):
        for mu in range(M):
            dot_before = 0.0
            dot_after = 0.0
            for k in range(N):
                dot_before += J_out[n2, k] * S_last[mu, k]
                dot_after += new_J[n2, k] * S_last[mu, k]
            gap_before[n2, mu] = (dot_before / SQRT_N) * S_out[mu, n2]
            gap_after[n2, mu] = (dot_after / SQRT_N) * S_out[mu, n2]
    
    return new_J, gap_before, gap_after


@njit(cache=True, fastmath=True, parallel=True)
def apply_J_updates_numba(J, new_J, accept_mask, delta_E):
    """
    应用权重更新 (Numba 加速版本)
    
    Args:
        J: shape (N, K) 的权重矩阵 (会被原地修改)
        new_J: shape (N, K) 的新权重矩阵
        accept_mask: shape (N,) 的布尔接受掩码
        delta_E: shape (N,) 的能量差数组
    
    Returns:
        总能量变化
    """
    N = J.shape[0]
    K = J.shape[1]
    total_dE = 0.0
    
    for n in prange(N):
        if accept_mask[n]:
            for k in range(K):
                J[n, k] = new_J[n, k]
            total_dE += delta_E[n]
    
    return total_dE


# ============================================================================
# 优化后的网络类 V3
# ============================================================================

class NetworkOptimizedV3:
    """
    S 和 J 更新的完全向量化 + Numba JIT 优化的神经网络 MC 模拟类
    """
    
    def __init__(self, M, N, L, N_in=784, N_out=2, beta=66.7, seed=42):
        """初始化网络"""
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
        
        # 初始化网络状态
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float64)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float64)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float64)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float64)
        self.J_in = np.random.randn(N, N_in).astype(np.float64)
        self.J_out = np.random.randn(N_out, N).astype(np.float64)
        
        # 归一化权重矩阵
        self._normalize_weights()
        
        self.H = 0.0
        self.H_history = []
    
    def _normalize_weights(self):
        """归一化所有权重矩阵"""
        # J_in: 每行范数为 sqrt(N_in)
        for n in range(self.N):
            norm = np.sqrt(np.sum(self.J_in[n] ** 2))
            self.J_in[n] *= np.sqrt(self.N_in) / norm
        
        # J_hidden: 每行范数为 sqrt(N)
        for l in range(self.num_hidden_bond_layers):
            for n in range(self.N):
                norm = np.sqrt(np.sum(self.J_hidden[l, n] ** 2))
                self.J_hidden[l, n] *= np.sqrt(self.N) / norm
        
        # J_out: 每行范数为 sqrt(N)
        for n in range(self.N_out):
            norm = np.sqrt(np.sum(self.J_out[n] ** 2))
            self.J_out[n] *= np.sqrt(self.N) / norm
    
    def copy_state_from(self, other):
        """从另一个网络复制状态"""
        self.S = other.S.copy()
        self.S_in = other.S_in.copy()
        self.S_out = other.S_out.copy()
        self.J_hidden = other.J_hidden.copy()
        self.J_in = other.J_in.copy()
        self.J_out = other.J_out.copy()
        self.H = other.H
    
    # ========================================================================
    # S 更新方法 (沿用 V2)
    # ========================================================================
    
    @timethis
    def update_S_first_layer_vectorized(self):
        """向量化更新第一隐藏层的所有自旋"""
        part_gap_before = compute_part_gap_in_before_vectorized(
            self.J_in, self.J_hidden[0], 
            self.S_in, self.S[:, 0, :], self.S[:, 1, :],
            self.SQRT_N_IN, self.SQRT_N
        )
        part_gap_after = compute_part_gap_in_after_vectorized(
            self.J_in, self.J_hidden[0],
            self.S_in, self.S[:, 0, :], self.S[:, 1, :],
            self.SQRT_N_IN, self.SQRT_N
        )
        
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((self.M, self.N))
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        total_dE = apply_spin_updates_numba(self.S[:, 0, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_S_middle_layer_vectorized(self, l_s):
        """向量化更新中间隐藏层的所有自旋"""
        l_h = l_s - 1
        
        part_gap_before = compute_part_gap_hidden_before_vectorized(
            self.J_hidden[l_h], self.J_hidden[l_s],
            self.S[:, l_h, :], self.S[:, l_s, :], self.S[:, l_s + 1, :],
            self.SQRT_N
        )
        part_gap_after = compute_part_gap_hidden_after_vectorized(
            self.J_hidden[l_h], self.J_hidden[l_s],
            self.S[:, l_h, :], self.S[:, l_s, :], self.S[:, l_s + 1, :],
            self.SQRT_N
        )
        
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((self.M, self.N))
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        total_dE = apply_spin_updates_numba(self.S[:, l_s, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_S_last_layer_vectorized(self):
        """向量化更新最后隐藏层的所有自旋"""
        l_s = self.num_hidden_node_layers - 1
        
        part_gap_before = compute_part_gap_out_before_vectorized(
            self.J_hidden[-1], self.J_out,
            self.S[:, -2, :], self.S[:, -1, :], self.S_out,
            self.SQRT_N
        )
        part_gap_after = compute_part_gap_out_after_vectorized(
            self.J_hidden[-1], self.J_out,
            self.S[:, -2, :], self.S[:, -1, :], self.S_out,
            self.SQRT_N
        )
        
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((self.M, self.N))
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        total_dE = apply_spin_updates_numba(self.S[:, l_s, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_all_S_vectorized(self):
        """更新所有隐藏层的自旋"""
        self.update_S_first_layer_vectorized()
        
        for l_s in range(1, self.num_hidden_node_layers - 1):
            self.update_S_middle_layer_vectorized(l_s)
        
        if self.num_hidden_node_layers > 1:
            self.update_S_last_layer_vectorized()
    
    # ========================================================================
    # J 更新方法 (新增)
    # ========================================================================
    
    @timethis
    def update_J_in_vectorized(self):
        """向量化更新输入层权重"""
        N, N_in = self.N, self.N_in
        
        # 为每行随机选择一个列和扰动值
        n1_array = np.random.randint(0, N_in, size=N).astype(np.int64)
        x_array = np.random.normal(size=N)
        
        # 计算更新
        new_J, gap_before, gap_after = compute_J_update_in_vectorized(
            self.J_in, self.S_in, self.S[:, 0, :],
            n1_array, x_array,
            self.RAT, self.RESCALE_J, self.SQRT_N_IN, float(N_in)
        )
        
        # 计算能量差
        E_before = calc_ener_2d_rows_numba(gap_before)
        E_after = calc_ener_2d_rows_numba(gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random(N)
        accept_mask = metropolis_accept_1d_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_J_updates_numba(self.J_in, new_J, accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_J_hidden_layer_vectorized(self, l):
        """向量化更新第 l 层隐藏权重"""
        N = self.N
        
        # 为每行随机选择一个列和扰动值
        n1_array = np.random.randint(0, N, size=N).astype(np.int64)
        x_array = np.random.normal(size=N)
        
        # 计算更新
        new_J, gap_before, gap_after = compute_J_update_hidden_vectorized(
            self.J_hidden[l], self.S[:, l, :], self.S[:, l + 1, :],
            n1_array, x_array,
            self.RAT, self.RESCALE_J, self.SQRT_N, float(N)
        )
        
        # 计算能量差
        E_before = calc_ener_2d_rows_numba(gap_before)
        E_after = calc_ener_2d_rows_numba(gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random(N)
        accept_mask = metropolis_accept_1d_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_J_updates_numba(self.J_hidden[l], new_J, accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_J_out_vectorized(self):
        """向量化更新输出层权重"""
        N, N_out = self.N, self.N_out
        
        # 为每行随机选择一个列和扰动值
        n1_array = np.random.randint(0, N, size=N_out).astype(np.int64)
        x_array = np.random.normal(size=N_out)
        
        # 计算更新
        new_J, gap_before, gap_after = compute_J_update_out_vectorized(
            self.J_out, self.S[:, -1, :], self.S_out,
            n1_array, x_array,
            self.RAT, self.RESCALE_J, self.SQRT_N, float(N)
        )
        
        # 计算能量差
        E_before = calc_ener_2d_rows_numba(gap_before)
        E_after = calc_ener_2d_rows_numba(gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random(N_out)
        accept_mask = metropolis_accept_1d_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_J_updates_numba(self.J_out, new_J, accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_all_J_vectorized(self):
        """更新所有权重"""
        self.update_J_in_vectorized()
        
        for l in range(self.num_hidden_bond_layers):
            self.update_J_hidden_layer_vectorized(l)
        
        self.update_J_out_vectorized()
    
    # ========================================================================
    # 完整 MC 步
    # ========================================================================
    
    def _compute_update_ratio(self):
        """
        计算 S 和 J 更新的比例参数
        
        原始实现中，每次随机选择一个变量（S 或 J）更新，概率与变量数量成正比：
        - P(更新S) = num_variables / num = num_S / (num_S + num_J)
        - P(更新J) = num_bonds / num = num_J / (num_S + num_J)
        
        向量化后，每次 S sweep 更新 S_per_sweep 个 S，每次 J sweep 更新 J_per_sweep 个 J。
        为保持原始比例，需要计算每次 S sweep 对应多少次 J sweep。
        """
        # 原始变量数量
        num_S = self.N * self.M * self.num_hidden_node_layers  # S 总数
        num_J = (self.N * self.N_in +                          # J_in
                 self.N * self.N * self.num_hidden_bond_layers + # J_hidden
                 self.N_out * self.N)                           # J_out
        
        # 每次 sweep 更新的数量
        # S sweep: 更新所有隐藏层的 S（但边界层在原始实现中也会被更新）
        S_per_sweep = self.M * self.N * self.num_hidden_node_layers
        
        # J sweep: 每层每行更新一个元素
        J_per_sweep = (self.N +                                 # J_in: N 行
                       self.N * self.num_hidden_bond_layers +   # J_hidden: (L-2)*N 行
                       self.N_out)                              # J_out: N_out 行
        
        # 原始比例: num_S : num_J
        # 向量化后每步: S_per_sweep : J_per_sweep * J_sweeps_per_S_sweep
        # 要保持比例: S_per_sweep / (J_per_sweep * J_sweeps) = num_S / num_J
        # 解得: J_sweeps = (S_per_sweep * num_J) / (J_per_sweep * num_S)
        
        J_sweeps_per_S_sweep = (S_per_sweep * num_J) / (J_per_sweep * num_S)
        
        return J_sweeps_per_S_sweep, num_S, num_J, S_per_sweep, J_per_sweep
    
    @timethis
    def mc_step_vectorized(self):
        """
        执行一个完整的 MC 步（更新 S 和 J）
        
        注意：此方法保持原始实现中 S 和 J 的更新比例。
        原始实现中 P(S) : P(J) = num_S : num_J ≈ 19:1
        """
        self.update_all_S_vectorized()
        self.update_all_J_vectorized()
    
    @timethis
    def mc_step_vectorized_balanced(self):
        """
        执行一个平衡的 MC 步（保持原始 S/J 更新比例）
        
        通过调整 J 更新的频率来匹配原始实现中的 S:J 比例。
        """
        J_sweeps_ratio, num_S, num_J, S_per_sweep, J_per_sweep = self._compute_update_ratio()
        
        # 更新 S
        self.update_all_S_vectorized()
        
        # 根据比例更新 J
        # 使用累积方式处理非整数比例
        if not hasattr(self, '_J_sweep_accumulator'):
            self._J_sweep_accumulator = 0.0
        
        self._J_sweep_accumulator += J_sweeps_ratio
        
        while self._J_sweep_accumulator >= 1.0:
            self.update_all_J_vectorized()
            self._J_sweep_accumulator -= 1.0
    
    def mc_main_vectorized(self, num_steps=100, verbose=False, balanced=True):
        """
        运行多个 MC 步
        
        Args:
            num_steps: MC 步数
            verbose: 是否打印进度
            balanced: 是否使用平衡模式（保持原始 S/J 比例）
        """
        # 打印比例信息
        if verbose:
            J_sweeps_ratio, num_S, num_J, S_per_sweep, J_per_sweep = self._compute_update_ratio()
            print(f"S/J 更新比例信息:")
            print(f"  num_S = {num_S}, num_J = {num_J}")
            print(f"  原始比例 num_S:num_J = {num_S/num_J:.2f}:1")
            print(f"  S_per_sweep = {S_per_sweep}, J_per_sweep = {J_per_sweep}")
            print(f"  J_sweeps_per_S_sweep = {J_sweeps_ratio:.4f}")
            print(f"  balanced mode = {balanced}")
        
        for step in range(num_steps):
            if balanced:
                self.mc_step_vectorized_balanced()
            else:
                self.mc_step_vectorized()
            
            self.H_history.append(self.H)
            if verbose and (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{num_steps}, H = {self.H:.4f}")


# ============================================================================
# 用于验证的原始实现
# ============================================================================

class NetworkOriginalReference:
    """
    原始实现的参考版本 - 用于验证优化版本的正确性
    """
    
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
        for n in range(N):
            norm = np.sqrt(np.sum(self.J_in[n] ** 2))
            self.J_in[n] *= np.sqrt(N_in) / norm
        for l in range(L-2):
            for n in range(N):
                norm = np.sqrt(np.sum(self.J_hidden[l, n] ** 2))
                self.J_hidden[l, n] *= np.sqrt(N) / norm
        for n in range(N_out):
            norm = np.sqrt(np.sum(self.J_out[n] ** 2))
            self.J_out[n] *= np.sqrt(N) / norm
        
        self.H = 0.0
    
    def soft_core_potential(self, h):
        """软核势能函数"""
        return np.where(h < 0, h ** 2, 0.0)
    
    def calc_ener(self, r):
        """计算能量"""
        return np.sum(self.soft_core_potential(r))
    
    def part_gap_hidden_shift(self, l, n2, J_row):
        """计算隐藏层权重更新的 gap"""
        gap = (J_row @ self.S[:, l, :].T / self.SQRT_N) * self.S[:, l + 1, n2]
        return gap
    
    def update_J_hidden_serial(self, l, n2, n1, x):
        """串行更新单个隐藏层权重"""
        # 创建新权重行
        new_J_row = self.J_hidden[l, n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * self.RAT) * self.RESCALE_J
        
        # Rescale
        norm = np.sqrt(np.sum(new_J_row ** 2))
        new_J_row *= np.sqrt(self.N) / norm
        
        # 计算能量差
        gap_before = self.part_gap_hidden_shift(l, n2, self.J_hidden[l, n2])
        gap_after = self.part_gap_hidden_shift(l, n2, new_J_row)
        delta_E = self.calc_ener(gap_after) - self.calc_ener(gap_before)
        
        # Metropolis
        rand = np.random.random()
        if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
            self.J_hidden[l, n2] = new_J_row
            self.H += delta_E
            return True
        return False


# ============================================================================
# JIT 预热和测试
# ============================================================================

def warmup_jit():
    """预热 JIT 编译"""
    print("预热 JIT 编译...")
    net = NetworkOptimizedV3(10, 3, 5, seed=0)
    for _ in range(3):
        net.mc_step_vectorized()
    print("JIT 预热完成")


if __name__ == "__main__":
    warmup_jit()
    
    print("\n测试 V3 优化版本 (S + J 更新)...")
    net = NetworkOptimizedV3(120, 3, 10)
    
    import time
    start = time.time()
    for _ in range(10):
        net.mc_step_vectorized()
    end = time.time()
    
    print(f"10 个完整 MC 步耗时: {end - start:.4f}s")
    print(f"每步平均耗时: {(end - start) / 10 * 1000:.2f}ms")
