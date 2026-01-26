"""
Network_optimized_v2.py - 完全向量化 + Numba JIT 优化版本

本模块在 Network_optimized.py 的基础上进行进一步优化：
1. 消除 part_gap_*_after_flip_layer 函数中的嵌套循环，实现完全向量化
2. 应用 Numba JIT 编译加速计算密集型函数

作者：Manus AI
日期：2026-01-26
"""

import numpy as np
from numba import njit, prange
from functools import wraps
from time import time
import os

# ============================================================================
# Numba 加速的核心计算函数
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
def calc_ener_2d_numba(r):
    """
    计算 2D 数组每行的能量 (Numba 加速版本)
    
    Args:
        r: shape (M, N, K) 或 (M*N, K) 的数组
    
    Returns:
        shape (M, N) 或 (M*N,) 的能量数组
    """
    rows = r.shape[0]
    cols = r.shape[1]
    result = np.zeros(rows, dtype=np.float32)
    
    for i in range(rows):
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
    result = np.zeros((M, N), dtype=np.float32)
    
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
def compute_part_gap_hidden_before_vectorized(J_hidden_lh, J_hidden_ls, S_lh, S_ls, S_ls_plus1, SQRT_N):
    """
    完全向量化计算中间层翻转前的 part_gap (Numba 加速版本)
    
    Args:
        J_hidden_lh: shape (N, N) - 前一层的权重矩阵
        J_hidden_ls: shape (N, N) - 当前层的权重矩阵
        S_lh: shape (M, N) - 前一层的自旋
        S_ls: shape (M, N) - 当前层的自旋
        S_ls_plus1: shape (M, N) - 下一层的自旋
        SQRT_N: sqrt(N)
    
    Returns:
        part_gap: shape (M, N, N+1)
    """
    M = S_ls.shape[0]
    N = S_ls.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 前一层到当前层的 gap
    for n in prange(N):
        for mu in range(M):
            h_prev = 0.0
            for k in range(N):
                h_prev += J_hidden_lh[n, k] * S_lh[mu, k]
            part_gap[mu, n, 0] = (h_prev / SQRT_N) * S_ls[mu, n]
    
    # 计算 part_gap[:, :, 1:] - 当前层到下一层的 gap
    # 这些 gap 对于同一个 mu 的所有 n 是相同的
    for mu in prange(M):
        for n2 in range(N):
            J_S = 0.0
            for k in range(N):
                J_S += J_hidden_ls[n2, k] * S_ls[mu, k]
            gap_val = (J_S / SQRT_N) * S_ls_plus1[mu, n2]
            for n in range(N):
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_hidden_after_vectorized(J_hidden_lh, J_hidden_ls, S_lh, S_ls, S_ls_plus1, SQRT_N):
    """
    完全向量化计算中间层翻转后的 part_gap (Numba 加速版本)
    
    关键优化：消除嵌套循环，使用数学技巧直接计算翻转效果
    
    翻转 S[mu, l_s, n] 后：
    - part_gap[mu, n, 0] 符号取反
    - part_gap[mu, n, 1:] 需要重新计算，但可以利用线性性质：
      新的 J @ S_flipped = J @ S - 2 * J[:, n] * S[n]
    """
    M = S_ls.shape[0]
    N = S_ls.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 翻转后符号取反
    for n in prange(N):
        for mu in range(M):
            h_prev = 0.0
            for k in range(N):
                h_prev += J_hidden_lh[n, k] * S_lh[mu, k]
            # 翻转后 S[mu, l_s, n] 变为 -S[mu, l_s, n]
            part_gap[mu, n, 0] = (h_prev / SQRT_N) * (-S_ls[mu, n])
    
    # 计算 part_gap[:, n, 1:] - 利用线性性质避免嵌套循环
    # 对于翻转位置 n，新的 J @ S_flipped = J @ S_original - 2 * J[:, n] * S_original[n]
    for mu in prange(M):
        # 首先计算原始的 J @ S
        J_S_original = np.zeros(N, dtype=np.float32)
        for n2 in range(N):
            for k in range(N):
                J_S_original[n2] += J_hidden_ls[n2, k] * S_ls[mu, k]
        
        # 对于每个翻转位置 n
        for n in range(N):
            # 计算翻转后的 J @ S_flipped
            # J @ S_flipped = J @ S_original - 2 * J[:, n] * S_original[n]
            S_n = S_ls[mu, n]
            for n2 in range(N):
                J_S_flipped = J_S_original[n2] - 2.0 * J_hidden_ls[n2, n] * S_n
                gap_val = (J_S_flipped / SQRT_N) * S_ls_plus1[mu, n2]
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_in_before_vectorized(J_in, J_hidden_0, S_in, S_0, S_1, SQRT_N_IN, SQRT_N):
    """完全向量化计算第一层翻转前的 part_gap"""
    M = S_0.shape[0]
    N = S_0.shape[1]
    N_in = S_in.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 输入层 gap
    for n in prange(N):
        for mu in range(M):
            h_in = 0.0
            for k in range(N_in):
                h_in += J_in[n, k] * S_in[mu, k]
            part_gap[mu, n, 0] = (h_in / SQRT_N_IN) * S_0[mu, n]
    
    # 计算 part_gap[:, :, 1:] - 到第二层的 gap
    for mu in prange(M):
        for n2 in range(N):
            J_S = 0.0
            for k in range(N):
                J_S += J_hidden_0[n2, k] * S_0[mu, k]
            gap_val = (J_S / SQRT_N) * S_1[mu, n2]
            for n in range(N):
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_in_after_vectorized(J_in, J_hidden_0, S_in, S_0, S_1, SQRT_N_IN, SQRT_N):
    """完全向量化计算第一层翻转后的 part_gap"""
    M = S_0.shape[0]
    N = S_0.shape[1]
    N_in = S_in.shape[1]
    part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 翻转后符号取反
    for n in prange(N):
        for mu in range(M):
            h_in = 0.0
            for k in range(N_in):
                h_in += J_in[n, k] * S_in[mu, k]
            part_gap[mu, n, 0] = (h_in / SQRT_N_IN) * (-S_0[mu, n])
    
    # 计算 part_gap[:, n, 1:] - 利用线性性质
    for mu in prange(M):
        J_S_original = np.zeros(N, dtype=np.float32)
        for n2 in range(N):
            for k in range(N):
                J_S_original[n2] += J_hidden_0[n2, k] * S_0[mu, k]
        
        for n in range(N):
            S_n = S_0[mu, n]
            for n2 in range(N):
                J_S_flipped = J_S_original[n2] - 2.0 * J_hidden_0[n2, n] * S_n
                gap_val = (J_S_flipped / SQRT_N) * S_1[mu, n2]
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_out_before_vectorized(J_hidden_last, J_out, S_second_last, S_last, S_out, SQRT_N):
    """完全向量化计算最后层翻转前的 part_gap"""
    M = S_last.shape[0]
    N = S_last.shape[1]
    N_out = S_out.shape[1]
    part_gap = np.zeros((M, N, N_out + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 前一层 gap
    for n in prange(N):
        for mu in range(M):
            h_prev = 0.0
            for k in range(N):
                h_prev += J_hidden_last[n, k] * S_second_last[mu, k]
            part_gap[mu, n, 0] = (h_prev / SQRT_N) * S_last[mu, n]
    
    # 计算 part_gap[:, :, 1:] - 输出层 gap
    for mu in prange(M):
        for n2 in range(N_out):
            J_S = 0.0
            for k in range(N):
                J_S += J_out[n2, k] * S_last[mu, k]
            gap_val = (J_S / SQRT_N) * S_out[mu, n2]
            for n in range(N):
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def compute_part_gap_out_after_vectorized(J_hidden_last, J_out, S_second_last, S_last, S_out, SQRT_N):
    """完全向量化计算最后层翻转后的 part_gap"""
    M = S_last.shape[0]
    N = S_last.shape[1]
    N_out = S_out.shape[1]
    part_gap = np.zeros((M, N, N_out + 1), dtype=np.float32)
    
    # 计算 part_gap[:, n, 0] - 翻转后符号取反
    for n in prange(N):
        for mu in range(M):
            h_prev = 0.0
            for k in range(N):
                h_prev += J_hidden_last[n, k] * S_second_last[mu, k]
            part_gap[mu, n, 0] = (h_prev / SQRT_N) * (-S_last[mu, n])
    
    # 计算 part_gap[:, n, 1:] - 利用线性性质
    for mu in prange(M):
        J_S_original = np.zeros(N_out, dtype=np.float32)
        for n2 in range(N_out):
            for k in range(N):
                J_S_original[n2] += J_out[n2, k] * S_last[mu, k]
        
        for n in range(N):
            S_n = S_last[mu, n]
            for n2 in range(N_out):
                J_S_flipped = J_S_original[n2] - 2.0 * J_out[n2, n] * S_n
                gap_val = (J_S_flipped / SQRT_N) * S_out[mu, n2]
                part_gap[mu, n, 1 + n2] = gap_val
    
    return part_gap


@njit(cache=True, fastmath=True, parallel=True)
def metropolis_accept_numba(delta_E, beta, EPS, rand_array):
    """
    Metropolis 接受/拒绝决策 (Numba 加速版本)
    
    Returns:
        accept_mask: shape (M, N) 的布尔数组
    """
    M = delta_E.shape[0]
    N = delta_E.shape[1]
    accept_mask = np.zeros((M, N), dtype=np.bool_)
    
    for mu in prange(M):
        for n in range(N):
            dE = delta_E[mu, n]
            if dE < EPS:
                accept_mask[mu, n] = True
            elif rand_array[mu, n] < np.exp(-dE * beta):
                accept_mask[mu, n] = True
    
    return accept_mask


@njit(cache=True, fastmath=True)
def apply_spin_updates_numba(S_layer, accept_mask, delta_E):
    """
    应用自旋更新 (Numba 加速版本)
    
    Returns:
        updated S_layer, total energy change
    """
    M = S_layer.shape[0]
    N = S_layer.shape[1]
    total_dE = 0.0
    
    for mu in range(M):
        for n in range(N):
            if accept_mask[mu, n]:
                S_layer[mu, n] = -S_layer[mu, n]
                total_dE += delta_E[mu, n]
    
    return total_dE


# ============================================================================
# 计时装饰器
# ============================================================================

time_dict = {}

def timethis(fun):
    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time()
        result = fun(*args, **kwargs)
        end = time()
        name = fun.__name__
        if name in time_dict:
            time_dict[name] += end - start
        else:
            time_dict[name] = end - start
        return result
    return wrapper


def reset_timing():
    """重置计时器"""
    global time_dict
    time_dict = {}


def print_timing_stats():
    """打印计时统计"""
    print("\n" + "=" * 60)
    print("Timing Statistics:")
    print("=" * 60)
    total = sum(time_dict.values())
    for name, t in sorted(time_dict.items(), key=lambda x: -x[1]):
        pct = 100 * t / total if total > 0 else 0
        print(f"  {name:40s}: {t:8.3f}s ({pct:5.1f}%)")
    print(f"  {'TOTAL':40s}: {total:8.3f}s")
    print("=" * 60)


# ============================================================================
# 优化的 Network 类 V2
# ============================================================================

class NetworkOptimizedV2:
    """
    完全向量化 + Numba JIT 优化的神经网络 MC 模拟类
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
        
        # 初始化网络状态
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float32)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
        self.H_history = []
    
    def copy_state_from(self, other):
        """从另一个网络复制状态"""
        self.S = other.S.copy()
        self.S_in = other.S_in.copy()
        self.S_out = other.S_out.copy()
        self.J_hidden = other.J_hidden.copy()
        self.J_in = other.J_in.copy()
        self.J_out = other.J_out.copy()
        self.H = other.H
    
    @timethis
    def update_S_first_layer_vectorized(self):
        """向量化更新第一隐藏层的所有自旋"""
        # 计算翻转前后的 part_gap
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
        
        # 计算能量差
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random((self.M, self.N)).astype(np.float32)
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_spin_updates_numba(self.S[:, 0, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_S_middle_layer_vectorized(self, l_s):
        """向量化更新中间隐藏层的所有自旋"""
        l_h = l_s - 1
        
        # 计算翻转前后的 part_gap
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
        
        # 计算能量差
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random((self.M, self.N)).astype(np.float32)
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_spin_updates_numba(self.S[:, l_s, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_S_last_layer_vectorized(self):
        """向量化更新最后隐藏层的所有自旋"""
        l_s = self.num_hidden_node_layers - 1
        
        # 计算翻转前后的 part_gap
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
        
        # 计算能量差
        E_before = calc_ener_3d_numba(part_gap_before)
        E_after = calc_ener_3d_numba(part_gap_after)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random((self.M, self.N)).astype(np.float32)
        accept_mask = metropolis_accept_numba(delta_E, self.beta, self.EPS, rand_array)
        
        # 应用更新
        total_dE = apply_spin_updates_numba(self.S[:, l_s, :], accept_mask, delta_E)
        self.H += total_dE
    
    @timethis
    def update_all_S_vectorized(self):
        """更新所有隐藏层的自旋"""
        # 第一层
        self.update_S_first_layer_vectorized()
        
        # 中间层
        for l_s in range(1, self.num_hidden_node_layers - 1):
            self.update_S_middle_layer_vectorized(l_s)
        
        # 最后一层
        if self.num_hidden_node_layers > 1:
            self.update_S_last_layer_vectorized()


# ============================================================================
# 用于验证的原始实现 (无优化)
# ============================================================================

def soft_core_potential_numpy(h):
    """软核势能函数 (NumPy 版本)"""
    return np.heaviside(-h, 1.0) * np.power(h, 2)


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
        
        np.random.seed(seed)
        self.S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.float32)
        self.S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
        self.S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
        self.J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
        self.J_in = np.random.randn(N, N_in).astype(np.float32)
        self.J_out = np.random.randn(N_out, N).astype(np.float32)
        
        self.H = 0.0
    
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        """计算中间层自旋翻转前的 part_gap (原始实现)"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
        """计算中间层自旋翻转后的 part_gap (原始实现)"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        # 临时翻转
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        # 恢复
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        return part_gap


def warmup_jit():
    """预热 JIT 编译"""
    print("预热 JIT 编译...")
    net = NetworkOptimizedV2(10, 3, 5, seed=0)
    for _ in range(3):
        net.update_all_S_vectorized()
    print("JIT 预热完成")


if __name__ == "__main__":
    # 简单测试
    warmup_jit()
    
    print("\n测试优化版本...")
    net = NetworkOptimizedV2(120, 3, 10)
    
    import time
    start = time.time()
    for _ in range(10):
        net.update_all_S_vectorized()
    end = time.time()
    
    print(f"10 个 MC 步耗时: {end - start:.4f}s")
    print(f"每步平均耗时: {(end - start) / 10 * 1000:.2f}ms")
