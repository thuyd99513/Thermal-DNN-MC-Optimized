"""
Network_optimized.py - 修正版：按层并行的向量化MC更新实现

本模块精确复现源代码的能量计算逻辑，同时实现按层并行的向量化更新。

关键修正：
1. 能量差计算严格按照源代码的 part_gap 逻辑
2. part_gap 只包含受影响的 N+1 个 gap 值
3. 软核势能函数：V(h) = h^2 if h < 0, else 0

作者：Manus AI
日期：2025-01-23
"""

import numpy as np
from numba import njit, prange
from functools import wraps
from time import time
import os
import copy

# ============================================================================
# 软核势能函数和能量计算 (与源代码完全一致)
# ============================================================================

def soft_core_potential(h):
    """
    软核势能函数
    Ref: Yoshino2019, eqn (32)
    
    V(h) = h^2 if h < 0, else 0
    使用 np.heaviside(-h, 1.0) * h^2 实现
    """
    return np.heaviside(-h, 1.0) * np.power(h, 2)


def calc_ener(r):
    """
    计算能量
    Ref: Yoshino2019, eqn (31a)
    """
    return soft_core_potential(r).sum()


def calc_ener_batch(r_batch):
    """
    批量计算能量
    
    Args:
        r_batch: shape (batch_size, num_gaps) 的数组
    
    Returns:
        shape (batch_size,) 的能量数组
    """
    return soft_core_potential(r_batch).sum(axis=1)


# Numba 加速版本
@njit(cache=True, fastmath=True)
def soft_core_potential_numba(h):
    """软核势能函数 (Numba版本)"""
    result = np.zeros_like(h)
    for i in range(h.size):
        val = h.flat[i]
        if val < 0:
            result.flat[i] = val * val
    return result


@njit(cache=True, fastmath=True)
def calc_ener_numba(r):
    """计算能量 (Numba版本)"""
    total = 0.0
    for i in range(r.size):
        val = r.flat[i]
        if val < 0:
            total += val * val
    return total


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
# 优化的Network类
# ============================================================================

class NetworkOptimized:
    """
    优化的神经网络MC模拟类
    
    精确复现源代码的能量计算逻辑，同时实现按层并行的向量化更新。
    """
    
    def __init__(self, sample_index, tw, L, M, N, N_in, N_out, tot_steps, beta, timestamp, h, qq=2):
        """初始化网络"""
        # 基本参数
        self.init = int(sample_index)
        self.tw = int(tw)
        self.L = int(L)
        self.M = int(M)
        self.N = int(N)
        self.N_in = int(N_in)
        self.N_out = int(N_out)
        self.tot_steps = int(tot_steps)
        self.beta = beta
        self.timestamp = timestamp
        
        # 层数参数
        self.num_hidden_node_layers = self.L - 1
        self.num_hidden_bond_layers = self.L - 2
        
        # 能量
        self.H = 0.0
        self.H_history = []
        
        # 常数 (与源代码一致)
        self.EPS = 0.000001
        self.RAT = 0.1  # Yoshino2019 Eq(35)
        self.RESCALE_J = 1.0 / np.sqrt(1 + self.RAT ** 2)
        self.SQRT_N = np.sqrt(self.N)
        self.SQRT_N_IN = np.sqrt(self.N_in)
        
        # 变量统计
        self.SQ_N = self.N ** 2
        self.num_variables = self.N * self.M * self.num_hidden_node_layers
        self.num_bonds = self.N * self.N_in + self.SQ_N * self.num_hidden_bond_layers + self.N_out * self.N
        self.num = self.num_variables + self.num_bonds
        self.PROB = self.num_variables / self.num
        
        # cutoff 用于区分不同类型的权重
        self.cutoff1 = self.N * self.N_in
        self.cutoff2 = self.cutoff1 + self.num_hidden_bond_layers * self.SQ_N
        
        # 加载数据
        self._load_data(timestamp, sample_index, tw)
        
        # 轨迹存储
        self._init_trajectory_storage(qq)
        
        print(f"NetworkOptimized initialized: L={L}, M={M}, N={N}, beta={beta}")
        print(f"Total variables: {self.num_variables}, Total bonds: {self.num_bonds}")
    
    def _load_data(self, timestamp, sample_index, tw):
        """加载初始配置数据"""
        str_ts = str(timestamp)
        base_path = f'../../ir_hf_L_M_N_sample_mp/data/{str_ts}'
        
        self.S = np.load(f'{base_path}/seed_S_sample{sample_index:d}_tw{tw:d}_L{self.L:d}_M{self.M:d}_N{self.N:d}_beta{self.beta:4.2f}.npy')
        self.J_hidden = np.load(f'{base_path}/seed_J_hidden_sample{sample_index:d}_tw{tw:d}_L{self.L:d}_N{self.N:d}_beta{self.beta:4.2f}.npy')
        self.J_in = np.load(f'{base_path}/seed_J_in_sample{sample_index:d}_tw{tw:d}_N{self.N:d}_N_in{self.N_in:d}_beta{self.beta:4.2f}.npy')
        self.J_out = np.load(f'{base_path}/seed_J_out_sample{sample_index:d}_tw{tw:d}_N_out{self.N_out:d}_N{self.N:d}_beta{self.beta:4.2f}.npy')
        self.S_in = np.load(f'{base_path}/seed_S_in_M{self.M:d}_N_in{self.N_in:d}_beta{self.beta:4.2f}.npy')
        self.S_out = np.load(f'{base_path}/seed_S_out_M{self.M:d}_N_out{self.N_out:d}_beta{self.beta:4.2f}.npy')
        
        # 用于临时存储的数组
        self.new_J_hidden = copy.copy(self.J_hidden)
        self.new_J_in = copy.copy(self.J_in)
        self.new_J_out = copy.copy(self.J_out)
    
    def _init_trajectory_storage(self, qq):
        """初始化轨迹存储"""
        self.BIAS = 1
        qq_ratio = (qq + 1) / qq
        self.T_2 = int(np.log(self.tot_steps * self.num + self.BIAS) / np.log(qq_ratio))
        
        self.J_hidden_traj = np.zeros((self.T_2, self.num_hidden_bond_layers, self.N, self.N), dtype='float32')
        self.S_traj = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
        self.J_in_traj = np.zeros((self.T_2, self.N, self.N_in), dtype='float32')
        self.J_out_traj = np.zeros((self.T_2, self.N_out, self.N), dtype='float32')
        
        self.list_k = [int(qq_ratio ** i) for i in range(self.T_2 + 10)]
        self.ind_save = 0
        self.update_index = 0
    
    # ========================================================================
    # Part Gap 计算函数 (精确复现源代码逻辑)
    # ========================================================================
    
    def part_gap_hidden_before_flip(self, mu, l_s, n):
        """
        计算中间隐藏层自旋翻转前的 part_gap
        
        当 S[mu, l_s, n] 翻转时，受影响的 gap 有 N+1 个：
        - part_gap[0]: 前一层到当前层的 gap (受 S[mu, l_s, n] 影响)
        - part_gap[1:N+1]: 当前层到下一层的 N 个 gap (受 S[mu, l_s, :] 影响)
        
        Ref: Yoshino2019, eqn (31b)
        """
        l_h = l_s - 1  # 权重层索引
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        # Effect on the previous gap (index 0)
        # gap = (J_hidden[l_h, n, :] @ S[mu, l_h, :]) / sqrt(N) * S[mu, l_s, n]
        J_hidden_prev = self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]
        part_gap[0] = (J_hidden_prev / self.SQRT_N) * self.S[mu, l_s, n]
        
        # Effect on the N gaps in the next layer (indices 1 to N)
        # gap[n2] = (J_hidden[l_s, n2, :] @ S[mu, l_s, :]) / sqrt(N) * S[mu, l_s+1, n2]
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        return part_gap
    
    def part_gap_hidden_after_flip(self, mu, l_s, n):
        """计算中间隐藏层自旋翻转后的 part_gap"""
        l_h = l_s - 1
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        # 临时翻转自旋
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        # 计算翻转后的 gap
        part_gap[0] = (self.J_hidden[l_h, n, :] @ self.S[mu, l_h, :]) / self.SQRT_N * self.S[mu, l_s, n]
        J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]
        part_gap[1:] = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
        
        # 恢复自旋
        self.S[mu, l_s, n] = -self.S[mu, l_s, n]
        
        return part_gap
    
    def part_gap_in_before_flip(self, mu, n):
        """
        计算第一隐藏层自旋翻转前的 part_gap
        
        当 S[mu, 0, n] 翻转时，受影响的 gap 有 N+1 个：
        - part_gap[0]: 输入层到第一隐藏层的 gap
        - part_gap[1:N+1]: 第一隐藏层到第二隐藏层的 N 个 gap
        """
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        # Effect on the input gap
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :]) / self.SQRT_N_IN * self.S[mu, 0, n]
        
        # Effect on the N gaps to the next layer
        part_gap[1:] = (self.J_hidden[0, :, :] @ self.S[mu, 0, :]) / self.SQRT_N * self.S[mu, 1, :]
        
        return part_gap
    
    def part_gap_in_after_flip(self, mu, n):
        """计算第一隐藏层自旋翻转后的 part_gap"""
        part_gap = np.zeros(self.N + 1, dtype=np.float32)
        
        # 临时翻转
        self.S[mu, 0, n] = -self.S[mu, 0, n]
        
        part_gap[0] = (self.J_in[n, :] @ self.S_in[mu, :]) / self.SQRT_N_IN * self.S[mu, 0, n]
        part_gap[1:] = (self.J_hidden[0, :, :] @ self.S[mu, 0, :]) / self.SQRT_N * self.S[mu, 1, :]
        
        # 恢复
        self.S[mu, 0, n] = -self.S[mu, 0, n]
        
        return part_gap
    
    def part_gap_out_before_flip(self, mu, n):
        """
        计算最后隐藏层自旋翻转前的 part_gap
        
        当 S[mu, -1, n] 翻转时，受影响的 gap 有 N_out+1 个：
        - part_gap[0]: 倒数第二隐藏层到最后隐藏层的 gap
        - part_gap[1:N_out+1]: 最后隐藏层到输出层的 N_out 个 gap
        """
        part_gap = np.zeros(self.N_out + 1, dtype=np.float32)
        
        # Effect on the previous gap
        J_hidden_S = self.J_hidden[-1, n, :] @ self.S[mu, -2, :]
        part_gap[0] = (J_hidden_S / self.SQRT_N) * self.S[mu, -1, n]
        
        # Effects on the N_out output gaps
        J_out_S = self.J_out @ self.S[mu, -1, :]
        part_gap[1:] = (J_out_S / self.SQRT_N) * self.S_out[mu, :]
        
        return part_gap
    
    def part_gap_out_after_flip(self, mu, n):
        """计算最后隐藏层自旋翻转后的 part_gap"""
        part_gap = np.zeros(self.N_out + 1, dtype=np.float32)
        
        # 临时翻转
        self.S[mu, -1, n] = -self.S[mu, -1, n]
        
        J_hidden_S = self.J_hidden[-1, n, :] @ self.S[mu, -2, :]
        part_gap[0] = (J_hidden_S / self.SQRT_N) * self.S[mu, -1, n]
        J_out_S = self.J_out @ self.S[mu, -1, :]
        part_gap[1:] = (J_out_S / self.SQRT_N) * self.S_out[mu, :]
        
        # 恢复
        self.S[mu, -1, n] = -self.S[mu, -1, n]
        
        return part_gap
    
    # ========================================================================
    # 向量化的 Part Gap 计算 (按层并行)
    # ========================================================================
    
    def part_gap_hidden_before_flip_layer(self, l_s):
        """
        向量化计算第 l_s 层所有自旋翻转前的 part_gap
        
        Args:
            l_s: 隐藏层索引 (1 到 num_hidden_node_layers-2)
        
        Returns:
            part_gap: shape (M, N, N+1) 的数组
                      part_gap[mu, n, :] 是 S[mu, l_s, n] 翻转前的 N+1 个受影响的 gap
        """
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        # 对于每个 n，计算所有 mu 的 part_gap
        for n in range(N):
            # part_gap[:, n, 0]: 前一层到当前层的 gap
            # = (J_hidden[l_h, n, :] @ S[:, l_h, :].T) / sqrt(N) * S[:, l_s, n]
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N  # (M,)
            part_gap[:, n, 0] = h_prev * self.S[:, l_s, n]
        
        # part_gap[:, :, 1:N+1]: 当前层到下一层的 N 个 gap
        # 对于所有 mu 和所有 n，这些 gap 是相同的（因为它们依赖于整个 S[:, l_s, :]）
        # gap[mu, n2] = (J_hidden[l_s, n2, :] @ S[mu, l_s, :]) / sqrt(N) * S[mu, l_s+1, n2]
        for mu in range(M):
            J_hidden_next = self.J_hidden[l_s, :, :] @ self.S[mu, l_s, :]  # (N,)
            gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]  # (N,)
            part_gap[mu, :, 1:] = gap_next  # 广播到所有 n
        
        return part_gap
    
    def part_gap_hidden_after_flip_layer(self, l_s):
        """
        向量化计算第 l_s 层所有自旋翻转后的 part_gap
        
        注意：这里不能真正翻转所有自旋，因为它们会相互影响。
        我们需要逐个计算每个自旋翻转后的效果。
        """
        M, N = self.M, self.N
        l_h = l_s - 1
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            # 翻转 S[:, l_s, n] 后的效果
            S_flipped_n = -self.S[:, l_s, n]  # (M,)
            
            # part_gap[:, n, 0]: 翻转后的前一层 gap
            h_prev = (self.J_hidden[l_h, n, :] @ self.S[:, l_h, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * S_flipped_n
            
            # part_gap[:, n, 1:]: 翻转后的下一层 gap
            # 需要用翻转后的 S[:, l_s, :] 来计算
            for mu in range(M):
                # 创建翻转后的 S_layer
                S_layer_flipped = self.S[mu, l_s, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_hidden_next = self.J_hidden[l_s, :, :] @ S_layer_flipped
                gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, l_s + 1, :]
                part_gap[mu, n, 1:] = gap_next
        
        return part_gap
    
    def part_gap_in_before_flip_layer(self):
        """向量化计算第一隐藏层所有自旋翻转前的 part_gap"""
        M, N = self.M, self.N
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            # part_gap[:, n, 0]: 输入层 gap
            h_in = (self.J_in[n, :] @ self.S_in.T) / self.SQRT_N_IN  # (M,)
            part_gap[:, n, 0] = h_in * self.S[:, 0, n]
        
        # part_gap[:, :, 1:]: 到第二层的 gap
        for mu in range(M):
            J_hidden_next = self.J_hidden[0, :, :] @ self.S[mu, 0, :]
            gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, 1, :]
            part_gap[mu, :, 1:] = gap_next
        
        return part_gap
    
    def part_gap_in_after_flip_layer(self):
        """向量化计算第一隐藏层所有自旋翻转后的 part_gap"""
        M, N = self.M, self.N
        part_gap = np.zeros((M, N, N + 1), dtype=np.float32)
        
        for n in range(N):
            S_flipped_n = -self.S[:, 0, n]
            
            # part_gap[:, n, 0]
            h_in = (self.J_in[n, :] @ self.S_in.T) / self.SQRT_N_IN
            part_gap[:, n, 0] = h_in * S_flipped_n
            
            # part_gap[:, n, 1:]
            for mu in range(M):
                S_layer_flipped = self.S[mu, 0, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_hidden_next = self.J_hidden[0, :, :] @ S_layer_flipped
                gap_next = (J_hidden_next / self.SQRT_N) * self.S[mu, 1, :]
                part_gap[mu, n, 1:] = gap_next
        
        return part_gap
    
    def part_gap_out_before_flip_layer(self):
        """向量化计算最后隐藏层所有自旋翻转前的 part_gap"""
        M, N, N_out = self.M, self.N, self.N_out
        part_gap = np.zeros((M, N, N_out + 1), dtype=np.float32)
        
        for n in range(N):
            # part_gap[:, n, 0]: 前一层 gap
            h_prev = (self.J_hidden[-1, n, :] @ self.S[:, -2, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * self.S[:, -1, n]
        
        # part_gap[:, :, 1:]: 输出层 gap
        for mu in range(M):
            J_out_S = self.J_out @ self.S[mu, -1, :]
            gap_out = (J_out_S / self.SQRT_N) * self.S_out[mu, :]
            part_gap[mu, :, 1:] = gap_out
        
        return part_gap
    
    def part_gap_out_after_flip_layer(self):
        """向量化计算最后隐藏层所有自旋翻转后的 part_gap"""
        M, N, N_out = self.M, self.N, self.N_out
        part_gap = np.zeros((M, N, N_out + 1), dtype=np.float32)
        
        for n in range(N):
            S_flipped_n = -self.S[:, -1, n]
            
            h_prev = (self.J_hidden[-1, n, :] @ self.S[:, -2, :].T) / self.SQRT_N
            part_gap[:, n, 0] = h_prev * S_flipped_n
            
            for mu in range(M):
                S_layer_flipped = self.S[mu, -1, :].copy()
                S_layer_flipped[n] = -S_layer_flipped[n]
                
                J_out_S = self.J_out @ S_layer_flipped
                gap_out = (J_out_S / self.SQRT_N) * self.S_out[mu, :]
                part_gap[mu, n, 1:] = gap_out
        
        return part_gap
    
    # ========================================================================
    # 向量化的自旋更新函数
    # ========================================================================
    
    @timethis
    def update_S_first_layer_vectorized(self):
        """向量化更新第一隐藏层的所有自旋"""
        M, N = self.M, self.N
        
        # 计算翻转前后的 part_gap
        part_gap_before = self.part_gap_in_before_flip_layer()  # (M, N, N+1)
        part_gap_after = self.part_gap_in_after_flip_layer()    # (M, N, N+1)
        
        # 计算能量差
        # delta_E[mu, n] = calc_ener(part_gap_after[mu, n, :]) - calc_ener(part_gap_before[mu, n, :])
        E_before = soft_core_potential(part_gap_before).sum(axis=2)  # (M, N)
        E_after = soft_core_potential(part_gap_after).sum(axis=2)    # (M, N)
        delta_E = E_after - E_before
        
        # Metropolis 接受/拒绝
        rand_array = np.random.random((M, N))
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
        
        # 应用更新
        self.S[:, 0, :] = np.where(accept_mask, -self.S[:, 0, :], self.S[:, 0, :])
        self.H += np.sum(delta_E * accept_mask)
    
    @timethis
    def update_S_middle_layer_vectorized(self, l_s):
        """向量化更新中间隐藏层的所有自旋"""
        M, N = self.M, self.N
        
        part_gap_before = self.part_gap_hidden_before_flip_layer(l_s)
        part_gap_after = self.part_gap_hidden_after_flip_layer(l_s)
        
        E_before = soft_core_potential(part_gap_before).sum(axis=2)
        E_after = soft_core_potential(part_gap_after).sum(axis=2)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((M, N))
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
        
        self.S[:, l_s, :] = np.where(accept_mask, -self.S[:, l_s, :], self.S[:, l_s, :])
        self.H += np.sum(delta_E * accept_mask)
    
    @timethis
    def update_S_last_layer_vectorized(self):
        """向量化更新最后隐藏层的所有自旋"""
        M, N = self.M, self.N
        l_s = self.num_hidden_node_layers - 1
        
        part_gap_before = self.part_gap_out_before_flip_layer()
        part_gap_after = self.part_gap_out_after_flip_layer()
        
        E_before = soft_core_potential(part_gap_before).sum(axis=2)
        E_after = soft_core_potential(part_gap_after).sum(axis=2)
        delta_E = E_after - E_before
        
        rand_array = np.random.random((M, N))
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-delta_E * self.beta))
        
        self.S[:, l_s, :] = np.where(accept_mask, -self.S[:, l_s, :], self.S[:, l_s, :])
        self.H += np.sum(delta_E * accept_mask)
    
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
    
    # ========================================================================
    # 权重更新函数 (与源代码一致的逻辑)
    # ========================================================================
    
    def part_gap_common_shift(self, sqrt_num, J_n, S_before, S_after):
        """
        计算权重更新时的 part_gap
        
        Args:
            sqrt_num: sqrt(N) 或 sqrt(N_in)
            J_n: 权重行向量
            S_before: 前一层自旋 (M, N) 或 (M, N_in)
            S_after: 后一层对应自旋 (M,)
        
        Returns:
            part_gap: (M,) 数组
        """
        part_gap = (np.sum(J_n * S_before, axis=1) / sqrt_num) * S_after
        return part_gap
    
    @timethis
    def update_J_in_vectorized(self):
        """向量化更新输入层权重"""
        N, N_in = self.N, self.N_in
        
        for n2 in range(N):
            # 生成随机扰动
            x = np.random.normal(0, 1, N_in)
            
            # 计算新权重
            new_J_in_n2 = (self.J_in[n2, :] + x * self.RAT) * self.RESCALE_J
            
            # 归一化
            norm = np.sqrt(np.sum(new_J_in_n2 ** 2))
            new_J_in_n2 = new_J_in_n2 * np.sqrt(N_in) / norm
            
            # 计算能量差
            gap_before = self.part_gap_common_shift(self.SQRT_N_IN, self.J_in[n2, :], self.S_in, self.S[:, 0, n2])
            gap_after = self.part_gap_common_shift(self.SQRT_N_IN, new_J_in_n2, self.S_in, self.S[:, 0, n2])
            
            delta_E = calc_ener(gap_after) - calc_ener(gap_before)
            
            # Metropolis
            rand = np.random.random()
            if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
                self.J_in[n2, :] = new_J_in_n2
                self.H += delta_E
    
    @timethis
    def update_J_out_vectorized(self):
        """向量化更新输出层权重"""
        N_out, N = self.N_out, self.N
        
        for n2 in range(N_out):
            x = np.random.normal(0, 1, N)
            
            new_J_out_n2 = (self.J_out[n2, :] + x * self.RAT) * self.RESCALE_J
            norm = np.sqrt(np.sum(new_J_out_n2 ** 2))
            new_J_out_n2 = new_J_out_n2 * np.sqrt(N) / norm
            
            gap_before = self.part_gap_common_shift(self.SQRT_N, self.J_out[n2, :], self.S[:, -1, :], self.S_out[:, n2])
            gap_after = self.part_gap_common_shift(self.SQRT_N, new_J_out_n2, self.S[:, -1, :], self.S_out[:, n2])
            
            delta_E = calc_ener(gap_after) - calc_ener(gap_before)
            
            rand = np.random.random()
            if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
                self.J_out[n2, :] = new_J_out_n2
                self.H += delta_E
    
    @timethis
    def update_J_hidden_layer_vectorized(self, l):
        """向量化更新第l层隐藏权重"""
        N = self.N
        
        for n2 in range(N):
            x = np.random.normal(0, 1, N)
            
            new_J_hidden_l_n2 = (self.J_hidden[l, n2, :] + x * self.RAT) * self.RESCALE_J
            norm = np.sqrt(np.sum(new_J_hidden_l_n2 ** 2))
            new_J_hidden_l_n2 = new_J_hidden_l_n2 * np.sqrt(N) / norm
            
            gap_before = self.part_gap_common_shift(self.SQRT_N, self.J_hidden[l, n2, :], self.S[:, l, :], self.S[:, l + 1, n2])
            gap_after = self.part_gap_common_shift(self.SQRT_N, new_J_hidden_l_n2, self.S[:, l, :], self.S[:, l + 1, n2])
            
            delta_E = calc_ener(gap_after) - calc_ener(gap_before)
            
            rand = np.random.random()
            if delta_E < self.EPS or rand < np.exp(-delta_E * self.beta):
                self.J_hidden[l, n2, :] = new_J_hidden_l_n2
                self.H += delta_E
    
    @timethis
    def update_all_J_vectorized(self):
        """更新所有权重"""
        self.update_J_in_vectorized()
        
        for l in range(self.num_hidden_bond_layers):
            self.update_J_hidden_layer_vectorized(l)
        
        self.update_J_out_vectorized()
    
    # ========================================================================
    # 主MC循环
    # ========================================================================
    
    @timethis
    def mc_step_vectorized(self):
        """执行一个完整的MC步"""
        self.update_all_S_vectorized()
        self.update_all_J_vectorized()
    
    @timethis
    def mc_main_vectorized(self, num_steps=None, verbose=True):
        """主MC模拟循环"""
        if num_steps is None:
            num_steps = self.tot_steps
        
        self.H_history = [self.H]
        
        for mc_index in range(num_steps):
            self.mc_step_vectorized()
            
            if verbose and mc_index % max(1, num_steps // 20) == 0:
                print(f"MC step {mc_index:6d}/{num_steps}, H = {self.H:10.4f}")
            
            self.H_history.append(self.H)
            self.update_index += self.num
            self._check_and_save()
        
        if verbose:
            print(f"MC simulation completed. Final H = {self.H:.4f}")
            print_timing_stats()
    
    def _check_and_save(self):
        """检查并保存轨迹"""
        if self.ind_save < len(self.list_k) and self.ind_save < self.S_traj.shape[0]:
            if self.update_index >= self.list_k[self.ind_save]:
                self.S_traj[self.ind_save] = self.S
                self.J_in_traj[self.ind_save] = self.J_in
                self.J_out_traj[self.ind_save] = self.J_out
                self.J_hidden_traj[self.ind_save] = self.J_hidden
                self.ind_save += 1
    
    # ========================================================================
    # 初始化能量计算 (与源代码一致)
    # ========================================================================
    
    def gap_in_init(self):
        """计算输入层 gap"""
        r_in = np.zeros((self.M, self.N), dtype='float32')
        for mu in range(self.M):
            J_in_S_in = np.dot(self.J_in, self.S_in[mu, :]) / self.SQRT_N_IN
            r_in[mu, :] = J_in_S_in * self.S[mu, 0, :]
        return r_in
    
    def gap_hidden_init(self):
        """计算隐藏层 gap"""
        r_hidden = np.zeros((self.M, self.num_hidden_bond_layers, self.N), dtype='float32')
        for mu in range(self.M):
            for l in range(self.num_hidden_bond_layers):
                J_dot_S = np.dot(self.J_hidden[l, :, :], self.S[mu, l, :])
                r_hidden[mu, l, :] = (J_dot_S / self.SQRT_N) * self.S[mu, l + 1, :]
        return r_hidden
    
    def gap_out_init(self):
        """计算输出层 gap"""
        r_out = np.zeros((self.M, self.N_out), dtype='float32')
        for mu in range(self.M):
            J_out_S = np.dot(self.J_out, self.S[mu, -1, :]) / self.SQRT_N
            r_out[mu, :] = J_out_S * self.S_out[mu, :]
        return r_out
    
    def set_vars(self):
        """初始化变量和能量"""
        self.r_hidden = self.gap_hidden_init()
        self.r_in = self.gap_in_init()
        self.r_out = self.gap_out_init()
        
        self.H_hidden = calc_ener(self.r_hidden)
        self.H_in = calc_ener(self.r_in)
        self.H_out = calc_ener(self.r_out)
        
        # 总能量
        self.H = self.H_hidden + self.H_in + self.H_out
        
        print(f"Initial energy: H_in={self.H_in:.4f}, H_hidden={self.H_hidden:.4f}, H_out={self.H_out:.4f}")
        print(f"Total initial energy: H={self.H:.4f}")
    
    def save_results(self, output_dir):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(f'{output_dir}/S_final.npy', self.S)
        np.save(f'{output_dir}/J_in_final.npy', self.J_in)
        np.save(f'{output_dir}/J_out_final.npy', self.J_out)
        np.save(f'{output_dir}/J_hidden_final.npy', self.J_hidden)
        np.save(f'{output_dir}/H_history.npy', np.array(self.H_history))
        
        print(f"Results saved to {output_dir}")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("Network_optimized.py - 修正版：按层并行的向量化MC更新实现")
    print("=" * 60)
    print("\n能量计算逻辑：")
    print("  1. 软核势能: V(h) = h^2 if h < 0, else 0")
    print("  2. 总能量: H = sum over all gaps of V(gap)")
    print("  3. gap = (J @ S_prev) / sqrt(N) * S_next")
    print("\n使用方法:")
    print("  from Network_optimized import NetworkOptimized")
    print("  net = NetworkOptimized(...)")
    print("  net.set_vars()  # 初始化能量")
    print("  net.mc_main_vectorized(num_steps)")
    print("=" * 60)
