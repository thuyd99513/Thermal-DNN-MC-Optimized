"""
Network_vectorized.py - 按层并行的向量化MC更新实现

本模块实现了基于"按层并行"策略的蒙特卡洛更新算法。
核心思想：同一层的自旋之间没有直接耦合，因此可以并行更新。

优化策略：
1. 自旋更新：按层并行，一次性更新一层所有 M*N 个自旋
2. 权重更新：按层并行，一次性更新一层所有 N*N 个权重
3. 使用NumPy向量化操作替代Python循环

预期加速：10x - 100x (相比原始串行实现)

作者：Manus AI
日期：2025-01-23
"""

import sys
import copy
from functools import wraps
import numpy as np
import os
from time import time
from numba import njit

# ============================================================================
# Numba JIT 加速的核心计算函数
# ============================================================================

@njit(cache=True)
def soft_core_potential_jit(h):
    """
    软核势能函数 (Numba JIT加速版本)
    Ref: Yoshino2019, eqn (32)
    
    V(h) = h^2 if h < 0, else 0
    """
    result = np.zeros_like(h)
    for i in range(h.size):
        if h.flat[i] < 0:
            result.flat[i] = h.flat[i] ** 2
    return result


@njit(cache=True)
def calc_ener_jit(r):
    """
    计算能量 (Numba JIT加速版本)
    Ref: Yoshino2019, eqn (31a)
    """
    total = 0.0
    for i in range(r.size):
        if r.flat[i] < 0:
            total += r.flat[i] ** 2
    return total


def soft_core_potential(h):
    """
    软核势能函数 (NumPy向量化版本)
    """
    return np.where(h < 0, h ** 2, 0.0)


def calc_ener(r):
    """
    计算能量 (NumPy向量化版本)
    """
    return np.sum(soft_core_potential(r))


def calc_ener_batch(r_batch):
    """
    批量计算能量
    
    Args:
        r_batch: shape (batch_size, ...) 的数组
    
    Returns:
        shape (batch_size,) 的能量数组
    """
    # 对除第一维外的所有维度求和
    return np.sum(soft_core_potential(r_batch), axis=tuple(range(1, r_batch.ndim)))


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
        if fun.__name__ in time_dict:
            time_dict[fun.__name__] += end - start
        else:
            time_dict[fun.__name__] = end - start
        return result
    return wrapper


# ============================================================================
# 向量化的Network类
# ============================================================================

class NetworkVectorized:
    """
    向量化的神经网络MC模拟类
    
    实现按层并行的蒙特卡洛更新策略：
    - 自旋S的更新：按层进行，同一层的所有自旋可以并行更新
    - 权重J的更新：按层进行，同一层的所有权重可以并行更新
    """
    
    def __init__(self, sample_index, tw, L, M, N, N_in, N_out, tot_steps, beta, timestamp, h, qq=2):
        """
        初始化网络参数
        
        Args:
            sample_index: 样本索引
            tw: 等待时间
            L: 网络层数
            M: 训练样本数
            N: 每层隐藏节点数
            N_in: 输入节点数 (784 for MNIST)
            N_out: 输出节点数 (2 for binary classification)
            tot_steps: 总MC步数
            beta: 逆温度
            timestamp: 时间戳
            h: 主机索引
            qq: 采样参数
        """
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
        
        # 层数相关参数
        self.num_hidden_node_layers = self.L - 1  # 隐藏节点层数
        self.num_hidden_bond_layers = self.L - 2  # 隐藏权重层数
        
        # 能量相关
        self.H = 0.0
        self.H_history = []
        
        # 加载初始配置
        str_timestamp = str(timestamp)
        self.S = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_sample{:d}_tw{:d}_L{:d}_M{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, L, M, N, beta))
        self.J_hidden = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_hidden_sample{:d}_tw{:d}_L{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, L, N, beta))
        self.J_in = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_in_sample{:d}_tw{:d}_N{:d}_N_in{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, N, N_in, beta))
        self.J_out = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_J_out_sample{:d}_tw{:d}_N_out{:d}_N{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, sample_index, tw, N_out, N, beta))
        
        # 输入输出自旋 (固定边界条件)
        self.S_in = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_in_M{:d}_N_in{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, M, N_in, beta))
        self.S_out = np.load(
            '../../ir_hf_L_M_N_sample_mp/data/{}/seed_S_out_M{:d}_N_out{:d}_beta{:4.2f}.npy'.format(
                str_timestamp, M, N_out, beta))
        
        # 计算相关常数
        self.EPS = 1e-6
        self.RAT = 0.1  # 权重更新步长
        self.RESCALE_J = 1.0 / np.sqrt(1 + self.RAT ** 2)
        self.SQRT_N = np.sqrt(self.N)
        self.SQRT_N_IN = np.sqrt(self.N_in)
        
        # 变量数量统计
        self.num_variables = self.N * self.M * self.num_hidden_node_layers
        self.num_bonds = self.N * self.N_in + (self.N ** 2) * self.num_hidden_bond_layers + self.N_out * self.N
        self.num = self.num_variables + self.num_bonds
        
        # 更新概率 (自旋 vs 权重)
        self.PROB = self.num_variables / self.num
        
        # 轨迹存储相关
        self.ind_save = 0
        self.BIAS = 1
        self.T_2 = self._generate_traj_sampling_num(self.tot_steps * self.num, self.BIAS, qq)
        
        # 初始化轨迹存储数组
        self.J_hidden_traj_hyperfine = np.zeros((self.T_2, self.num_hidden_bond_layers, self.N, self.N), dtype='float32')
        self.S_traj_hyperfine = np.zeros((self.T_2, self.M, self.num_hidden_node_layers, self.N), dtype='int8')
        self.J_in_traj_hyperfine = np.zeros((self.T_2, self.N, self.N_in), dtype='float32')
        self.J_out_traj_hyperfine = np.zeros((self.T_2, self.N_out, self.N), dtype='float32')
        
        # 采样时间点列表
        qq_ratio = (qq + 1) / qq
        self.list_k_4_hyperfine = [int(qq_ratio ** i) for i in range(self.T_2 + 10)]
        
        # 更新计数器
        self.update_index = 0
        
        print(f"NetworkVectorized initialized: L={L}, M={M}, N={N}, beta={beta}")
        print(f"Total variables: {self.num_variables}, Total bonds: {self.num_bonds}")
        print(f"Trajectory sampling points: {self.T_2}")
    
    def _generate_traj_sampling_num(self, num_steps, bias, qq):
        """生成轨迹采样数量"""
        if qq == 1:
            return int(np.log2(num_steps + bias))
        elif qq > 0:
            return int(np.log(num_steps + bias) / np.log((qq + 1) / qq))
        return 10
    
    # ========================================================================
    # 核心向量化更新函数：自旋S的按层并行更新
    # ========================================================================
    
    @timethis
    def update_S_layer_vectorized(self, l):
        """
        向量化更新第l层的所有自旋
        
        同一层的自旋之间没有直接耦合，因此可以并行更新。
        对于第l层的每个自旋 S[mu, l, n]，其能量只依赖于：
        - 前一层的自旋 S[mu, l-1, :] 和权重 J[l-1, n, :]
        - 后一层的自旋 S[mu, l+1, :] 和权重 J[l, :, n]
        
        Args:
            l: 层索引 (0 到 num_hidden_node_layers-1)
        """
        M, N = self.M, self.N
        beta = self.beta
        EPS = self.EPS
        
        if l == 0:
            # 第一隐藏层：与输入层相连
            self._update_first_hidden_layer_vectorized()
        elif l == self.num_hidden_node_layers - 1:
            # 最后隐藏层：与输出层相连
            self._update_last_hidden_layer_vectorized()
        else:
            # 中间隐藏层
            self._update_middle_hidden_layer_vectorized(l)
    
    def _update_first_hidden_layer_vectorized(self):
        """
        向量化更新第一隐藏层 (l=0) 的所有自旋
        
        能量贡献：
        1. 与输入层的连接: r_in[mu, n] = (J_in[n, :] @ S_in[mu, :]) / sqrt(N_in) * S[mu, 0, n]
        2. 与第二隐藏层的连接: r_hidden[mu, 0, n2] = (J_hidden[0, n2, :] @ S[mu, 0, :]) / sqrt(N) * S[mu, 1, n2]
        """
        M, N = self.M, self.N
        
        # ====== 计算翻转前的能量 ======
        # 输入层连接的gap: shape (M, N)
        # J_in: (N, N_in), S_in: (M, N_in) -> (M, N)
        gap_in_before = (self.J_in @ self.S_in.T).T / self.SQRT_N_IN * self.S[:, 0, :]
        
        # 第一隐藏层到第二隐藏层的gap: shape (M, N)
        # J_hidden[0]: (N, N), S[:, 0, :]: (M, N) -> (M, N)
        gap_hidden_before = np.einsum('ij,mj->mi', self.J_hidden[0], self.S[:, 0, :]) / self.SQRT_N * self.S[:, 1, :]
        
        # 每个自旋翻转前的局部能量: shape (M, N)
        # 对于自旋 S[mu, 0, n]，其能量贡献来自 gap_in[mu, n] 和所有 gap_hidden[mu, :] 中包含 S[mu, 0, n] 的项
        # 简化：只计算与该自旋直接相关的能量变化
        energy_before = soft_core_potential(gap_in_before) + np.sum(soft_core_potential(gap_hidden_before), axis=1, keepdims=True)
        
        # ====== 计算翻转后的能量 ======
        # 翻转所有自旋
        S_flipped = -self.S[:, 0, :]
        
        gap_in_after = (self.J_in @ self.S_in.T).T / self.SQRT_N_IN * S_flipped
        gap_hidden_after = np.einsum('ij,mj->mi', self.J_hidden[0], S_flipped) / self.SQRT_N * self.S[:, 1, :]
        
        energy_after = soft_core_potential(gap_in_after) + np.sum(soft_core_potential(gap_hidden_after), axis=1, keepdims=True)
        
        # ====== 计算能量差 (逐个自旋) ======
        # 这里需要更精确地计算每个自旋翻转的能量差
        delta_E = self._compute_delta_E_first_layer()
        
        # ====== Metropolis 接受/拒绝 ======
        rand_array = np.random.rand(M, N)
        accept_prob = np.exp(-self.beta * np.maximum(delta_E, 0))
        accept_mask = (delta_E < self.EPS) | (rand_array < accept_prob)
        
        # ====== 应用更新 ======
        self.S[:, 0, :] = np.where(accept_mask, -self.S[:, 0, :], self.S[:, 0, :])
        self.H += np.sum(delta_E[accept_mask])
    
    def _compute_delta_E_first_layer(self):
        """
        精确计算第一隐藏层每个自旋翻转的能量差
        
        对于 S[mu, 0, n] 的翻转，受影响的gap有：
        1. gap_in[mu, n]: 直接包含 S[mu, 0, n]
        2. gap_hidden[mu, 0, n2] for all n2: 通过 J_hidden[0, n2, :] @ S[mu, 0, :] 间接包含
        
        Returns:
            delta_E: shape (M, N) 的能量差数组
        """
        M, N = self.M, self.N
        delta_E = np.zeros((M, N), dtype=np.float32)
        
        for n in range(N):
            # 计算翻转 S[:, 0, n] 前后的能量差
            
            # Part 1: gap_in 的变化
            # gap_in[mu, n] = (J_in[n, :] @ S_in[mu, :]) / sqrt(N_in) * S[mu, 0, n]
            h_in = (self.J_in[n, :] @ self.S_in.T) / self.SQRT_N_IN  # shape (M,)
            gap_in_before = h_in * self.S[:, 0, n]
            gap_in_after = h_in * (-self.S[:, 0, n])
            
            # Part 2: gap_hidden 的变化 (所有 n2)
            # gap_hidden[mu, 0, n2] = (J_hidden[0, n2, :] @ S[mu, 0, :]) / sqrt(N) * S[mu, 1, n2]
            # 翻转 S[mu, 0, n] 会影响所有 n2 的 gap_hidden
            
            # 计算 J_hidden[0, :, :] @ S[mu, 0, :] 在翻转前后的变化
            # 变化量 = J_hidden[0, :, n] * (-2 * S[mu, 0, n])
            
            # Before: J_hidden[0] @ S[:, 0, :].T -> (N, M)
            h_hidden_before = (self.J_hidden[0] @ self.S[:, 0, :].T).T / self.SQRT_N  # (M, N)
            gap_hidden_before = h_hidden_before * self.S[:, 1, :]  # (M, N)
            
            # After: 只有第n列的贡献变化
            delta_h = -2 * self.S[:, 0, n:n+1] * self.J_hidden[0, :, n:n+1].T / self.SQRT_N  # (M, N)
            h_hidden_after = h_hidden_before + delta_h
            gap_hidden_after = h_hidden_after * self.S[:, 1, :]
            
            # 能量差
            E_before = calc_ener_jit(gap_in_before) + np.sum(soft_core_potential(gap_hidden_before), axis=1)
            E_after = calc_ener_jit(gap_in_after) + np.sum(soft_core_potential(gap_hidden_after), axis=1)
            
            delta_E[:, n] = E_after - E_before
        
        return delta_E
    
    def _update_last_hidden_layer_vectorized(self):
        """
        向量化更新最后隐藏层的所有自旋
        """
        M, N = self.M, self.N
        l = self.num_hidden_node_layers - 1  # 最后隐藏层索引
        l_bond = l - 1  # 对应的权重层索引
        
        delta_E = self._compute_delta_E_last_layer()
        
        # Metropolis 接受/拒绝
        rand_array = np.random.rand(M, N)
        accept_prob = np.exp(-self.beta * np.maximum(delta_E, 0))
        accept_mask = (delta_E < self.EPS) | (rand_array < accept_prob)
        
        # 应用更新
        self.S[:, l, :] = np.where(accept_mask, -self.S[:, l, :], self.S[:, l, :])
        self.H += np.sum(delta_E[accept_mask])
    
    def _compute_delta_E_last_layer(self):
        """
        计算最后隐藏层每个自旋翻转的能量差
        """
        M, N = self.M, self.N
        l = self.num_hidden_node_layers - 1
        l_bond = l - 1
        
        delta_E = np.zeros((M, N), dtype=np.float32)
        
        for n in range(N):
            # Part 1: 与前一层的连接 (gap from J_hidden[l_bond])
            h_prev = (self.J_hidden[l_bond, n, :] @ self.S[:, l-1, :].T) / self.SQRT_N  # (M,)
            gap_prev_before = h_prev * self.S[:, l, n]
            gap_prev_after = h_prev * (-self.S[:, l, n])
            
            # Part 2: 与输出层的连接 (gap_out)
            # gap_out[mu, n_out] = (J_out[n_out, :] @ S[mu, l, :]) / sqrt(N) * S_out[mu, n_out]
            h_out_before = (self.J_out @ self.S[:, l, :].T).T / self.SQRT_N  # (M, N_out)
            gap_out_before = h_out_before * self.S_out  # (M, N_out)
            
            # 翻转后
            delta_h_out = -2 * self.S[:, l, n:n+1] * self.J_out[:, n:n+1].T / self.SQRT_N  # (M, N_out)
            h_out_after = h_out_before + delta_h_out
            gap_out_after = h_out_after * self.S_out
            
            # 能量差
            E_before = soft_core_potential(gap_prev_before) + np.sum(soft_core_potential(gap_out_before), axis=1)
            E_after = soft_core_potential(gap_prev_after) + np.sum(soft_core_potential(gap_out_after), axis=1)
            
            delta_E[:, n] = E_after - E_before
        
        return delta_E
    
    def _update_middle_hidden_layer_vectorized(self, l):
        """
        向量化更新中间隐藏层的所有自旋
        
        Args:
            l: 层索引 (1 到 num_hidden_node_layers-2)
        """
        M, N = self.M, self.N
        l_bond_prev = l - 1  # 前一层权重
        l_bond_next = l      # 后一层权重
        
        delta_E = self._compute_delta_E_middle_layer(l)
        
        # Metropolis 接受/拒绝
        rand_array = np.random.rand(M, N)
        accept_prob = np.exp(-self.beta * np.maximum(delta_E, 0))
        accept_mask = (delta_E < self.EPS) | (rand_array < accept_prob)
        
        # 应用更新
        self.S[:, l, :] = np.where(accept_mask, -self.S[:, l, :], self.S[:, l, :])
        self.H += np.sum(delta_E[accept_mask])
    
    def _compute_delta_E_middle_layer(self, l):
        """
        计算中间隐藏层每个自旋翻转的能量差
        """
        M, N = self.M, self.N
        l_bond_prev = l - 1
        l_bond_next = l
        
        delta_E = np.zeros((M, N), dtype=np.float32)
        
        for n in range(N):
            # Part 1: 与前一层的连接
            h_prev = (self.J_hidden[l_bond_prev, n, :] @ self.S[:, l-1, :].T) / self.SQRT_N
            gap_prev_before = h_prev * self.S[:, l, n]
            gap_prev_after = h_prev * (-self.S[:, l, n])
            
            # Part 2: 与后一层的连接
            h_next_before = (self.J_hidden[l_bond_next] @ self.S[:, l, :].T).T / self.SQRT_N
            gap_next_before = h_next_before * self.S[:, l+1, :]
            
            delta_h_next = -2 * self.S[:, l, n:n+1] * self.J_hidden[l_bond_next, :, n:n+1].T / self.SQRT_N
            h_next_after = h_next_before + delta_h_next
            gap_next_after = h_next_after * self.S[:, l+1, :]
            
            # 能量差
            E_before = soft_core_potential(gap_prev_before) + np.sum(soft_core_potential(gap_next_before), axis=1)
            E_after = soft_core_potential(gap_prev_after) + np.sum(soft_core_potential(gap_next_after), axis=1)
            
            delta_E[:, n] = E_after - E_before
        
        return delta_E
    
    # ========================================================================
    # 核心向量化更新函数：权重J的按层并行更新
    # ========================================================================
    
    @timethis
    def update_J_layer_vectorized(self, layer_type, l=None):
        """
        向量化更新权重
        
        Args:
            layer_type: 'in', 'hidden', 'out'
            l: 对于hidden层，指定层索引
        """
        if layer_type == 'in':
            self._update_J_in_vectorized()
        elif layer_type == 'out':
            self._update_J_out_vectorized()
        elif layer_type == 'hidden':
            self._update_J_hidden_layer_vectorized(l)
    
    def _update_J_in_vectorized(self):
        """
        向量化更新输入层权重 J_in
        
        J_in: shape (N, N_in)
        对于每个 J_in[n2, n1]，其能量贡献通过 gap_in[mu, n2] 体现
        """
        N, N_in = self.N, self.N_in
        M = self.M
        
        # 生成随机扰动
        x = np.random.normal(0, 1, (N, N_in)).astype(np.float32)
        
        # 计算新的权重
        new_J_in = (self.J_in + x * self.RAT) * self.RESCALE_J
        
        # 重新归一化每一行
        norms = np.sqrt(np.sum(new_J_in ** 2, axis=1, keepdims=True))
        new_J_in = new_J_in * np.sqrt(N_in) / norms
        
        # 计算能量差 (逐行)
        delta_E = np.zeros(N, dtype=np.float32)
        
        for n2 in range(N):
            # gap_in[mu, n2] = (J_in[n2, :] @ S_in[mu, :]) / sqrt(N_in) * S[mu, 0, n2]
            gap_before = (self.J_in[n2, :] @ self.S_in.T) / self.SQRT_N_IN * self.S[:, 0, n2]
            gap_after = (new_J_in[n2, :] @ self.S_in.T) / self.SQRT_N_IN * self.S[:, 0, n2]
            
            delta_E[n2] = calc_ener(gap_after) - calc_ener(gap_before)
        
        # Metropolis 接受/拒绝 (逐行)
        rand_array = np.random.rand(N)
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-self.beta * delta_E))
        
        # 应用更新
        self.J_in[accept_mask] = new_J_in[accept_mask]
        self.H += np.sum(delta_E[accept_mask])
    
    def _update_J_out_vectorized(self):
        """
        向量化更新输出层权重 J_out
        """
        N_out, N = self.N_out, self.N
        M = self.M
        
        # 生成随机扰动
        x = np.random.normal(0, 1, (N_out, N)).astype(np.float32)
        
        # 计算新的权重
        new_J_out = (self.J_out + x * self.RAT) * self.RESCALE_J
        
        # 重新归一化每一行
        norms = np.sqrt(np.sum(new_J_out ** 2, axis=1, keepdims=True))
        new_J_out = new_J_out * np.sqrt(N) / norms
        
        # 计算能量差 (逐行)
        delta_E = np.zeros(N_out, dtype=np.float32)
        
        for n2 in range(N_out):
            gap_before = (self.J_out[n2, :] @ self.S[:, -1, :].T) / self.SQRT_N * self.S_out[:, n2]
            gap_after = (new_J_out[n2, :] @ self.S[:, -1, :].T) / self.SQRT_N * self.S_out[:, n2]
            
            delta_E[n2] = calc_ener(gap_after) - calc_ener(gap_before)
        
        # Metropolis 接受/拒绝
        rand_array = np.random.rand(N_out)
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-self.beta * delta_E))
        
        # 应用更新
        self.J_out[accept_mask] = new_J_out[accept_mask]
        self.H += np.sum(delta_E[accept_mask])
    
    def _update_J_hidden_layer_vectorized(self, l):
        """
        向量化更新第l层隐藏权重 J_hidden[l]
        
        Args:
            l: 权重层索引 (0 到 num_hidden_bond_layers-1)
        """
        N = self.N
        M = self.M
        
        # 生成随机扰动
        x = np.random.normal(0, 1, (N, N)).astype(np.float32)
        
        # 计算新的权重
        new_J_hidden_l = (self.J_hidden[l] + x * self.RAT) * self.RESCALE_J
        
        # 重新归一化每一行
        norms = np.sqrt(np.sum(new_J_hidden_l ** 2, axis=1, keepdims=True))
        new_J_hidden_l = new_J_hidden_l * np.sqrt(N) / norms
        
        # 计算能量差 (逐行)
        delta_E = np.zeros(N, dtype=np.float32)
        
        for n2 in range(N):
            # gap_hidden[mu, l, n2] = (J_hidden[l, n2, :] @ S[mu, l, :]) / sqrt(N) * S[mu, l+1, n2]
            gap_before = (self.J_hidden[l, n2, :] @ self.S[:, l, :].T) / self.SQRT_N * self.S[:, l+1, n2]
            gap_after = (new_J_hidden_l[n2, :] @ self.S[:, l, :].T) / self.SQRT_N * self.S[:, l+1, n2]
            
            delta_E[n2] = calc_ener(gap_after) - calc_ener(gap_before)
        
        # Metropolis 接受/拒绝
        rand_array = np.random.rand(N)
        accept_mask = (delta_E < self.EPS) | (rand_array < np.exp(-self.beta * delta_E))
        
        # 应用更新
        self.J_hidden[l, accept_mask] = new_J_hidden_l[accept_mask]
        self.H += np.sum(delta_E[accept_mask])
    
    # ========================================================================
    # 主MC循环
    # ========================================================================
    
    @timethis
    def mc_step_vectorized(self):
        """
        执行一个完整的MC步 (向量化版本)
        
        一个MC步包括：
        1. 更新所有隐藏层的自旋 (按层并行)
        2. 更新所有权重 (按层并行)
        """
        # 更新自旋：遍历所有隐藏层
        for l in range(self.num_hidden_node_layers):
            self.update_S_layer_vectorized(l)
        
        # 更新权重：输入层
        self.update_J_layer_vectorized('in')
        
        # 更新权重：隐藏层
        for l in range(self.num_hidden_bond_layers):
            self.update_J_layer_vectorized('hidden', l)
        
        # 更新权重：输出层
        self.update_J_layer_vectorized('out')
    
    @timethis
    def mc_main_vectorized(self, num_steps=None):
        """
        主MC模拟循环 (向量化版本)
        
        Args:
            num_steps: MC步数，默认使用 self.tot_steps
        """
        if num_steps is None:
            num_steps = self.tot_steps
        
        self.H_history.append(self.H)
        
        for mc_index in range(num_steps):
            self.mc_step_vectorized()
            
            # 记录能量历史
            if mc_index % 100 == 0:
                self.H_history.append(self.H)
                print(f"MC step {mc_index}/{num_steps}, H = {self.H:.4f}")
            
            # 保存轨迹 (可选)
            self.update_index += self.num
            self._check_and_save_hyperfine()
    
    def _check_and_save_hyperfine(self):
        """检查并保存轨迹"""
        if self.ind_save < len(self.list_k_4_hyperfine):
            if self.update_index >= self.list_k_4_hyperfine[self.ind_save]:
                if self.ind_save < self.S_traj_hyperfine.shape[0]:
                    self.S_traj_hyperfine[self.ind_save] = self.S
                    self.J_in_traj_hyperfine[self.ind_save] = self.J_in
                    self.J_out_traj_hyperfine[self.ind_save] = self.J_out
                    self.J_hidden_traj_hyperfine[self.ind_save] = self.J_hidden
                    self.ind_save += 1
    
    # ========================================================================
    # 完全向量化版本：一次性更新所有自旋
    # ========================================================================
    
    @timethis
    def update_all_S_fully_vectorized(self):
        """
        完全向量化的自旋更新
        
        一次性计算所有自旋的能量差，然后并行做出接受/拒绝决策。
        注意：这种方法假设同一层的自旋更新是独立的。
        """
        M, N = self.M, self.N
        
        # 对每一层进行更新
        for l in range(self.num_hidden_node_layers):
            # 计算该层所有自旋翻转的能量差
            if l == 0:
                delta_E = self._compute_delta_E_first_layer_fully_vectorized()
            elif l == self.num_hidden_node_layers - 1:
                delta_E = self._compute_delta_E_last_layer_fully_vectorized()
            else:
                delta_E = self._compute_delta_E_middle_layer_fully_vectorized(l)
            
            # Metropolis 接受/拒绝
            rand_array = np.random.rand(M, N)
            accept_prob = np.exp(-self.beta * np.clip(delta_E, 0, 100))  # clip防止溢出
            accept_mask = (delta_E < self.EPS) | (rand_array < accept_prob)
            
            # 应用更新
            self.S[:, l, :] = np.where(accept_mask, -self.S[:, l, :], self.S[:, l, :])
            self.H += np.sum(delta_E * accept_mask)
    
    def _compute_delta_E_first_layer_fully_vectorized(self):
        """完全向量化计算第一层能量差"""
        M, N = self.M, self.N
        
        # 当前gap值
        # gap_in: (M, N)
        h_in = (self.J_in @ self.S_in.T).T / self.SQRT_N_IN  # (M, N)
        gap_in_before = h_in * self.S[:, 0, :]
        gap_in_after = h_in * (-self.S[:, 0, :])
        
        # gap_hidden: (M, N) - 需要考虑每个自旋翻转对所有gap_hidden的影响
        # 这里简化处理：只考虑直接相关的能量变化
        
        # 能量差 (简化版本)
        delta_E = (soft_core_potential(gap_in_after) - soft_core_potential(gap_in_before))
        
        # 加上对下一层gap的影响
        # J_hidden[0] @ S[:, 0, :].T 的变化
        for n in range(N):
            delta_h = -2 * self.S[:, 0, n:n+1] * self.J_hidden[0, :, n].reshape(1, -1) / self.SQRT_N
            h_next_before = (self.J_hidden[0] @ self.S[:, 0, :].T).T / self.SQRT_N
            h_next_after = h_next_before + delta_h
            
            gap_next_before = h_next_before * self.S[:, 1, :]
            gap_next_after = h_next_after * self.S[:, 1, :]
            
            delta_E[:, n] += np.sum(soft_core_potential(gap_next_after) - soft_core_potential(gap_next_before), axis=1)
        
        return delta_E
    
    def _compute_delta_E_last_layer_fully_vectorized(self):
        """完全向量化计算最后层能量差"""
        M, N = self.M, self.N
        l = self.num_hidden_node_layers - 1
        l_bond = l - 1
        
        delta_E = np.zeros((M, N), dtype=np.float32)
        
        # 与前一层的连接
        h_prev = (self.J_hidden[l_bond] @ self.S[:, l-1, :].T).T / self.SQRT_N  # (M, N)
        # h_prev[mu, n] = J_hidden[l_bond, n, :] @ S[mu, l-1, :] / sqrt(N)
        # 需要重新组织：对于每个n，h_prev应该是 J_hidden[l_bond, n, :] @ S[:, l-1, :].T
        
        for n in range(N):
            h_n = (self.J_hidden[l_bond, n, :] @ self.S[:, l-1, :].T) / self.SQRT_N  # (M,)
            gap_prev_before = h_n * self.S[:, l, n]
            gap_prev_after = h_n * (-self.S[:, l, n])
            
            # 与输出层的连接
            h_out_before = (self.J_out @ self.S[:, l, :].T).T / self.SQRT_N  # (M, N_out)
            delta_h_out = -2 * self.S[:, l, n:n+1] * self.J_out[:, n].reshape(1, -1) / self.SQRT_N
            h_out_after = h_out_before + delta_h_out
            
            gap_out_before = h_out_before * self.S_out
            gap_out_after = h_out_after * self.S_out
            
            delta_E[:, n] = (soft_core_potential(gap_prev_after).flatten() - soft_core_potential(gap_prev_before).flatten() +
                            np.sum(soft_core_potential(gap_out_after) - soft_core_potential(gap_out_before), axis=1))
        
        return delta_E
    
    def _compute_delta_E_middle_layer_fully_vectorized(self, l):
        """完全向量化计算中间层能量差"""
        M, N = self.M, self.N
        l_bond_prev = l - 1
        l_bond_next = l
        
        delta_E = np.zeros((M, N), dtype=np.float32)
        
        for n in range(N):
            # 与前一层的连接
            h_prev = (self.J_hidden[l_bond_prev, n, :] @ self.S[:, l-1, :].T) / self.SQRT_N
            gap_prev_before = h_prev * self.S[:, l, n]
            gap_prev_after = h_prev * (-self.S[:, l, n])
            
            # 与后一层的连接
            h_next_before = (self.J_hidden[l_bond_next] @ self.S[:, l, :].T).T / self.SQRT_N
            delta_h_next = -2 * self.S[:, l, n:n+1] * self.J_hidden[l_bond_next, :, n].reshape(1, -1) / self.SQRT_N
            h_next_after = h_next_before + delta_h_next
            
            gap_next_before = h_next_before * self.S[:, l+1, :]
            gap_next_after = h_next_after * self.S[:, l+1, :]
            
            delta_E[:, n] = (soft_core_potential(gap_prev_after).flatten() - soft_core_potential(gap_prev_before).flatten() +
                            np.sum(soft_core_potential(gap_next_after) - soft_core_potential(gap_next_before), axis=1))
        
        return delta_E


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("Network_vectorized.py - 按层并行的向量化MC更新实现")
    print("=" * 60)
    print("使用方法:")
    print("  from Network_vectorized import NetworkVectorized")
    print("  net = NetworkVectorized(sample_index, tw, L, M, N, N_in, N_out, tot_steps, beta, timestamp, h)")
    print("  net.mc_main_vectorized(num_steps)")
    print("=" * 60)
