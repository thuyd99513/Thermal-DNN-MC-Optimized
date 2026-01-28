#!/usr/bin/env python3
"""
experiment2_quick.py - 快速版实验二：全局训练动态

使用较小的参数快速验证实验代码
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from time import time
from numba import njit, prange

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Network_optimized_v3 import NetworkOptimizedV3

# ============================================================================
# 生成合成的二分类数据
# ============================================================================

def generate_mnist_like_data(train_size=2000, test_size=400, seed=42):
    """
    生成类似 MNIST 的合成二分类数据
    
    模拟论文中的设置：
    - 输入维度 N_in = 784 (28x28)
    - 输出维度 N_out = 2 (二分类)
    - 值为 ±1
    """
    np.random.seed(seed)
    N_in = 784
    N_out = 2
    
    # 生成两类数据，每类有不同的模式
    # 类别 0: 左半部分倾向于 +1
    # 类别 1: 右半部分倾向于 +1
    
    def generate_class(n_samples, class_id):
        X = np.random.choice([-1, 1], size=(n_samples, N_in)).astype(np.float64)
        # 添加类别特征
        if class_id == 0:
            # 前半部分更可能是 +1
            X[:, :N_in//2] = np.where(np.random.rand(n_samples, N_in//2) > 0.3, 1, -1)
        else:
            # 后半部分更可能是 +1
            X[:, N_in//2:] = np.where(np.random.rand(n_samples, N_in//2) > 0.3, 1, -1)
        return X
    
    # 生成训练集
    n_train_per_class = train_size // 2
    X_train_0 = generate_class(n_train_per_class, 0)
    X_train_1 = generate_class(n_train_per_class, 1)
    X_train = np.vstack([X_train_0, X_train_1])
    
    y_train = np.zeros((train_size, N_out))
    y_train[:n_train_per_class, 0] = 1
    y_train[:n_train_per_class, 1] = -1
    y_train[n_train_per_class:, 0] = -1
    y_train[n_train_per_class:, 1] = 1
    
    # 生成测试集
    n_test_per_class = test_size // 2
    X_test_0 = generate_class(n_test_per_class, 0)
    X_test_1 = generate_class(n_test_per_class, 1)
    X_test = np.vstack([X_test_0, X_test_1])
    
    y_test = np.zeros((test_size, N_out))
    y_test[:n_test_per_class, 0] = 1
    y_test[:n_test_per_class, 1] = -1
    y_test[n_test_per_class:, 0] = -1
    y_test[n_test_per_class:, 1] = 1
    
    # 打乱数据
    train_idx = np.random.permutation(train_size)
    test_idx = np.random.permutation(test_size)
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    print(f"生成合成数据: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# 网络类扩展
# ============================================================================

class TDLMWithData(NetworkOptimizedV3):
    """扩展的热力学深度学习机，支持真实数据"""
    
    def __init__(self, X_train, y_train, N, L, beta=1e5, seed=42):
        M = X_train.shape[0]
        N_in = X_train.shape[1]
        N_out = y_train.shape[1]
        
        super().__init__(M, N, L, N_in=N_in, N_out=N_out, beta=beta, seed=seed)
        
        # 设置固定的边界条件
        self.S_in = X_train.copy()
        self.S_out = y_train.copy()
        self.X_train = X_train
        self.y_train = y_train
        
        print(f"网络初始化: M={M}, N={N}, L={L}, N_in={N_in}, N_out={N_out}")


# ============================================================================
# 能量和准确率计算
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_layer_energy_numba(gap):
    """计算单层能量"""
    total = 0.0
    for i in range(gap.size):
        val = gap.flat[i]
        if val < 0:
            total += val * val
    return total


def compute_total_energy(net):
    """计算总能量"""
    total_energy = 0.0
    
    # 第一层
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    gap_0 = h_0 * net.S[:, 0, :]
    total_energy += compute_layer_energy_numba(gap_0)
    
    # 中间层
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        gap_l = h_l * net.S[:, l+1, :]
        total_energy += compute_layer_energy_numba(gap_l)
    
    # 输出层
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N
    gap_out = h_out * net.S_out
    total_energy += compute_layer_energy_numba(gap_out)
    
    return total_energy


def dnn_forward_pass(X, J_in, J_hidden, J_out, SQRT_N_IN, SQRT_N):
    """标准 DNN 前向传播"""
    # 第一隐藏层
    h = X @ J_in.T / SQRT_N_IN
    S = np.sign(h)
    S[S == 0] = 1
    
    # 中间隐藏层
    for l in range(J_hidden.shape[0]):
        h = S @ J_hidden[l].T / SQRT_N
        S = np.sign(h)
        S[S == 0] = 1
    
    # 输出层
    h = S @ J_out.T / SQRT_N
    output = np.sign(h)
    output[output == 0] = 1
    
    return output


def compute_test_accuracy(net, X_test, y_test):
    """计算测试集准确率"""
    output = dnn_forward_pass(
        X_test, net.J_in, net.J_hidden, net.J_out,
        net.SQRT_N_IN, net.SQRT_N
    )
    correct = np.sum(np.all(output == y_test, axis=1))
    return correct / len(y_test)


def compute_training_accuracy(net):
    """计算训练集准确率"""
    correct = 0
    for mu in range(net.M):
        all_satisfied = True
        
        # 检查第一层
        h_0 = net.J_in @ net.S_in[mu] / net.SQRT_N_IN
        gap_0 = h_0 * net.S[mu, 0, :]
        if np.any(gap_0 < 0):
            all_satisfied = False
        
        # 检查中间层
        if all_satisfied:
            for l in range(net.num_hidden_bond_layers):
                h_l = net.J_hidden[l] @ net.S[mu, l, :] / net.SQRT_N
                gap_l = h_l * net.S[mu, l+1, :]
                if np.any(gap_l < 0):
                    all_satisfied = False
                    break
        
        # 检查输出层
        if all_satisfied:
            h_out = net.J_out @ net.S[mu, -1, :] / net.SQRT_N
            gap_out = h_out * net.S_out[mu]
            if np.any(gap_out < 0):
                all_satisfied = False
        
        if all_satisfied:
            correct += 1
    
    return correct / net.M


# ============================================================================
# 自相关和重叠
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_spin_autocorrelation_numba(S_t, S_tw):
    """计算自旋自相关函数"""
    N_dof = S_t.size
    total = 0.0
    for i in range(N_dof):
        total += S_t.flat[i] * S_tw.flat[i]
    return total / N_dof


@njit(cache=True, fastmath=True)
def compute_replica_overlap_numba(S_a, S_b):
    """计算副本间重叠"""
    N_dof = S_a.size
    total = 0.0
    for i in range(N_dof):
        total += S_a.flat[i] * S_b.flat[i]
    return total / N_dof


def compute_layer_autocorrelation(S_t, S_tw, layer):
    """计算特定层的自相关函数"""
    return np.mean(S_t[:, layer, :] * S_tw[:, layer, :])


def compute_layer_overlap(S_a, S_b, layer):
    """计算特定层的副本间重叠"""
    return np.mean(S_a[:, layer, :] * S_b[:, layer, :])


# ============================================================================
# 实验主函数
# ============================================================================

def run_experiment2(X_train, y_train, X_test, y_test, N, L, beta, 
                    mc_steps, t_w_list, log_interval=100, seed=42):
    """运行实验二"""
    M = X_train.shape[0]
    alpha = M / N
    ln_alpha = np.log(alpha)
    
    print(f"\n{'='*60}")
    print(f"实验配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    print(f"β={beta:.0e}, MC steps={mc_steps}")
    print(f"{'='*60}")
    
    # 初始化两个副本
    np.random.seed(seed)
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    
    # 共享权重初始化
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    # 初始能量
    E0 = compute_total_energy(net_a)
    print(f"初始能量 E(0) = {E0:.4f}")
    
    # 存储结果
    time_points = [0]
    energy_ratio = [1.0]
    accuracy_train = [compute_training_accuracy(net_a)]
    accuracy_test = [compute_test_accuracy(net_a, X_test, y_test)]
    
    S_tw_configs = {tw: None for tw in t_w_list}
    autocorr_data = {tw: {'t': [], 'c': []} for tw in t_w_list}
    overlap_data = {tw: {'t': [], 'q': []} for tw in t_w_list}
    layer_autocorr_data = {tw: {l: {'t': [], 'c': []} for l in range(L-1)} for tw in t_w_list}
    layer_overlap_data = {l: {'t': [], 'q': []} for l in range(L-1)}
    
    start_time = time()
    
    for step in range(1, mc_steps + 1):
        # MC 步
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        # 同步权重
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        # 记录数据
        if step % log_interval == 0 or step == 1:
            t = step
            time_points.append(t)
            
            E_t = compute_total_energy(net_a)
            energy_ratio.append(E_t / E0 if E0 > 0 else 1.0)
            
            A_train_t = compute_training_accuracy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            accuracy_train.append(A_train_t)
            accuracy_test.append(A_test_t)
            
            # 保存 t_w 配置
            for tw in t_w_list:
                if step == tw:
                    S_tw_configs[tw] = net_a.S.copy()
            
            # 计算自相关和重叠
            for tw in t_w_list:
                if S_tw_configs[tw] is not None and step >= tw:
                    c_t = compute_spin_autocorrelation_numba(net_a.S, S_tw_configs[tw])
                    autocorr_data[tw]['t'].append(t)
                    autocorr_data[tw]['c'].append(c_t)
                    
                    q_t = compute_replica_overlap_numba(net_a.S, net_b.S)
                    overlap_data[tw]['t'].append(t)
                    overlap_data[tw]['q'].append(q_t)
                    
                    for l in range(L-1):
                        c_l = compute_layer_autocorrelation(net_a.S, S_tw_configs[tw], l)
                        layer_autocorr_data[tw][l]['t'].append(t)
                        layer_autocorr_data[tw][l]['c'].append(c_l)
            
            for l in range(L-1):
                q_l = compute_layer_overlap(net_a.S, net_b.S, l)
                layer_overlap_data[l]['t'].append(t)
                layer_overlap_data[l]['q'].append(q_l)
        
        # 进度
        if step % max(1, mc_steps // 5) == 0:
            elapsed = time() - start_time
            E_t = compute_total_energy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            print(f"  Step {step:>6}/{mc_steps} ({100*step/mc_steps:>5.1f}%), "
                  f"E/E0={E_t/E0:.4e}, A_test={A_test_t:.4f}", flush=True)
    
    total_time = time() - start_time
    print(f"完成，耗时: {total_time:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta,
        'alpha': alpha, 'ln_alpha': ln_alpha,
        'E0': E0,
        'time_points': np.array(time_points),
        'energy_ratio': np.array(energy_ratio),
        'accuracy_train': np.array(accuracy_train),
        'accuracy_test': np.array(accuracy_test),
        'autocorr_data': autocorr_data,
        'overlap_data': overlap_data,
        'layer_autocorr_data': layer_autocorr_data,
        'layer_overlap_data': layer_overlap_data,
        't_w_list': t_w_list,
        'total_time': total_time
    }


# ============================================================================
# 绘图
# ============================================================================

def plot_results(results_list, output_dir):
    """绘制结果"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    # (a) E(t)/E(0)
    ax = axes[0, 0]
    for i, r in enumerate(results_list):
        t = r['time_points']
        mask = t > 0
        ax.loglog(t[mask], r['energy_ratio'][mask], 'o-', markersize=3, 
                  color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('E(t)/E(0)')
    ax.set_title('(a) Energy decay')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (b) A_train(t)
    ax = axes[0, 1]
    for i, r in enumerate(results_list):
        ax.semilogx(r['time_points'], r['accuracy_train'], 'o-', markersize=3,
                    color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('A_train(t)')
    ax.set_title('(b) Training accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (c) A_test(t)
    ax = axes[0, 2]
    for i, r in enumerate(results_list):
        ax.semilogx(r['time_points'], r['accuracy_test'], 'o-', markersize=3,
                    color=colors[i], label=f"ln α={r['ln_alpha']:.1f}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('A_test(t)')
    ax.set_title('(c) Test accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # (d) Final values vs ln α
    ax = axes[0, 3]
    ln_alphas = [r['ln_alpha'] for r in results_list]
    E_stars = [r['energy_ratio'][-1] for r in results_list]
    A_test_stars = [r['accuracy_test'][-1] for r in results_list]
    
    ax2 = ax.twinx()
    line1, = ax.semilogy(ln_alphas, E_stars, 'bo-', markersize=8, label='E*/E(0)')
    line2, = ax2.plot(ln_alphas, A_test_stars, 'rs-', markersize=8, label='A*_test')
    ax.set_xlabel('ln α')
    ax.set_ylabel('E*/E(0)', color='b')
    ax2.set_ylabel('A*_test', color='r')
    ax.set_title('(d) Final values vs ln α')
    ax.legend(handles=[line1, line2], fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (e) c(t, t_w)
    ax = axes[1, 0]
    if results_list:
        r = results_list[0]
        tw_colors = plt.cm.plasma(np.linspace(0, 1, len(r['t_w_list'])))
        for i, tw in enumerate(r['t_w_list']):
            data = r['autocorr_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['c'], 'o-', markersize=3,
                            color=tw_colors[i], label=f"t_w={tw}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('c(t, t_w)')
    ax.set_title('(e) Spin autocorrelation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (f) q(t, t_w)
    ax = axes[1, 1]
    if results_list:
        r = results_list[0]
        for i, tw in enumerate(r['t_w_list']):
            data = r['overlap_data'][tw]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['q'], 'o-', markersize=3,
                            color=tw_colors[i], label=f"t_w={tw}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('q(t, t_w)')
    ax.set_title('(f) Replica overlap')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (g) Layer autocorrelation
    ax = axes[1, 2]
    if results_list:
        r = results_list[0]
        tw = r['t_w_list'][0] if r['t_w_list'] else None
        if tw is not None:
            L = r['L']
            layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
            for l in range(L-1):
                data = r['layer_autocorr_data'][tw][l]
                if len(data['t']) > 0:
                    ax.semilogx(data['t'], data['c'], 'o-', markersize=2,
                                color=layer_colors[l], label=f"l={l+1}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel(f'c_l(t, t_w)')
    ax.set_title('(g) Layer autocorrelation')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    # (h) Layer overlap
    ax = axes[1, 3]
    if results_list:
        r = results_list[0]
        L = r['L']
        layer_colors = plt.cm.coolwarm(np.linspace(0, 1, L-1))
        for l in range(L-1):
            data = r['layer_overlap_data'][l]
            if len(data['t']) > 0:
                ax.semilogx(data['t'], data['q'], 'o-', markersize=2,
                            color=layer_colors[l], label=f"l={l+1}")
    ax.set_xlabel('t (MC steps)')
    ax.set_ylabel('q_l(t)')
    ax.set_title('(h) Layer overlap')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.1)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(output_dir, f'experiment2_figure2_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图表保存到: {filepath}")
    return filepath


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("实验二：全局训练动态 (Figure 2) - 快速版")
    print("=" * 70)
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'reports', 'experiment2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成数据
    print("\n生成合成数据...")
    X_train, y_train, X_test, y_test = generate_mnist_like_data(
        train_size=2000, test_size=400, seed=42
    )
    
    # 实验参数（较小的规模以快速运行）
    L = 10
    beta = 1e5
    mc_steps = 5000  # 减少步数
    t_w_list = [100, 500, 1000]
    log_interval = 50
    
    # 配置
    configs = [
        {'N': 100},  # ln α ≈ 3.0
        {'N': 50},   # ln α ≈ 3.7
        {'N': 20},   # ln α ≈ 4.6
    ]
    
    print(f"\n实验配置: L={L}, β={beta:.0e}, MC steps={mc_steps}")
    
    # 预热 JIT
    print("\n预热 JIT...")
    warmup_net = TDLMWithData(X_train[:10], y_train[:10], N=5, L=5, beta=beta, seed=0)
    for _ in range(3):
        warmup_net.mc_step_vectorized()
    print("JIT 预热完成")
    
    # 运行实验
    results_list = []
    for i, config in enumerate(configs):
        N = config['N']
        print(f"\n配置 {i+1}/{len(configs)}: N={N}")
        
        results = run_experiment2(
            X_train, y_train, X_test, y_test,
            N, L, beta, mc_steps, t_w_list,
            log_interval=log_interval, seed=42+i
        )
        results_list.append(results)
    
    # 绘图
    print("\n生成图表...")
    filepath = plot_results(results_list, output_dir)
    
    # 结果摘要
    print("\n" + "=" * 60)
    print("结果摘要")
    print("=" * 60)
    print(f"{'ln α':>8} | {'E*/E(0)':>12} | {'A*_train':>10} | {'A*_test':>10}")
    print("-" * 50)
    for r in results_list:
        print(f"{r['ln_alpha']:>8.2f} | {r['energy_ratio'][-1]:>12.4e} | "
              f"{r['accuracy_train'][-1]:>10.4f} | {r['accuracy_test'][-1]:>10.4f}")
    
    return results_list, filepath


if __name__ == "__main__":
    main()
