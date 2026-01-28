#!/usr/bin/env python3
"""
experiment2_mini.py - 迷你版实验二：全局训练动态

使用非常小的参数快速验证实验代码
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from time import time
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Network_optimized_v3 import NetworkOptimizedV3

# ============================================================================
# 数据生成
# ============================================================================

def generate_binary_data(M_train=200, M_test=50, N_in=784, seed=42):
    """生成二分类数据"""
    np.random.seed(seed)
    N_out = 2
    
    # 类别 0: 前半部分倾向于 +1
    # 类别 1: 后半部分倾向于 +1
    def generate_class(n_samples, class_id):
        X = np.random.choice([-1, 1], size=(n_samples, N_in)).astype(np.float64)
        if class_id == 0:
            X[:, :N_in//2] = np.where(np.random.rand(n_samples, N_in//2) > 0.3, 1, -1)
        else:
            X[:, N_in//2:] = np.where(np.random.rand(n_samples, N_in//2) > 0.3, 1, -1)
        return X
    
    # 训练集
    n_train_per_class = M_train // 2
    X_train = np.vstack([generate_class(n_train_per_class, 0), 
                         generate_class(n_train_per_class, 1)])
    y_train = np.zeros((M_train, N_out))
    y_train[:n_train_per_class, 0] = 1
    y_train[:n_train_per_class, 1] = -1
    y_train[n_train_per_class:, 0] = -1
    y_train[n_train_per_class:, 1] = 1
    
    # 测试集
    n_test_per_class = M_test // 2
    X_test = np.vstack([generate_class(n_test_per_class, 0), 
                        generate_class(n_test_per_class, 1)])
    y_test = np.zeros((M_test, N_out))
    y_test[:n_test_per_class, 0] = 1
    y_test[:n_test_per_class, 1] = -1
    y_test[n_test_per_class:, 0] = -1
    y_test[n_test_per_class:, 1] = 1
    
    # 打乱
    train_idx = np.random.permutation(M_train)
    test_idx = np.random.permutation(M_test)
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# 网络类
# ============================================================================

class TDLMWithData(NetworkOptimizedV3):
    def __init__(self, X_train, y_train, N, L, beta=1e5, seed=42):
        M = X_train.shape[0]
        N_in = X_train.shape[1]
        N_out = y_train.shape[1]
        super().__init__(M, N, L, N_in=N_in, N_out=N_out, beta=beta, seed=seed)
        self.S_in = X_train.copy()
        self.S_out = y_train.copy()


# ============================================================================
# 物理量计算
# ============================================================================

@njit(cache=True, fastmath=True)
def compute_energy_numba(gap):
    total = 0.0
    for i in range(gap.size):
        val = gap.flat[i]
        if val < 0:
            total += val * val
    return total


def compute_total_energy(net):
    total_energy = 0.0
    
    h_0 = (net.J_in @ net.S_in.T).T / net.SQRT_N_IN
    gap_0 = h_0 * net.S[:, 0, :]
    total_energy += compute_energy_numba(gap_0)
    
    for l in range(net.num_hidden_bond_layers):
        h_l = (net.J_hidden[l] @ net.S[:, l, :].T).T / net.SQRT_N
        gap_l = h_l * net.S[:, l+1, :]
        total_energy += compute_energy_numba(gap_l)
    
    h_out = (net.J_out @ net.S[:, -1, :].T).T / net.SQRT_N
    gap_out = h_out * net.S_out
    total_energy += compute_energy_numba(gap_out)
    
    return total_energy


def dnn_forward_pass(X, J_in, J_hidden, J_out, SQRT_N_IN, SQRT_N):
    h = X @ J_in.T / SQRT_N_IN
    S = np.sign(h)
    S[S == 0] = 1
    
    for l in range(J_hidden.shape[0]):
        h = S @ J_hidden[l].T / SQRT_N
        S = np.sign(h)
        S[S == 0] = 1
    
    h = S @ J_out.T / SQRT_N
    output = np.sign(h)
    output[output == 0] = 1
    return output


def compute_test_accuracy(net, X_test, y_test):
    output = dnn_forward_pass(X_test, net.J_in, net.J_hidden, net.J_out,
                              net.SQRT_N_IN, net.SQRT_N)
    correct = np.sum(np.all(output == y_test, axis=1))
    return correct / len(y_test)


def compute_training_accuracy(net):
    correct = 0
    for mu in range(net.M):
        all_satisfied = True
        
        h_0 = net.J_in @ net.S_in[mu] / net.SQRT_N_IN
        gap_0 = h_0 * net.S[mu, 0, :]
        if np.any(gap_0 < 0):
            all_satisfied = False
        
        if all_satisfied:
            for l in range(net.num_hidden_bond_layers):
                h_l = net.J_hidden[l] @ net.S[mu, l, :] / net.SQRT_N
                gap_l = h_l * net.S[mu, l+1, :]
                if np.any(gap_l < 0):
                    all_satisfied = False
                    break
        
        if all_satisfied:
            h_out = net.J_out @ net.S[mu, -1, :] / net.SQRT_N
            gap_out = h_out * net.S_out[mu]
            if np.any(gap_out < 0):
                all_satisfied = False
        
        if all_satisfied:
            correct += 1
    
    return correct / net.M


@njit(cache=True, fastmath=True)
def compute_autocorr_numba(S_t, S_tw):
    N_dof = S_t.size
    total = 0.0
    for i in range(N_dof):
        total += S_t.flat[i] * S_tw.flat[i]
    return total / N_dof


@njit(cache=True, fastmath=True)
def compute_overlap_numba(S_a, S_b):
    N_dof = S_a.size
    total = 0.0
    for i in range(N_dof):
        total += S_a.flat[i] * S_b.flat[i]
    return total / N_dof


# ============================================================================
# 实验
# ============================================================================

def run_experiment(X_train, y_train, X_test, y_test, N, L, beta, 
                   mc_steps, t_w_list, log_interval=10, seed=42):
    M = X_train.shape[0]
    alpha = M / N
    ln_alpha = np.log(alpha)
    
    print(f"配置: M={M}, N={N}, L={L}, α={alpha:.1f}, ln α={ln_alpha:.2f}")
    
    np.random.seed(seed)
    net_a = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed)
    net_b = TDLMWithData(X_train, y_train, N, L, beta=beta, seed=seed+1000)
    
    net_b.J_in = net_a.J_in.copy()
    net_b.J_hidden = net_a.J_hidden.copy()
    net_b.J_out = net_a.J_out.copy()
    
    E0 = compute_total_energy(net_a)
    print(f"E(0) = {E0:.2f}")
    
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
        net_a.mc_step_vectorized()
        net_b.mc_step_vectorized()
        
        net_b.J_in = net_a.J_in.copy()
        net_b.J_hidden = net_a.J_hidden.copy()
        net_b.J_out = net_a.J_out.copy()
        
        if step % log_interval == 0 or step == 1:
            t = step
            time_points.append(t)
            
            E_t = compute_total_energy(net_a)
            energy_ratio.append(E_t / E0 if E0 > 0 else 1.0)
            
            A_train_t = compute_training_accuracy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            accuracy_train.append(A_train_t)
            accuracy_test.append(A_test_t)
            
            for tw in t_w_list:
                if step == tw:
                    S_tw_configs[tw] = net_a.S.copy()
            
            for tw in t_w_list:
                if S_tw_configs[tw] is not None and step >= tw:
                    c_t = compute_autocorr_numba(net_a.S, S_tw_configs[tw])
                    autocorr_data[tw]['t'].append(t)
                    autocorr_data[tw]['c'].append(c_t)
                    
                    q_t = compute_overlap_numba(net_a.S, net_b.S)
                    overlap_data[tw]['t'].append(t)
                    overlap_data[tw]['q'].append(q_t)
                    
                    for l in range(L-1):
                        c_l = np.mean(net_a.S[:, l, :] * S_tw_configs[tw][:, l, :])
                        layer_autocorr_data[tw][l]['t'].append(t)
                        layer_autocorr_data[tw][l]['c'].append(c_l)
            
            for l in range(L-1):
                q_l = np.mean(net_a.S[:, l, :] * net_b.S[:, l, :])
                layer_overlap_data[l]['t'].append(t)
                layer_overlap_data[l]['q'].append(q_l)
        
        if step % max(1, mc_steps // 5) == 0:
            E_t = compute_total_energy(net_a)
            A_test_t = compute_test_accuracy(net_a, X_test, y_test)
            print(f"  Step {step}/{mc_steps}: E/E0={E_t/E0:.4e}, A_test={A_test_t:.4f}")
    
    print(f"完成，耗时: {time()-start_time:.1f}s")
    
    return {
        'M': M, 'N': N, 'L': L, 'beta': beta,
        'alpha': alpha, 'ln_alpha': ln_alpha, 'E0': E0,
        'time_points': np.array(time_points),
        'energy_ratio': np.array(energy_ratio),
        'accuracy_train': np.array(accuracy_train),
        'accuracy_test': np.array(accuracy_test),
        'autocorr_data': autocorr_data,
        'overlap_data': overlap_data,
        'layer_autocorr_data': layer_autocorr_data,
        'layer_overlap_data': layer_overlap_data,
        't_w_list': t_w_list
    }


def plot_results(results_list, output_dir):
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
    
    # (d) Final values
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


def main():
    print("=" * 60)
    print("实验二：全局训练动态 (Figure 2) - 迷你版")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '..', 'reports', 'experiment2_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 小规模数据
    print("\n生成数据...")
    X_train, y_train, X_test, y_test = generate_binary_data(
        M_train=200, M_test=50, N_in=784, seed=42
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 小规模参数
    L = 6  # 减少层数
    beta = 1e5
    mc_steps = 500  # 减少步数
    t_w_list = [50, 100, 200]
    log_interval = 10
    
    # 配置
    configs = [
        {'N': 20},   # ln α ≈ 2.3
        {'N': 10},   # ln α ≈ 3.0
        {'N': 5},    # ln α ≈ 3.7
    ]
    
    print(f"\n参数: L={L}, β={beta:.0e}, MC steps={mc_steps}")
    
    # 预热
    print("\n预热 JIT...")
    warmup_net = TDLMWithData(X_train[:10], y_train[:10], N=3, L=4, beta=beta, seed=0)
    for _ in range(3):
        warmup_net.mc_step_vectorized()
    print("完成")
    
    # 运行
    results_list = []
    for i, config in enumerate(configs):
        N = config['N']
        print(f"\n配置 {i+1}/{len(configs)}: N={N}")
        results = run_experiment(X_train, y_train, X_test, y_test,
                                 N, L, beta, mc_steps, t_w_list,
                                 log_interval=log_interval, seed=42+i)
        results_list.append(results)
    
    # 绘图
    print("\n生成图表...")
    filepath = plot_results(results_list, output_dir)
    
    # 摘要
    print("\n" + "=" * 50)
    print("结果摘要")
    print("=" * 50)
    print(f"{'ln α':>8} | {'E*/E(0)':>12} | {'A*_train':>10} | {'A*_test':>10}")
    print("-" * 50)
    for r in results_list:
        print(f"{r['ln_alpha']:>8.2f} | {r['energy_ratio'][-1]:>12.4e} | "
              f"{r['accuracy_train'][-1]:>10.4f} | {r['accuracy_test'][-1]:>10.4f}")
    
    return results_list, filepath


if __name__ == "__main__":
    main()
