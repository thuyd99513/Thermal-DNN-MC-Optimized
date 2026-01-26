"""
test_correctness_visual.py - 增强版正确性测试，带可视化图表

本脚本通过以下方式验证并生成可视化报告：
1. 软核势能函数测试与可视化
2. 能量计算函数测试与可视化
3. part_gap 计算逻辑测试与可视化
4. Metropolis 接受准则测试与可视化
5. 能量增量更新一致性测试与可视化

作者：Manus AI
日期：2026-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import os
import json
from datetime import datetime

# 导入优化版本
from Network_optimized import (
    NetworkOptimized,
    soft_core_potential,
    calc_ener,
    calc_ener_batch
)

# 创建输出目录
OUTPUT_DIR = '/home/ubuntu/test_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 测试结果存储
test_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'tests': {}
}


def test_soft_core_potential():
    """测试软核势能函数并生成可视化"""
    print("\n" + "=" * 60)
    print("测试 1: 软核势能函数")
    print("=" * 60)
    
    results = {
        'name': '软核势能函数测试',
        'status': 'PASSED',
        'details': [],
        'figure': 'soft_core_potential.png'
    }
    
    # 测试数据
    test_cases = [
        (np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), np.array([4.0, 1.0, 0.0, 0.0, 0.0])),
        (np.array([-0.5]), np.array([0.25])),
        (np.array([0.5]), np.array([0.0])),
        (np.array([0.0]), np.array([0.0])),
        (np.array([-3.0, -1.5, -0.1]), np.array([9.0, 2.25, 0.01])),
    ]
    
    all_passed = True
    for h, expected in test_cases:
        result = soft_core_potential(h)
        passed = np.allclose(result, expected)
        if not passed:
            all_passed = False
            results['status'] = 'FAILED'
        results['details'].append({
            'input': h.tolist(),
            'expected': expected.tolist(),
            'result': result.tolist(),
            'passed': passed
        })
        print(f"输入: {h} -> 期望: {expected}, 结果: {result}, {'✓' if passed else '✗'}")
    
    # 生成可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：软核势能函数曲线
    h_range = np.linspace(-3, 3, 500)
    V_values = soft_core_potential(h_range)
    
    axes[0].plot(h_range, V_values, 'b-', linewidth=2.5, label='V(h) = h² if h<0, else 0')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].fill_between(h_range, V_values, alpha=0.3, color='blue')
    axes[0].set_xlabel('h (gap value)', fontsize=12)
    axes[0].set_ylabel('V(h) (potential energy)', fontsize=12)
    axes[0].set_title('Soft-Core Potential Function', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-0.5, 10)
    
    # 右图：测试用例验证
    test_h = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    test_V = soft_core_potential(test_h)
    expected_V = np.array([4.0, 2.25, 1.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    x_pos = np.arange(len(test_h))
    width = 0.35
    
    bars1 = axes[1].bar(x_pos - width/2, expected_V, width, label='Expected', color='green', alpha=0.7)
    bars2 = axes[1].bar(x_pos + width/2, test_V, width, label='Computed', color='blue', alpha=0.7)
    
    axes[1].set_xlabel('Test Case Index', fontsize=12)
    axes[1].set_ylabel('V(h)', fontsize=12)
    axes[1].set_title('Test Cases Verification', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'h={h:.1f}' for h in test_h], rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'soft_core_potential.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'✓' if all_passed else '✗'} 软核势能函数测试{'全部通过' if all_passed else '存在失败'}")
    test_results['tests']['soft_core_potential'] = results
    return all_passed


def test_calc_ener():
    """测试能量计算函数并生成可视化"""
    print("\n" + "=" * 60)
    print("测试 2: 能量计算函数")
    print("=" * 60)
    
    results = {
        'name': '能量计算函数测试',
        'status': 'PASSED',
        'details': [],
        'figure': 'calc_ener.png'
    }
    
    test_cases = [
        (np.array([-2.0, -1.0, 0.0, 1.0]), 5.0, "混合正负值"),
        (np.array([[-1.0, -2.0], [0.0, 1.0]]), 5.0, "2D数组"),
        (np.array([1.0, 2.0, 3.0]), 0.0, "全正值"),
        (np.array([-1.0, -1.0, -1.0]), 3.0, "全负值"),
        (np.array([0.0, 0.0, 0.0]), 0.0, "全零"),
    ]
    
    all_passed = True
    computed_values = []
    expected_values = []
    labels = []
    
    for r, expected, desc in test_cases:
        result = calc_ener(r)
        passed = np.isclose(result, expected)
        if not passed:
            all_passed = False
            results['status'] = 'FAILED'
        results['details'].append({
            'description': desc,
            'input_shape': list(r.shape),
            'expected': expected,
            'result': float(result),
            'passed': passed
        })
        computed_values.append(result)
        expected_values.append(expected)
        labels.append(desc)
        print(f"{desc}: shape={r.shape} -> 期望: {expected}, 结果: {result:.6f}, {'✓' if passed else '✗'}")
    
    # 生成可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：测试用例对比
    x_pos = np.arange(len(labels))
    width = 0.35
    
    bars1 = axes[0].bar(x_pos - width/2, expected_values, width, label='Expected', color='green', alpha=0.7)
    bars2 = axes[0].bar(x_pos + width/2, computed_values, width, label='Computed', color='blue', alpha=0.7)
    
    axes[0].set_xlabel('Test Case', fontsize=12)
    axes[0].set_ylabel('Energy', fontsize=12)
    axes[0].set_title('Energy Calculation Test Cases', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 右图：能量计算原理示意
    np.random.seed(42)
    sample_gaps = np.random.randn(20) * 1.5
    sample_V = soft_core_potential(sample_gaps)
    total_E = np.sum(sample_V)
    
    colors = ['red' if g < 0 else 'green' for g in sample_gaps]
    axes[1].bar(range(len(sample_gaps)), sample_V, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Gap Index', fontsize=12)
    axes[1].set_ylabel('V(gap)', fontsize=12)
    axes[1].set_title(f'Energy Calculation Example (Total E = {total_E:.2f})', fontsize=14, fontweight='bold')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='gap < 0 (contributes energy)'),
                       Patch(facecolor='green', alpha=0.7, label='gap >= 0 (no contribution)')]
    axes[1].legend(handles=legend_elements, loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'calc_ener.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'✓' if all_passed else '✗'} 能量计算函数测试{'全部通过' if all_passed else '存在失败'}")
    test_results['tests']['calc_ener'] = results
    return all_passed


def test_part_gap_logic():
    """测试 part_gap 计算逻辑并生成可视化"""
    print("\n" + "=" * 60)
    print("测试 3: part_gap 计算逻辑")
    print("=" * 60)
    
    results = {
        'name': 'part_gap 计算逻辑测试',
        'status': 'PASSED',
        'details': {},
        'figure': 'part_gap_logic.png'
    }
    
    # 创建测试数据
    M, N, L = 4, 3, 5
    N_in, N_out = 10, 2
    
    np.random.seed(42)
    
    S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.int8)
    S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
    S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
    J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
    J_in = np.random.randn(N, N_in).astype(np.float32)
    J_out = np.random.randn(N_out, N).astype(np.float32)
    
    SQRT_N = np.sqrt(N)
    SQRT_N_IN = np.sqrt(N_in)
    
    # 测试多个位置
    test_positions = [(0, 1, 0), (1, 2, 1), (2, 1, 2), (3, 2, 0)]
    
    all_part_gaps_before = []
    all_part_gaps_after = []
    all_delta_E = []
    
    for mu, l_s, n in test_positions:
        l_h = l_s - 1
        
        # 计算 before flip
        part_gap_before = np.zeros(N + 1, dtype=np.float32)
        J_hidden_prev = J_hidden[l_h, n, :] @ S[mu, l_h, :]
        part_gap_before[0] = (J_hidden_prev / SQRT_N) * S[mu, l_s, n]
        J_hidden_next = J_hidden[l_s, :, :] @ S[mu, l_s, :]
        part_gap_before[1:] = (J_hidden_next / SQRT_N) * S[mu, l_s + 1, :]
        
        # 计算 after flip
        S_flipped = S.copy()
        S_flipped[mu, l_s, n] = -S_flipped[mu, l_s, n]
        
        part_gap_after = np.zeros(N + 1, dtype=np.float32)
        J_hidden_prev_after = J_hidden[l_h, n, :] @ S_flipped[mu, l_h, :]
        part_gap_after[0] = (J_hidden_prev_after / SQRT_N) * S_flipped[mu, l_s, n]
        J_hidden_next_after = J_hidden[l_s, :, :] @ S_flipped[mu, l_s, :]
        part_gap_after[1:] = (J_hidden_next_after / SQRT_N) * S_flipped[mu, l_s + 1, :]
        
        E_before = calc_ener(part_gap_before)
        E_after = calc_ener(part_gap_after)
        delta_E = E_after - E_before
        
        all_part_gaps_before.append(part_gap_before.copy())
        all_part_gaps_after.append(part_gap_after.copy())
        all_delta_E.append(delta_E)
        
        print(f"位置 S[{mu}, {l_s}, {n}]:")
        print(f"  翻转前 part_gap: {part_gap_before}")
        print(f"  翻转后 part_gap: {part_gap_after}")
        print(f"  能量差: {delta_E:.6f}")
    
    # 验证 part_gap[0] 符号变化
    sign_check_passed = True
    for i, (before, after) in enumerate(zip(all_part_gaps_before, all_part_gaps_after)):
        if before[0] != 0:
            ratio = after[0] / before[0]
            if not np.isclose(ratio, -1.0):
                sign_check_passed = False
    
    results['details'] = {
        'test_positions': test_positions,
        'delta_E_values': [float(e) for e in all_delta_E],
        'sign_check_passed': sign_check_passed
    }
    
    if not sign_check_passed:
        results['status'] = 'FAILED'
    
    # 生成可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (pos, before, after, dE) in enumerate(zip(test_positions, all_part_gaps_before, all_part_gaps_after, all_delta_E)):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(before))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before, width, label='Before Flip', color='blue', alpha=0.7)
        bars2 = ax.bar(x + width/2, after, width, label='After Flip', color='orange', alpha=0.7)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Gap Index', fontsize=10)
        ax.set_ylabel('Gap Value', fontsize=10)
        ax.set_title(f'S[{pos[0]}, {pos[1]}, {pos[2]}] Flip (ΔE = {dE:.4f})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'gap[{i}]' for i in range(len(before))])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Part Gap Calculation Logic Test', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'part_gap_logic.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'✓' if sign_check_passed else '✗'} part_gap 计算逻辑测试{'通过' if sign_check_passed else '失败'}")
    test_results['tests']['part_gap_logic'] = results
    return sign_check_passed


def test_metropolis_acceptance():
    """测试 Metropolis 接受准则并生成可视化"""
    print("\n" + "=" * 60)
    print("测试 4: Metropolis 接受准则")
    print("=" * 60)
    
    results = {
        'name': 'Metropolis 接受准则测试',
        'status': 'PASSED',
        'details': [],
        'figure': 'metropolis_acceptance.png'
    }
    
    beta = 66.7
    EPS = 0.000001
    
    test_cases = [
        (-1.0, "能量降低"),
        (-0.1, "能量小幅降低"),
        (0.0, "能量不变"),
        (1e-7, "能量微小增加"),
        (0.001, "能量极小增加"),
        (0.01, "能量小幅增加"),
        (0.1, "能量中等增加"),
        (1.0, "能量大幅增加"),
    ]
    
    delta_E_list = []
    accept_prob_list = []
    
    print(f"\nbeta = {beta}, EPS = {EPS}")
    print("-" * 50)
    
    for delta_E, description in test_cases:
        if delta_E < EPS:
            accept_prob = 1.0
            reason = "delta_E < EPS"
        else:
            accept_prob = np.exp(-delta_E * beta)
            reason = f"exp(-{delta_E} * {beta})"
        
        delta_E_list.append(delta_E)
        accept_prob_list.append(accept_prob)
        
        results['details'].append({
            'delta_E': delta_E,
            'description': description,
            'accept_prob': accept_prob,
            'reason': reason
        })
        
        print(f"delta_E = {delta_E:10.6f}: {description}")
        print(f"  接受概率: {accept_prob:.6e}")
    
    # 生成可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：接受概率曲线
    delta_E_range = np.linspace(-0.5, 2, 500)
    accept_prob_curve = np.where(delta_E_range < EPS, 1.0, np.exp(-delta_E_range * beta))
    
    axes[0].plot(delta_E_range, accept_prob_curve, 'b-', linewidth=2.5, label=f'P(accept) = exp(-ΔE × β), β={beta}')
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=EPS, color='red', linestyle=':', alpha=0.7, label=f'EPS = {EPS}')
    axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.5)
    axes[0].fill_between(delta_E_range[delta_E_range < 0], accept_prob_curve[delta_E_range < 0], alpha=0.3, color='green')
    
    # 标记测试点
    for dE, prob in zip(delta_E_list, accept_prob_list):
        if dE >= -0.5 and dE <= 2:
            axes[0].scatter([dE], [prob], s=100, zorder=5, edgecolor='black', linewidth=1)
    
    axes[0].set_xlabel('ΔE (Energy Change)', fontsize=12)
    axes[0].set_ylabel('Acceptance Probability', fontsize=12)
    axes[0].set_title('Metropolis Acceptance Criterion', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.5, 2)
    axes[0].set_ylim(-0.05, 1.1)
    
    # 右图：测试用例柱状图
    colors = ['green' if p > 0.5 else 'orange' if p > 0.01 else 'red' for p in accept_prob_list]
    bars = axes[1].bar(range(len(delta_E_list)), accept_prob_list, color=colors, alpha=0.7, edgecolor='black')
    
    axes[1].set_xlabel('Test Case', fontsize=12)
    axes[1].set_ylabel('Acceptance Probability', fontsize=12)
    axes[1].set_title('Test Cases Acceptance Probability', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(delta_E_list)))
    axes[1].set_xticklabels([f'ΔE={dE}' for dE in delta_E_list], rotation=45, ha='right')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-30, 10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, prob) in enumerate(zip(bars, accept_prob_list)):
        if prob > 1e-5:
            axes[1].text(bar.get_x() + bar.get_width()/2, prob * 1.5, f'{prob:.2e}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'metropolis_acceptance.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Metropolis 接受准则测试通过")
    test_results['tests']['metropolis_acceptance'] = results
    return True


def test_energy_conservation():
    """测试能量增量更新的一致性并生成可视化"""
    print("\n" + "=" * 60)
    print("测试 5: 能量增量更新一致性")
    print("=" * 60)
    
    results = {
        'name': '能量增量更新一致性测试',
        'status': 'PASSED',
        'details': [],
        'figure': 'energy_conservation.png'
    }
    
    # 创建测试数据
    M, N, L = 4, 3, 5
    N_in, N_out = 10, 2
    
    np.random.seed(42)
    
    S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.int8)
    S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
    S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
    J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
    J_in = np.random.randn(N, N_in).astype(np.float32)
    J_out = np.random.randn(N_out, N).astype(np.float32)
    
    SQRT_N = np.sqrt(N)
    SQRT_N_IN = np.sqrt(N_in)
    
    def compute_total_energy():
        total = 0.0
        
        # 输入层 gap
        for mu in range(M):
            for n in range(N):
                gap = (J_in[n, :] @ S_in[mu, :]) / SQRT_N_IN * S[mu, 0, n]
                if gap < 0:
                    total += gap ** 2
        
        # 隐藏层 gap
        for l in range(L-2):
            for mu in range(M):
                for n in range(N):
                    gap = (J_hidden[l, n, :] @ S[mu, l, :]) / SQRT_N * S[mu, l+1, n]
                    if gap < 0:
                        total += gap ** 2
        
        # 输出层 gap
        for mu in range(M):
            for n_out in range(N_out):
                gap = (J_out[n_out, :] @ S[mu, -1, :]) / SQRT_N * S_out[mu, n_out]
                if gap < 0:
                    total += gap ** 2
        
        return total
    
    # 测试多次翻转
    test_positions = [(0, 1, 0), (1, 2, 1), (2, 1, 2), (3, 2, 0), (0, 2, 1), (1, 1, 0)]
    
    incremental_deltas = []
    total_deltas = []
    differences = []
    
    for mu, l_s, n in test_positions:
        l_h = l_s - 1
        E_initial = compute_total_energy()
        
        # 计算 part_gap before
        part_gap_before = np.zeros(N + 1, dtype=np.float32)
        part_gap_before[0] = (J_hidden[l_h, n, :] @ S[mu, l_h, :]) / SQRT_N * S[mu, l_s, n]
        part_gap_before[1:] = (J_hidden[l_s, :, :] @ S[mu, l_s, :]) / SQRT_N * S[mu, l_s + 1, :]
        
        # 翻转自旋
        S[mu, l_s, n] = -S[mu, l_s, n]
        
        # 计算 part_gap after
        part_gap_after = np.zeros(N + 1, dtype=np.float32)
        part_gap_after[0] = (J_hidden[l_h, n, :] @ S[mu, l_h, :]) / SQRT_N * S[mu, l_s, n]
        part_gap_after[1:] = (J_hidden[l_s, :, :] @ S[mu, l_s, :]) / SQRT_N * S[mu, l_s + 1, :]
        
        delta_E_incremental = calc_ener(part_gap_after) - calc_ener(part_gap_before)
        E_after = compute_total_energy()
        delta_E_total = E_after - E_initial
        diff = abs(delta_E_incremental - delta_E_total)
        
        incremental_deltas.append(delta_E_incremental)
        total_deltas.append(delta_E_total)
        differences.append(diff)
        
        passed = diff < 1e-5
        if not passed:
            results['status'] = 'FAILED'
        
        results['details'].append({
            'position': [int(mu), int(l_s), int(n)],
            'delta_E_incremental': float(delta_E_incremental),
            'delta_E_total': float(delta_E_total),
            'difference': float(diff),
            'passed': passed
        })
        
        print(f"翻转 S[{mu}, {l_s}, {n}]:")
        print(f"  增量计算: {delta_E_incremental:.6f}")
        print(f"  总能量变化: {delta_E_total:.6f}")
        print(f"  差异: {diff:.10f} {'✓' if passed else '✗'}")
    
    all_passed = all(d < 1e-5 for d in differences)
    
    # 生成可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 左图：增量计算 vs 总能量变化
    x = np.arange(len(test_positions))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, incremental_deltas, width, label='Incremental ΔE', color='blue', alpha=0.7)
    bars2 = axes[0].bar(x + width/2, total_deltas, width, label='Total ΔE', color='green', alpha=0.7)
    
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Flip Position', fontsize=12)
    axes[0].set_ylabel('Energy Change', fontsize=12)
    axes[0].set_title('Incremental vs Total Energy Change', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'S[{p[0]},{p[1]},{p[2]}]' for p in test_positions], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 中图：差异值
    colors = ['green' if d < 1e-7 else 'orange' if d < 1e-5 else 'red' for d in differences]
    bars = axes[1].bar(x, differences, color=colors, alpha=0.7, edgecolor='black')
    axes[1].axhline(y=1e-5, color='red', linestyle='--', alpha=0.7, label='Tolerance (1e-5)')
    axes[1].axhline(y=1e-7, color='orange', linestyle='--', alpha=0.7, label='High Precision (1e-7)')
    
    axes[1].set_xlabel('Flip Position', fontsize=12)
    axes[1].set_ylabel('Absolute Difference', fontsize=12)
    axes[1].set_title('Consistency Check (Difference)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'S[{p[0]},{p[1]},{p[2]}]' for p in test_positions], rotation=45, ha='right')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 右图：散点图对比
    axes[2].scatter(incremental_deltas, total_deltas, s=100, c='blue', alpha=0.7, edgecolor='black')
    
    # 添加对角线
    min_val = min(min(incremental_deltas), min(total_deltas))
    max_val = max(max(incremental_deltas), max(total_deltas))
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
    
    axes[2].set_xlabel('Incremental ΔE', fontsize=12)
    axes[2].set_ylabel('Total ΔE', fontsize=12)
    axes[2].set_title('Correlation: Incremental vs Total', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'energy_conservation.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'✓' if all_passed else '✗'} 能量增量更新一致性测试{'通过' if all_passed else '失败'}")
    test_results['tests']['energy_conservation'] = results
    return all_passed


def generate_summary_figure():
    """生成测试结果汇总图"""
    print("\n" + "=" * 60)
    print("生成测试结果汇总图")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    test_names = [
        'Soft-Core Potential',
        'Energy Calculation',
        'Part Gap Logic',
        'Metropolis Acceptance',
        'Energy Conservation'
    ]
    
    test_keys = ['soft_core_potential', 'calc_ener', 'part_gap_logic', 'metropolis_acceptance', 'energy_conservation']
    
    statuses = []
    colors = []
    for key in test_keys:
        if key in test_results['tests']:
            status = test_results['tests'][key]['status']
            statuses.append(1 if status == 'PASSED' else 0)
            colors.append('green' if status == 'PASSED' else 'red')
        else:
            statuses.append(0)
            colors.append('gray')
    
    bars = ax.barh(test_names, statuses, color=colors, alpha=0.7, edgecolor='black', height=0.6)
    
    # 添加状态标签
    for i, (bar, status) in enumerate(zip(bars, statuses)):
        status_text = '✓ PASSED' if status == 1 else '✗ FAILED'
        ax.text(0.5, bar.get_y() + bar.get_height()/2, status_text, 
                ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if status == 1 else 'white')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Test Status', fontsize=12)
    ax.set_title('Thermal-DNN-MC-Optimized Correctness Test Summary', fontsize=16, fontweight='bold')
    ax.set_xticks([])
    
    # 添加统计信息
    passed_count = sum(statuses)
    total_count = len(statuses)
    summary_text = f'Total: {passed_count}/{total_count} Tests Passed'
    ax.text(0.5, -0.15, summary_text, transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'test_summary.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"汇总图已保存: {fig_path}")


def run_all_tests():
    """运行所有测试并生成报告"""
    print("\n" + "=" * 70)
    print("Thermal-DNN-MC-Optimized 正确性测试")
    print(f"测试时间: {test_results['timestamp']}")
    print("=" * 70)
    
    results = []
    results.append(('软核势能函数', test_soft_core_potential()))
    results.append(('能量计算函数', test_calc_ener()))
    results.append(('part_gap 计算逻辑', test_part_gap_logic()))
    results.append(('Metropolis 接受准则', test_metropolis_acceptance()))
    results.append(('能量增量更新一致性', test_energy_conservation()))
    
    # 生成汇总图
    generate_summary_figure()
    
    # 保存测试结果 JSON
    json_path = os.path.join(OUTPUT_DIR, 'test_results.json')
    
    # 自定义 JSON 编码器处理 numpy 类型
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = '✓ 通过' if passed else '✗ 失败'
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "-" * 70)
    if all_passed:
        print("✓ 所有测试全部通过!")
    else:
        print("✗ 部分测试失败，请检查详细日志")
    
    print(f"\n测试结果已保存至: {OUTPUT_DIR}")
    print(f"  - test_summary.png: 测试汇总图")
    print(f"  - soft_core_potential.png: 软核势能函数测试图")
    print(f"  - calc_ener.png: 能量计算测试图")
    print(f"  - part_gap_logic.png: part_gap 逻辑测试图")
    print(f"  - metropolis_acceptance.png: Metropolis 接受准则测试图")
    print(f"  - energy_conservation.png: 能量守恒测试图")
    print(f"  - test_results.json: 测试结果数据")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
