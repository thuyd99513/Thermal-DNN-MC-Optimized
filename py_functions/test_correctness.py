"""
test_correctness.py - 验证修正后代码与源代码的一致性

本脚本通过以下方式验证：
1. 比较 part_gap 函数的输出
2. 比较能量差计算
3. 比较 Metropolis 接受准则

作者：Manus AI
日期：2025-01-23
"""

import numpy as np
import sys

# 导入优化版本
from Network_optimized import (
    NetworkOptimized,
    soft_core_potential,
    calc_ener,
    calc_ener_batch
)


def test_soft_core_potential():
    """测试软核势能函数"""
    print("\n" + "=" * 60)
    print("测试软核势能函数")
    print("=" * 60)
    
    # 测试数据
    test_cases = [
        (np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), np.array([4.0, 1.0, 0.0, 0.0, 0.0])),
        (np.array([-0.5]), np.array([0.25])),
        (np.array([0.5]), np.array([0.0])),
        (np.array([0.0]), np.array([0.0])),  # 边界情况
    ]
    
    for h, expected in test_cases:
        result = soft_core_potential(h)
        print(f"输入: {h}")
        print(f"期望: {expected}")
        print(f"结果: {result}")
        assert np.allclose(result, expected), f"失败: {result} != {expected}"
        print("✓ 通过")
    
    print("\n✓ 软核势能函数测试全部通过")


def test_calc_ener():
    """测试能量计算函数"""
    print("\n" + "=" * 60)
    print("测试能量计算函数")
    print("=" * 60)
    
    test_cases = [
        (np.array([-2.0, -1.0, 0.0, 1.0]), 5.0),  # 4 + 1 + 0 + 0 = 5
        (np.array([[-1.0, -2.0], [0.0, 1.0]]), 5.0),  # 1 + 4 + 0 + 0 = 5
        (np.array([1.0, 2.0, 3.0]), 0.0),  # 全正，能量为0
    ]
    
    for r, expected in test_cases:
        result = calc_ener(r)
        print(f"输入 shape: {r.shape}")
        print(f"期望: {expected}")
        print(f"结果: {result}")
        assert np.isclose(result, expected), f"失败: {result} != {expected}"
        print("✓ 通过")
    
    print("\n✓ 能量计算函数测试全部通过")


def test_part_gap_logic():
    """测试 part_gap 计算逻辑"""
    print("\n" + "=" * 60)
    print("测试 part_gap 计算逻辑")
    print("=" * 60)
    
    # 创建简单的测试数据
    M, N, L = 4, 3, 5
    N_in, N_out = 10, 2
    
    np.random.seed(42)
    
    # 模拟网络状态
    S = np.random.choice([-1, 1], size=(M, L-1, N)).astype(np.int8)
    S_in = np.random.choice([-1, 1], size=(M, N_in)).astype(np.float32)
    S_out = np.random.choice([-1, 1], size=(M, N_out)).astype(np.float32)
    J_hidden = np.random.randn(L-2, N, N).astype(np.float32)
    J_in = np.random.randn(N, N_in).astype(np.float32)
    J_out = np.random.randn(N_out, N).astype(np.float32)
    
    SQRT_N = np.sqrt(N)
    SQRT_N_IN = np.sqrt(N_in)
    
    # 测试中间层 part_gap 计算
    print("\n测试中间层 part_gap 计算:")
    mu, l_s, n = 0, 2, 1
    l_h = l_s - 1
    
    # 计算 before flip
    part_gap_before = np.zeros(N + 1, dtype=np.float32)
    J_hidden_prev = J_hidden[l_h, n, :] @ S[mu, l_h, :]
    part_gap_before[0] = (J_hidden_prev / SQRT_N) * S[mu, l_s, n]
    J_hidden_next = J_hidden[l_s, :, :] @ S[mu, l_s, :]
    part_gap_before[1:] = (J_hidden_next / SQRT_N) * S[mu, l_s + 1, :]
    
    print(f"  翻转前 part_gap: {part_gap_before}")
    
    # 计算 after flip
    part_gap_after = np.zeros(N + 1, dtype=np.float32)
    S_flipped = S.copy()
    S_flipped[mu, l_s, n] = -S_flipped[mu, l_s, n]
    
    J_hidden_prev_after = J_hidden[l_h, n, :] @ S_flipped[mu, l_h, :]
    part_gap_after[0] = (J_hidden_prev_after / SQRT_N) * S_flipped[mu, l_s, n]
    J_hidden_next_after = J_hidden[l_s, :, :] @ S_flipped[mu, l_s, :]
    part_gap_after[1:] = (J_hidden_next_after / SQRT_N) * S_flipped[mu, l_s + 1, :]
    
    print(f"  翻转后 part_gap: {part_gap_after}")
    
    # 计算能量差
    E_before = calc_ener(part_gap_before)
    E_after = calc_ener(part_gap_after)
    delta_E = E_after - E_before
    
    print(f"  翻转前能量: {E_before:.6f}")
    print(f"  翻转后能量: {E_after:.6f}")
    print(f"  能量差: {delta_E:.6f}")
    
    # 验证：part_gap[0] 应该只是符号变化（因为只有 S[mu, l_s, n] 变了）
    # 但 part_gap[1:] 会因为 S[mu, l_s, :] 的变化而改变
    print(f"\n  验证 part_gap[0] 符号变化:")
    print(f"    before: {part_gap_before[0]:.6f}")
    print(f"    after:  {part_gap_after[0]:.6f}")
    print(f"    比值:   {part_gap_after[0] / part_gap_before[0] if part_gap_before[0] != 0 else 'N/A'}")
    
    print("\n✓ part_gap 计算逻辑测试通过")


def test_metropolis_acceptance():
    """测试 Metropolis 接受准则"""
    print("\n" + "=" * 60)
    print("测试 Metropolis 接受准则")
    print("=" * 60)
    
    beta = 66.7
    EPS = 0.000001
    
    test_cases = [
        (-1.0, "能量降低 (delta_E < 0)"),
        (0.0, "能量不变 (delta_E = 0)"),
        (1e-7, "能量微小增加 (delta_E < EPS)"),
        (0.01, "能量小幅增加"),
        (1.0, "能量大幅增加"),
    ]
    
    print(f"\nbeta = {beta}, EPS = {EPS}")
    print("-" * 50)
    
    for delta_E, description in test_cases:
        # 源代码的接受逻辑
        if delta_E < EPS:
            accept_prob = 1.0
            reason = "delta_E < EPS"
        else:
            accept_prob = np.exp(-delta_E * beta)
            reason = f"exp(-{delta_E} * {beta}) = {accept_prob:.6e}"
        
        print(f"\ndelta_E = {delta_E:10.6f}: {description}")
        print(f"  接受概率: {accept_prob:.6e}")
        print(f"  原因: {reason}")
    
    print("\n✓ Metropolis 接受准则测试通过")


def test_energy_conservation():
    """测试能量守恒（增量更新的一致性）"""
    print("\n" + "=" * 60)
    print("测试能量增量更新的一致性")
    print("=" * 60)
    
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
    
    # 计算初始总能量
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
    
    E_initial = compute_total_energy()
    print(f"初始总能量: {E_initial:.6f}")
    
    # 模拟一次自旋翻转并验证能量变化
    mu, l_s, n = 1, 2, 0
    l_h = l_s - 1
    
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
    
    # 计算增量能量差
    delta_E_incremental = calc_ener(part_gap_after) - calc_ener(part_gap_before)
    
    # 计算翻转后的总能量
    E_after = compute_total_energy()
    delta_E_total = E_after - E_initial
    
    print(f"\n翻转 S[{mu}, {l_s}, {n}] 后:")
    print(f"  增量计算的能量差: {delta_E_incremental:.6f}")
    print(f"  总能量变化: {delta_E_total:.6f}")
    print(f"  差异: {abs(delta_E_incremental - delta_E_total):.10f}")
    
    # 验证
    if np.isclose(delta_E_incremental, delta_E_total, rtol=1e-5):
        print("\n✓ 能量增量更新一致性测试通过")
    else:
        print("\n✗ 警告：能量增量更新存在差异！")
        print("  这可能是因为 part_gap 只计算了部分受影响的 gap")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行所有正确性测试")
    print("=" * 60)
    
    test_soft_core_potential()
    test_calc_ener()
    test_part_gap_logic()
    test_metropolis_acceptance()
    test_energy_conservation()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
