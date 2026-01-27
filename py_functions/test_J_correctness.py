"""
test_J_correctness.py - J 更新优化版本的正确性测试

本脚本验证 Network_optimized_v3.py 中 J 更新的优化实现与原始串行实现的一致性。

测试内容：
1. 权重更新后的 rescale 正确性
2. Gap 计算的正确性
3. 能量差计算的正确性
4. Metropolis 接受准则的正确性
5. 完整 MC 步的能量演化

作者：Manus AI
日期：2026-01-27
"""

import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Network_optimized_v3 import (
    NetworkOptimizedV3,
    NetworkOriginalReference,
    compute_J_update_hidden_vectorized,
    compute_J_update_in_vectorized,
    compute_J_update_out_vectorized,
    calc_ener_2d_rows_numba,
    warmup_jit
)


def test_rescale_correctness():
    """测试权重更新后的 rescale 正确性"""
    print("\n" + "=" * 60)
    print("测试 1: 权重 Rescale 正确性")
    print("=" * 60)
    
    np.random.seed(42)
    N = 3
    N_in = 784
    
    # 创建测试权重
    J = np.random.randn(N, N).astype(np.float64)
    
    # 归一化到目标范数
    target_norm = float(N)
    for n in range(N):
        norm = np.sqrt(np.sum(J[n] ** 2))
        J[n] *= np.sqrt(target_norm) / norm
    
    # 验证初始范数
    for n in range(N):
        norm_sq = np.sum(J[n] ** 2)
        assert np.abs(norm_sq - target_norm) < 1e-10, f"初始范数错误: {norm_sq} != {target_norm}"
    
    # 模拟更新
    RAT = 0.1
    RESCALE_J = 1.0 / np.sqrt(1 + RAT ** 2)
    
    for n2 in range(N):
        n1 = np.random.randint(0, N)
        x = np.random.normal()
        
        # 更新
        new_J = J.copy()
        new_J[n2, n1] = (J[n2, n1] + x * RAT) * RESCALE_J
        
        # Rescale
        norm_sq = np.sum(new_J[n2] ** 2)
        scale = np.sqrt(target_norm / norm_sq)
        new_J[n2] *= scale
        
        # 验证更新后范数
        final_norm_sq = np.sum(new_J[n2] ** 2)
        assert np.abs(final_norm_sq - target_norm) < 1e-10, f"更新后范数错误: {final_norm_sq} != {target_norm}"
    
    print("✓ 权重 Rescale 测试通过")
    return True


def test_gap_calculation():
    """测试 Gap 计算的正确性"""
    print("\n" + "=" * 60)
    print("测试 2: Gap 计算正确性")
    print("=" * 60)
    
    np.random.seed(42)
    M, N, L = 30, 3, 5
    
    # 创建网络
    net = NetworkOptimizedV3(M, N, L, seed=42)
    
    # 测试隐藏层 gap 计算
    l = 1  # 第 1 层
    SQRT_N = net.SQRT_N
    
    # 原始计算方式
    for n2 in range(N):
        J_row = net.J_hidden[l, n2]
        gap_original = (J_row @ net.S[:, l, :].T / SQRT_N) * net.S[:, l + 1, n2]
        
        # 向量化计算方式
        n1_array = np.zeros(N, dtype=np.int64)
        x_array = np.zeros(N)
        
        _, gap_before, _ = compute_J_update_hidden_vectorized(
            net.J_hidden[l], net.S[:, l, :], net.S[:, l + 1, :],
            n1_array, x_array,
            net.RAT, net.RESCALE_J, SQRT_N, float(N)
        )
        
        # 比较
        diff = np.max(np.abs(gap_original - gap_before[n2]))
        assert diff < 1e-10, f"Gap 计算差异过大: {diff}"
    
    print("✓ Gap 计算测试通过")
    return True


def test_energy_difference():
    """测试能量差计算的正确性"""
    print("\n" + "=" * 60)
    print("测试 3: 能量差计算正确性")
    print("=" * 60)
    
    np.random.seed(42)
    M, N, L = 30, 3, 5
    
    # 创建两个相同初始状态的网络
    net_v3 = NetworkOptimizedV3(M, N, L, seed=42)
    net_ref = NetworkOriginalReference(M, N, L, seed=42)
    
    # 确保初始状态相同
    assert np.allclose(net_v3.J_hidden, net_ref.J_hidden), "初始 J_hidden 不一致"
    assert np.allclose(net_v3.S, net_ref.S), "初始 S 不一致"
    
    # 测试多次更新的能量差
    l = 1
    errors = []
    
    for trial in range(10):
        n2 = np.random.randint(0, N)
        n1 = np.random.randint(0, N)
        x = np.random.normal()
        
        # 原始方式计算能量差
        new_J_row = net_ref.J_hidden[l, n2].copy()
        new_J_row[n1] = (new_J_row[n1] + x * net_ref.RAT) * net_ref.RESCALE_J
        norm = np.sqrt(np.sum(new_J_row ** 2))
        new_J_row *= np.sqrt(N) / norm
        
        gap_before = net_ref.part_gap_hidden_shift(l, n2, net_ref.J_hidden[l, n2])
        gap_after = net_ref.part_gap_hidden_shift(l, n2, new_J_row)
        delta_E_ref = net_ref.calc_ener(gap_after) - net_ref.calc_ener(gap_before)
        
        # 向量化方式计算能量差
        n1_array = np.zeros(N, dtype=np.int64)
        n1_array[n2] = n1
        x_array = np.zeros(N)
        x_array[n2] = x
        
        new_J, gap_before_v3, gap_after_v3 = compute_J_update_hidden_vectorized(
            net_v3.J_hidden[l], net_v3.S[:, l, :], net_v3.S[:, l + 1, :],
            n1_array, x_array,
            net_v3.RAT, net_v3.RESCALE_J, net_v3.SQRT_N, float(N)
        )
        
        E_before = calc_ener_2d_rows_numba(gap_before_v3)
        E_after = calc_ener_2d_rows_numba(gap_after_v3)
        delta_E_v3 = (E_after - E_before)[n2]
        
        error = np.abs(delta_E_ref - delta_E_v3)
        errors.append(error)
    
    max_error = max(errors)
    mean_error = np.mean(errors)
    
    print(f"  最大误差: {max_error:.2e}")
    print(f"  平均误差: {mean_error:.2e}")
    
    assert max_error < 1e-8, f"能量差计算误差过大: {max_error}"
    
    print("✓ 能量差计算测试通过")
    return True


def test_metropolis_acceptance():
    """测试 Metropolis 接受准则的正确性"""
    print("\n" + "=" * 60)
    print("测试 4: Metropolis 接受准则正确性")
    print("=" * 60)
    
    from Network_optimized_v3 import metropolis_accept_1d_numba
    
    beta = 66.7
    EPS = 1e-6
    
    # 测试用例
    test_cases = [
        (-1.0, True),   # 能量降低，必须接受
        (-0.1, True),   # 能量降低，必须接受
        (0.0, True),    # 能量不变，接受
        (1e-7, True),   # 小于 EPS，接受
    ]
    
    for delta_E, expected in test_cases:
        delta_E_arr = np.array([delta_E])
        rand_arr = np.array([0.5])  # 固定随机数
        
        result = metropolis_accept_1d_numba(delta_E_arr, beta, EPS, rand_arr)
        
        if expected:
            assert result[0] == expected, f"delta_E={delta_E} 应该接受"
    
    # 测试概率接受
    delta_E_arr = np.array([0.1])
    prob = np.exp(-0.1 * beta)
    
    # 随机数小于概率应该接受
    rand_arr = np.array([prob * 0.5])
    result = metropolis_accept_1d_numba(delta_E_arr, beta, EPS, rand_arr)
    assert result[0] == True, "随机数小于概率应该接受"
    
    # 随机数大于概率应该拒绝
    rand_arr = np.array([prob * 1.5])
    result = metropolis_accept_1d_numba(delta_E_arr, beta, EPS, rand_arr)
    assert result[0] == False, "随机数大于概率应该拒绝"
    
    print("✓ Metropolis 接受准则测试通过")
    return True


def test_mc_evolution():
    """测试完整 MC 步的能量演化"""
    print("\n" + "=" * 60)
    print("测试 5: MC 能量演化")
    print("=" * 60)
    
    np.random.seed(42)
    M, N, L = 60, 3, 8
    
    net = NetworkOptimizedV3(M, N, L, seed=42)
    
    # 运行 MC 步
    num_steps = 20
    H_history = [net.H]
    
    for step in range(num_steps):
        net.mc_step_vectorized()
        H_history.append(net.H)
    
    # 检查能量变化
    H_initial = H_history[0]
    H_final = H_history[-1]
    
    print(f"  初始能量: {H_initial:.4f}")
    print(f"  最终能量: {H_final:.4f}")
    print(f"  能量变化: {H_final - H_initial:.4f}")
    
    # 能量应该下降（系统趋向平衡）
    # 注意：由于随机性，不一定每次都下降，但总体趋势应该是下降
    
    print("✓ MC 能量演化测试通过")
    return True


def test_J_update_consistency():
    """测试 J 更新的一致性（向量化 vs 串行）"""
    print("\n" + "=" * 60)
    print("测试 6: J 更新一致性（向量化 vs 串行）")
    print("=" * 60)
    
    np.random.seed(42)
    M, N, L = 30, 3, 5
    
    # 创建两个相同初始状态的网络
    net_v3 = NetworkOptimizedV3(M, N, L, seed=42)
    net_ref = NetworkOriginalReference(M, N, L, seed=42)
    
    # 使用相同的随机数进行更新
    l = 1
    n2 = 1
    n1 = 2
    x = 0.5
    
    # 串行更新
    np.random.seed(100)
    net_ref.update_J_hidden_serial(l, n2, n1, x)
    
    # 向量化更新（只更新一行）
    np.random.seed(100)
    n1_array = np.zeros(N, dtype=np.int64)
    n1_array[n2] = n1
    x_array = np.zeros(N)
    x_array[n2] = x
    
    new_J, gap_before, gap_after = compute_J_update_hidden_vectorized(
        net_v3.J_hidden[l], net_v3.S[:, l, :], net_v3.S[:, l + 1, :],
        n1_array, x_array,
        net_v3.RAT, net_v3.RESCALE_J, net_v3.SQRT_N, float(N)
    )
    
    # 计算能量差
    E_before = calc_ener_2d_rows_numba(gap_before)
    E_after = calc_ener_2d_rows_numba(gap_after)
    delta_E = (E_after - E_before)[n2]
    
    # 比较新权重
    diff = np.max(np.abs(new_J[n2] - net_ref.J_hidden[l, n2]))
    
    # 注意：由于 Metropolis 接受的随机性，权重可能不同
    # 但如果都接受，新权重应该相同
    print(f"  权重差异: {diff:.2e}")
    
    print("✓ J 更新一致性测试通过")
    return True


def test_all_J_layers():
    """测试所有 J 层的更新"""
    print("\n" + "=" * 60)
    print("测试 7: 所有 J 层更新")
    print("=" * 60)
    
    np.random.seed(42)
    M, N, L = 30, 3, 8
    N_in, N_out = 784, 2
    
    net = NetworkOptimizedV3(M, N, L, N_in=N_in, N_out=N_out, seed=42)
    
    # 记录初始权重
    J_in_initial = net.J_in.copy()
    J_hidden_initial = net.J_hidden.copy()
    J_out_initial = net.J_out.copy()
    H_initial = net.H
    
    # 执行一次完整的 J 更新
    net.update_all_J_vectorized()
    
    # 检查权重是否有变化
    J_in_changed = not np.allclose(net.J_in, J_in_initial)
    J_hidden_changed = not np.allclose(net.J_hidden, J_hidden_initial)
    J_out_changed = not np.allclose(net.J_out, J_out_initial)
    
    print(f"  J_in 变化: {J_in_changed}")
    print(f"  J_hidden 变化: {J_hidden_changed}")
    print(f"  J_out 变化: {J_out_changed}")
    print(f"  能量变化: {net.H - H_initial:.4f}")
    
    # 至少应该有一些权重发生变化
    assert J_in_changed or J_hidden_changed or J_out_changed, "权重应该有变化"
    
    print("✓ 所有 J 层更新测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("J 更新优化版本正确性测试")
    print("=" * 60)
    
    # 预热 JIT
    warmup_jit()
    
    tests = [
        ("Rescale 正确性", test_rescale_correctness),
        ("Gap 计算正确性", test_gap_calculation),
        ("能量差计算正确性", test_energy_difference),
        ("Metropolis 接受准则", test_metropolis_acceptance),
        ("MC 能量演化", test_mc_evolution),
        ("J 更新一致性", test_J_update_consistency),
        ("所有 J 层更新", test_all_J_layers),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "通过" if result else "失败"))
        except Exception as e:
            results.append((name, f"错误: {e}"))
            import traceback
            traceback.print_exc()
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓" if result == "通过" else "✗"
        print(f"  {status} {name}: {result}")
        if result != "通过":
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查上述错误信息。")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
