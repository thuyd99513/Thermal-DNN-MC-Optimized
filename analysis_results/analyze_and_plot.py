#!/usr/bin/env python3
"""
Thermal DNN MC Simulation - Experiment 1 Analysis
绑制 Figure 1b 风格的相图热图，分析固-液-固分层结构
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# 设置中文字体支持
matplotlib.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据目录
DATA_DIR = Path("/home/ubuntu/analysis/experiment1_paper_20260130")
OUTPUT_DIR = Path("/home/ubuntu/analysis/output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_all_results():
    """加载所有实验结果"""
    results = {}
    
    # 加载配置
    with open(DATA_DIR / "config.json", 'r') as f:
        config = json.load(f)
    
    N_values = config['N_values']
    num_samples = config['num_samples']
    L = config['L']
    
    # 加载所有样本结果
    for N in N_values:
        results[N] = []
        for sample_idx in range(num_samples):
            filename = f"sample_{sample_idx:02d}_N_{N:02d}_result.json"
            filepath = DATA_DIR / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[N].append(data)
    
    return config, results

def compute_phase_diagram_data(config, results):
    """计算相图数据"""
    N_values = config['N_values']
    L = config['L']
    M = config['M']
    
    # 计算 ln(alpha) = ln(M/N)
    ln_alpha_values = [np.log(M / N) for N in N_values]
    
    # 层数（不包括输入层，共 L-1 个隐藏层连接）
    num_layers = L - 1  # 9 layers
    
    # 初始化数据矩阵
    q_star_matrix = np.zeros((len(N_values), num_layers))
    q_star_std_matrix = np.zeros((len(N_values), num_layers))
    
    for i, N in enumerate(N_values):
        sample_results = results[N]
        if sample_results:
            # 收集所有样本的 q_l_star
            all_q_l = []
            for sample in sample_results:
                q_l = sample.get('q_l_star', [])
                if len(q_l) == num_layers:
                    all_q_l.append(q_l)
            
            if all_q_l:
                all_q_l = np.array(all_q_l)
                # 跨样本平均
                q_star_matrix[i, :] = np.mean(all_q_l, axis=0)
                q_star_std_matrix[i, :] = np.std(all_q_l, axis=0)
    
    return ln_alpha_values, q_star_matrix, q_star_std_matrix

def plot_figure_1b_style(ln_alpha_values, q_star_matrix, config):
    """绑制 Figure 1b 风格的相图热图"""
    L = config['L']
    num_layers = L - 1
    
    # 相变边界
    q_critical = 1 / np.e  # ≈ 0.368
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建热图数据
    # X轴: 层索引 (1 to L-1)
    # Y轴: ln(alpha)
    layer_indices = np.arange(1, num_layers + 1)
    
    # 绑制热图
    im = ax.imshow(q_star_matrix, aspect='auto', cmap='RdYlBu_r',
                   vmin=0, vmax=1,
                   extent=[0.5, num_layers + 0.5, 
                          ln_alpha_values[-1] - 0.5, ln_alpha_values[0] + 0.5])
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label=r'$q^*_\ell$')
    cbar.ax.axhline(y=q_critical, color='white', linewidth=2, linestyle='--')
    cbar.ax.text(1.5, q_critical, r'$q^* = 1/e$', va='center', fontsize=10, color='white')
    
    # 标注固态/液态区域
    for i, ln_alpha in enumerate(ln_alpha_values):
        for j in range(num_layers):
            q_val = q_star_matrix[i, j]
            # 根据相变边界判断固态/液态
            if q_val > q_critical:
                marker = 'S'  # Solid
                color = 'white' if q_val > 0.6 else 'black'
            else:
                marker = 'L'  # Liquid
                color = 'black' if q_val > 0.3 else 'white'
            
            # 在每个格子中标注数值
            ax.text(j + 1, ln_alpha, f'{q_val:.2f}', 
                   ha='center', va='center', fontsize=8, color=color)
    
    # 设置坐标轴
    ax.set_xlabel(r'Layer index $\ell$', fontsize=14)
    ax.set_ylabel(r'$\ln(\alpha) = \ln(M/N)$', fontsize=14)
    ax.set_title('Phase Diagram: Overlap Parameter $q^*_\\ell$ vs Layer and $\\ln(\\alpha)$\n(Figure 1b Style)', 
                fontsize=14)
    
    # 设置刻度
    ax.set_xticks(layer_indices)
    ax.set_xticklabels([f'{l}' for l in layer_indices])
    ax.set_yticks(ln_alpha_values)
    ax.set_yticklabels([f'{la:.2f}' for la in ln_alpha_values])
    
    # 添加相变边界线
    # 找到每个 ln_alpha 下的相变位置
    for i, ln_alpha in enumerate(ln_alpha_values):
        for j in range(num_layers - 1):
            q1, q2 = q_star_matrix[i, j], q_star_matrix[i, j + 1]
            if (q1 - q_critical) * (q2 - q_critical) < 0:
                # 相变发生在 j 和 j+1 之间
                x_cross = j + 1 + (q_critical - q1) / (q2 - q1)
                ax.axvline(x=x_cross, ymin=0, ymax=1, color='lime', 
                          linewidth=1, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'figure_1b_phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'figure_1b_phase_diagram.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'figure_1b_phase_diagram.png'}")
    
    return fig

def plot_layer_profile(ln_alpha_values, q_star_matrix, q_star_std_matrix, config):
    """绑制各层 q* 的剖面图"""
    L = config['L']
    N_values = config['N_values']
    M = config['M']
    num_layers = L - 1
    
    q_critical = 1 / np.e
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layer_indices = np.arange(1, num_layers + 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
    
    for i, (N, color) in enumerate(zip(N_values, colors)):
        alpha = M / N
        label = f'N={N}, α={alpha:.0f}'
        ax.errorbar(layer_indices, q_star_matrix[i, :], 
                   yerr=q_star_std_matrix[i, :],
                   marker='o', markersize=8, linewidth=2,
                   color=color, label=label, capsize=3)
    
    # 添加相变边界线
    ax.axhline(y=q_critical, color='red', linewidth=2, linestyle='--', 
              label=f'Phase boundary $q^* = 1/e ≈ {q_critical:.3f}$')
    
    # 标注固态/液态区域
    ax.fill_between([0, num_layers + 1], q_critical, 1, alpha=0.1, color='blue', label='Solid phase')
    ax.fill_between([0, num_layers + 1], 0, q_critical, alpha=0.1, color='orange', label='Liquid phase')
    
    ax.set_xlabel(r'Layer index $\ell$', fontsize=14)
    ax.set_ylabel(r'Overlap parameter $q^*_\ell$', fontsize=14)
    ax.set_title('Layer-wise Overlap Parameter Profile\n(Solid-Liquid-Solid Structure)', fontsize=14)
    ax.set_xlim(0.5, num_layers + 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(layer_indices)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'layer_profile.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'layer_profile.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'layer_profile.png'}")
    
    return fig

def analyze_phase_structure(ln_alpha_values, q_star_matrix, config):
    """分析固-液-固分层结构"""
    L = config['L']
    N_values = config['N_values']
    M = config['M']
    num_layers = L - 1
    q_critical = 1 / np.e
    
    print("\n" + "="*70)
    print("固-液-固分层结构分析")
    print("="*70)
    print(f"\n相变边界: q* = 1/e ≈ {q_critical:.4f}")
    print(f"  - q* > {q_critical:.3f}: 固态 (Solid) - 配置被约束")
    print(f"  - q* < {q_critical:.3f}: 液态 (Liquid) - 配置自由")
    
    analysis_results = []
    
    for i, N in enumerate(N_values):
        alpha = M / N
        ln_alpha = ln_alpha_values[i]
        q_values = q_star_matrix[i, :]
        
        print(f"\n--- N = {N}, α = {alpha:.0f}, ln(α) = {ln_alpha:.2f} ---")
        
        # 判断每层的相态
        phases = []
        for j, q in enumerate(q_values):
            phase = "固态" if q > q_critical else "液态"
            phases.append(phase)
            print(f"  层 {j+1}: q* = {q:.4f} → {phase}")
        
        # 统计固态和液态层数
        solid_count = sum(1 for p in phases if p == "固态")
        liquid_count = sum(1 for p in phases if p == "液态")
        
        # 检查是否呈现"固-液-固"结构
        first_layer_solid = phases[0] == "固态"
        last_layer_solid = phases[-1] == "固态"
        middle_has_liquid = any(p == "液态" for p in phases[1:-1])
        
        structure = ""
        if first_layer_solid and last_layer_solid and middle_has_liquid:
            structure = "固-液-固 (Solid-Liquid-Solid)"
        elif solid_count == num_layers:
            structure = "全固态 (All Solid)"
        elif liquid_count == num_layers:
            structure = "全液态 (All Liquid)"
        else:
            structure = "混合结构"
        
        print(f"  结构: {structure}")
        print(f"  固态层: {solid_count}, 液态层: {liquid_count}")
        
        analysis_results.append({
            'N': N,
            'alpha': alpha,
            'ln_alpha': ln_alpha,
            'q_values': q_values.tolist(),
            'phases': phases,
            'structure': structure,
            'solid_count': solid_count,
            'liquid_count': liquid_count
        })
    
    return analysis_results

def compare_with_paper():
    """与论文结果进行对比"""
    print("\n" + "="*70)
    print("与论文结果对比")
    print("="*70)
    
    print("""
论文主要发现 (G. Huang et al., arXiv:2506.06789):

1. 固-液-固分层结构:
   - 边界层（靠近输入/输出）表现为固态 (q* > 1/e)
   - 中间层表现为液态 (q* < 1/e)
   - 这种结构在不同 α 值下都存在

2. 相变边界:
   - 理论预测: q* = 1/e ≈ 0.368
   - 实验验证: 边界层 q* ≈ 0.7-0.8, 中间层 q* ≈ 0.3-0.5

3. α 依赖性:
   - 较大的 α (小 N) 导致更明显的固态行为
   - 较小的 α (大 N) 导致更多液态层

4. 层依赖性:
   - 第一层（靠近输入）通常最固态
   - 最后一层（靠近输出）也呈固态
   - 中间层呈液态，但程度随层位置变化
""")

def main():
    print("="*70)
    print("Thermal DNN MC Simulation - Experiment 1 Analysis")
    print("="*70)
    
    # 加载数据
    print("\n加载实验数据...")
    config, results = load_all_results()
    print(f"配置: L={config['L']}, M={config['M']}, N_values={config['N_values']}")
    print(f"MC步数: {config['mc_steps']}, 样本数: {config['num_samples']}")
    
    # 计算相图数据
    print("\n计算相图数据...")
    ln_alpha_values, q_star_matrix, q_star_std_matrix = compute_phase_diagram_data(config, results)
    
    # 绑制图表
    print("\n绑制 Figure 1b 风格相图...")
    plot_figure_1b_style(ln_alpha_values, q_star_matrix, config)
    
    print("\n绑制层剖面图...")
    plot_layer_profile(ln_alpha_values, q_star_matrix, q_star_std_matrix, config)
    
    # 分析固-液-固结构
    analysis_results = analyze_phase_structure(ln_alpha_values, q_star_matrix, config)
    
    # 与论文对比
    compare_with_paper()
    
    # 保存分析结果
    with open(OUTPUT_DIR / 'analysis_results.json', 'w') as f:
        json.dump({
            'config': config,
            'ln_alpha_values': ln_alpha_values,
            'q_star_matrix': q_star_matrix.tolist(),
            'q_star_std_matrix': q_star_std_matrix.tolist(),
            'analysis': analysis_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存: {OUTPUT_DIR / 'analysis_results.json'}")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == "__main__":
    main()
