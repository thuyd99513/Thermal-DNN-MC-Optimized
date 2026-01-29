#!/usr/bin/env python3
"""
mnist_loader.py - MNIST 数据加载模块

根据论文 "Liquid and solid layers in a thermal deep learning machine" 的要求:
- 加载 MNIST 数据集
- 筛选数字 "0" 和 "1" (二分类)
- 将像素值二值化为 ±1
- 返回训练集 (2000 张) 和测试集 (400 张)

作者：Manus AI
日期：2026-01-29
"""

import numpy as np
import os
import gzip
import urllib.request
from pathlib import Path


# MNIST 数据集 URL
MNIST_URLS = {
    'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
}

# 备用 URL (如果原始 URL 不可用)
MNIST_URLS_BACKUP = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
}


def download_mnist(data_dir='./data/mnist'):
    """
    下载 MNIST 数据集
    
    Args:
        data_dir: 数据保存目录
    
    Returns:
        dict: 包含各文件路径的字典
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    files = {}
    for name, url in MNIST_URLS.items():
        filepath = data_dir / f'{name}.gz'
        files[name] = filepath
        
        if filepath.exists():
            print(f"  {name}: 已存在")
            continue
        
        print(f"  下载 {name}...")
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"    主 URL 失败，尝试备用 URL...")
            try:
                urllib.request.urlretrieve(MNIST_URLS_BACKUP[name], filepath)
            except Exception as e2:
                raise RuntimeError(f"下载失败: {e2}")
        
        print(f"    完成")
    
    return files


def load_mnist_images(filepath):
    """加载 MNIST 图像文件"""
    with gzip.open(filepath, 'rb') as f:
        # 读取文件头
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # 读取图像数据
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(num_images, rows * cols)
    
    return images


def load_mnist_labels(filepath):
    """加载 MNIST 标签文件"""
    with gzip.open(filepath, 'rb') as f:
        # 读取文件头
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels


def binarize_images(images, threshold=127):
    """
    将图像二值化为 ±1
    
    Args:
        images: shape (N, 784) 的图像数组，值范围 0-255
        threshold: 二值化阈值
    
    Returns:
        shape (N, 784) 的二值化图像，值为 ±1
    """
    # 像素值 > threshold → +1 (白)
    # 像素值 <= threshold → -1 (黑)
    return np.where(images > threshold, 1.0, -1.0).astype(np.float64)


def binarize_labels(labels):
    """
    将标签二值化为 ±1
    
    Args:
        labels: 标签数组，值为 0 或 1
    
    Returns:
        二值化标签，0 → -1, 1 → +1
    """
    return np.where(labels == 0, -1.0, 1.0).astype(np.float64)


def load_mnist_binary(data_dir='./data/mnist', 
                      num_train=2000, 
                      num_test=400,
                      digits=(0, 1),
                      seed=42,
                      verbose=True):
    """
    加载并处理 MNIST 数据集，用于 TDLM 模拟
    
    根据论文要求:
    - 只使用数字 0 和 1
    - 像素二值化为 ±1
    - 训练集 2000 张，测试集 400 张
    
    Args:
        data_dir: 数据目录
        num_train: 训练样本数 (默认 2000)
        num_test: 测试样本数 (默认 400)
        digits: 要使用的数字类别 (默认 (0, 1))
        seed: 随机种子 (用于打乱数据)
        verbose: 是否打印详细信息
    
    Returns:
        dict: {
            'train_images': shape (num_train, 784) 的训练图像，值为 ±1
            'train_labels': shape (num_train,) 的训练标签，值为 ±1
            'test_images': shape (num_test, 784) 的测试图像，值为 ±1
            'test_labels': shape (num_test,) 的测试标签，值为 ±1
        }
    """
    if verbose:
        print("加载 MNIST 数据集...")
    
    # 下载数据
    files = download_mnist(data_dir)
    
    # 加载原始数据
    train_images_raw = load_mnist_images(files['train_images'])
    train_labels_raw = load_mnist_labels(files['train_labels'])
    test_images_raw = load_mnist_images(files['test_images'])
    test_labels_raw = load_mnist_labels(files['test_labels'])
    
    if verbose:
        print(f"  原始训练集: {len(train_images_raw)} 张")
        print(f"  原始测试集: {len(test_images_raw)} 张")
    
    # 筛选指定数字
    digit_0, digit_1 = digits
    
    train_mask = (train_labels_raw == digit_0) | (train_labels_raw == digit_1)
    test_mask = (test_labels_raw == digit_0) | (test_labels_raw == digit_1)
    
    train_images_filtered = train_images_raw[train_mask]
    train_labels_filtered = train_labels_raw[train_mask]
    test_images_filtered = test_images_raw[test_mask]
    test_labels_filtered = test_labels_raw[test_mask]
    
    if verbose:
        print(f"  筛选数字 {digits} 后:")
        print(f"    训练集: {len(train_images_filtered)} 张")
        print(f"    测试集: {len(test_images_filtered)} 张")
    
    # 打乱数据
    np.random.seed(seed)
    train_perm = np.random.permutation(len(train_images_filtered))
    test_perm = np.random.permutation(len(test_images_filtered))
    
    train_images_shuffled = train_images_filtered[train_perm]
    train_labels_shuffled = train_labels_filtered[train_perm]
    test_images_shuffled = test_images_filtered[test_perm]
    test_labels_shuffled = test_labels_filtered[test_perm]
    
    # 选取指定数量
    if num_train > len(train_images_shuffled):
        raise ValueError(f"请求 {num_train} 张训练图像，但只有 {len(train_images_shuffled)} 张可用")
    if num_test > len(test_images_shuffled):
        raise ValueError(f"请求 {num_test} 张测试图像，但只有 {len(test_images_shuffled)} 张可用")
    
    train_images = train_images_shuffled[:num_train]
    train_labels = train_labels_shuffled[:num_train]
    test_images = test_images_shuffled[:num_test]
    test_labels = test_labels_shuffled[:num_test]
    
    # 二值化
    train_images_binary = binarize_images(train_images)
    test_images_binary = binarize_images(test_images)
    
    # 标签转换: 0 → -1, 1 → +1
    train_labels_binary = binarize_labels(train_labels)
    test_labels_binary = binarize_labels(test_labels)
    
    if verbose:
        print(f"  最终数据集:")
        print(f"    训练集: {num_train} 张, 标签分布: "
              f"{np.sum(train_labels_binary == -1)} 个 '{digit_0}', "
              f"{np.sum(train_labels_binary == 1)} 个 '{digit_1}'")
        print(f"    测试集: {num_test} 张, 标签分布: "
              f"{np.sum(test_labels_binary == -1)} 个 '{digit_0}', "
              f"{np.sum(test_labels_binary == 1)} 个 '{digit_1}'")
        print(f"  图像值范围: [{train_images_binary.min()}, {train_images_binary.max()}]")
        print(f"  标签值范围: [{train_labels_binary.min()}, {train_labels_binary.max()}]")
    
    return {
        'train_images': train_images_binary,  # shape (num_train, 784), values ±1
        'train_labels': train_labels_binary,  # shape (num_train,), values ±1
        'test_images': test_images_binary,    # shape (num_test, 784), values ±1
        'test_labels': test_labels_binary,    # shape (num_test,), values ±1
    }


def prepare_boundary_conditions(mnist_data, N_out=2):
    """
    准备 TDLM 模拟的边界条件
    
    Args:
        mnist_data: load_mnist_binary() 返回的数据字典
        N_out: 输出层维度 (默认 2)
    
    Returns:
        dict: {
            'S_in': shape (M, 784) 的输入边界条件
            'S_out': shape (M, N_out) 的输出边界条件
            'M': 样本数
            'N_in': 输入维度 (784)
        }
    """
    train_images = mnist_data['train_images']  # shape (M, 784)
    train_labels = mnist_data['train_labels']  # shape (M,)
    
    M = len(train_images)
    N_in = train_images.shape[1]  # 784
    
    # S_in: 输入图像作为边界条件
    S_in = train_images.copy()  # shape (M, 784)
    
    # S_out: 输出标签作为边界条件
    # 论文中 N_out = 2，需要将标签扩展为 (M, 2)
    # 一种常见做法: 第一个神经元表示类别 0，第二个表示类别 1
    # 或者简单地复制标签
    if N_out == 1:
        S_out = train_labels.reshape(-1, 1)
    elif N_out == 2:
        # 使用 one-hot 风格: label=-1 → [-1, +1], label=+1 → [+1, -1]
        S_out = np.zeros((M, 2), dtype=np.float64)
        S_out[:, 0] = train_labels  # 第一个神经元 = 标签
        S_out[:, 1] = -train_labels  # 第二个神经元 = 反标签
    else:
        # 复制标签到所有输出神经元
        S_out = np.tile(train_labels.reshape(-1, 1), (1, N_out))
    
    return {
        'S_in': S_in,
        'S_out': S_out,
        'M': M,
        'N_in': N_in,
    }


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MNIST 数据加载测试")
    print("=" * 60)
    
    # 加载数据
    mnist_data = load_mnist_binary(
        data_dir='./data/mnist',
        num_train=2000,
        num_test=400,
        digits=(0, 1),
        seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("准备边界条件")
    print("=" * 60)
    
    # 准备边界条件
    boundary = prepare_boundary_conditions(mnist_data, N_out=2)
    
    print(f"\nS_in shape: {boundary['S_in'].shape}")
    print(f"S_out shape: {boundary['S_out'].shape}")
    print(f"M: {boundary['M']}")
    print(f"N_in: {boundary['N_in']}")
    
    # 验证数据
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)
    
    print(f"S_in 唯一值: {np.unique(boundary['S_in'])}")
    print(f"S_out 唯一值: {np.unique(boundary['S_out'])}")
    
    # 显示一个样本
    print("\n第一个样本:")
    print(f"  S_in[0] 非零元素数: {np.sum(boundary['S_in'][0] != 0)}")
    print(f"  S_in[0] +1 数量: {np.sum(boundary['S_in'][0] == 1)}")
    print(f"  S_in[0] -1 数量: {np.sum(boundary['S_in'][0] == -1)}")
    print(f"  S_out[0]: {boundary['S_out'][0]}")
    
    print("\n测试完成!")
