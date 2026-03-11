#!/usr/bin/env python3
"""
SVD图像压缩测试脚本
演示不同压缩参数的效果
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# 添加当前目录到路径以便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svd_compress import SVDImageCompressor

def create_test_image(output_path: str = "test_image.png"):
    """创建测试图像"""
    print("创建测试图像...")
    
    # 创建一个简单的测试图像
    width, height = 256, 256
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加一些特征
    # 红色矩形
    img_array[50:100, 50:150, 0] = 255
    # 绿色矩形
    img_array[100:150, 100:200, 1] = 255
    # 蓝色矩形
    img_array[150:200, 50:150, 2] = 255
    
    # 渐变背景
    for y in range(height):
        for x in range(width):
            gray = (x + y) // 4
            img_array[y, x, :] += gray
    
    # 保存图像
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"测试图像已保存: {output_path}")
    
    return img_array

def test_basic_compression():
    """测试基本压缩功能"""
    print("\n" + "="*50)
    print("测试基本压缩功能")
    print("="*50)
    
    # 创建压缩器
    compressor = SVDImageCompressor(verbose=True)
    
    # 创建或加载测试图像
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        original_image = create_test_image(test_image_path)
    else:
        original_image = compressor.load_image(test_image_path)
    
    # 测试不同秩的压缩
    ranks = [10, 30, 50, 100]
    
    for rank in ranks:
        print(f"\n测试秩 = {rank}")
        print("-" * 30)
        
        # 压缩图像
        compressed_image, info = compressor.compress_image(original_image, rank)
        
        # 保存结果
        output_path = f"compressed_rank{rank}.png"
        compressor.save_image(compressed_image, output_path)
        
        # 显示信息
        print(f"压缩比: {info['compression_ratio']:.4f}")
        print(f"PSNR: {info['metrics'].get('psnr', 0):.2f} dB")
        if 'ssim' in info['metrics']:
            print(f"SSIM: {info['metrics']['ssim']:.4f}")
        
        # 计算文件大小
        original_size = os.path.getsize(test_image_path)
        compressed_size = os.path.getsize(output_path)
        size_ratio = compressed_size / original_size
        
        print(f"文件大小比: {size_ratio:.4f}")
        print(f"原始大小: {original_size / 1024:.2f} KB")
        print(f"压缩大小: {compressed_size / 1024:.2f} KB")

def test_compression_by_ratio():
    """测试按压缩比压缩"""
    print("\n" + "="*50)
    print("测试按压缩比压缩")
    print("="*50)
    
    compressor = SVDImageCompressor(verbose=True)
    
    # 加载测试图像
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        create_test_image(test_image_path)
    
    original_image = compressor.load_image(test_image_path)
    
    # 测试不同压缩比
    target_ratios = [0.1, 0.3, 0.5, 0.7]
    
    for ratio in target_ratios:
        print(f"\n目标压缩比 = {ratio}")
        print("-" * 30)
        
        # 按压缩比压缩
        compressed_image, info = compressor.compress_by_ratio(original_image, ratio)
        
        # 保存结果
        output_path = f"compressed_ratio{ratio:.2f}.png"
        compressor.save_image(compressed_image, output_path)
        
        # 显示信息
        print(f"实际压缩比: {info['compression_ratio']:.4f}")
        print(f"使用的秩: {info['rank']}")
        print(f"PSNR: {info['metrics'].get('psnr', 0):.2f} dB")

def test_compression_by_quality():
    """测试按质量压缩"""
    print("\n" + "="*50)
    print("测试按质量压缩")
    print("="*50)
    
    compressor = SVDImageCompressor(verbose=True)
    
    # 加载测试图像
    test_image_path = "test_image.png"
    if not os.path.exists(test_image_path):
        create_test_image(test_image_path)
    
    original_image = compressor.load_image(test_image_path)
    
    # 测试不同质量要求
    target_qualities = [0.7, 0.8, 0.9, 0.95]
    
    for quality in target_qualities:
        print(f"\n目标质量 = {quality}")
        print("-" * 30)
        
        # 按质量压缩
        compressed_image, info = compressor.compress_by_quality(original_image, quality)
        
        # 保存结果
        output_path = f"compressed_quality{quality:.2f}.png"
        compressor.save_image(compressed_image, output_path)
        
        # 显示信息
        print(f"实际SSIM: {info['metrics'].get('ssim', 0):.4f}")
        print(f"使用的秩: {info['rank']}")
        print(f"压缩比: {info['compression_ratio']:.4f}")
        print(f"PSNR: {info['metrics'].get('psnr', 0):.2f} dB")

def test_grayscale_image():
    """测试灰度图像压缩"""
    print("\n" + "="*50)
    print("测试灰度图像压缩")
    print("="*50)
    
    compressor = SVDImageCompressor(verbose=True)
    
    # 创建灰度测试图像
    width, height = 256, 256
    gray_image = np.zeros((height, width), dtype=np.uint8)
    
    # 创建渐变图案
    for y in range(height):
        for x in range(width):
            gray_image[y, x] = (x * y) // 256
    
    # 添加一些特征
    gray_image[50:100, 50:150] = 255  # 白色矩形
    gray_image[150:200, 100:200] = 128  # 灰色矩形
    
    # 保存灰度图像
    gray_path = "test_gray.png"
    Image.fromarray(gray_image).save(gray_path)
    print(f"灰度测试图像已保存: {gray_path}")
    
    # 测试压缩
    ranks = [10, 30, 50]
    
    for rank in ranks:
        print(f"\n测试秩 = {rank}")
        print("-" * 30)
        
        compressed_image, info = compressor.compress_image(gray_image, rank)
        
        output_path = f"gray_compressed_rank{rank}.png"
        compressor.save_image(compressed_image, output_path)
        
        print(f"压缩比: {info['compression_ratio']:.4f}")
        print(f"PSNR: {info['metrics'].get('psnr', 0):.2f} dB")
        if 'ssim' in info['metrics']:
            print(f"SSIM: {info['metrics']['ssim']:.4f}")

def test_visualization():
    """测试可视化功能"""
    print("\n" + "="*50)
    print("测试可视化功能")
    print("="*50)
    
    try:
        import matplotlib.pyplot as plt
        
        compressor = SVDImageCompressor(verbose=True)
        
        # 加载测试图像
        test_image_path = "test_image.png"
        if not os.path.exists(test_image_path):
            create_test_image(test_image_path)
        
        original_image = compressor.load_image(test_image_path)
        
        # 压缩图像
        compressed_image, info = compressor.compress_image(original_image, 30)
        
        # 生成对比图
        print("生成对比可视化图...")
        compressor.visualize_comparison(
            original_image, 
            compressed_image,
            "visualization_comparison.png"
        )
        
        print("可视化图已保存: visualization_comparison.png")
        
    except ImportError:
        print("matplotlib未安装，跳过可视化测试")
    except Exception as e:
        print(f"可视化测试失败: {e}")

def main():
    """主测试函数"""
    print("SVD图像压缩测试")
    print("="*50)
    
    # 运行各个测试
    test_basic_compression()
    test_compression_by_ratio()
    test_compression_by_quality()
    test_grayscale_image()
    test_visualization()
    
    print("\n" + "="*50)
    print("所有测试完成!")
    print("="*50)
    
    # 清理建议
    print("\n生成的文件:")
    for file in os.listdir("."):
        if file.startswith(("test_", "compressed_", "gray_", "visualization_")):
            print(f"  {file}")
    
    print("\n可以使用以下命令清理测试文件:")
    print("rm -f test_*.png compressed_*.png gray_*.png visualization_*.png")

if __name__ == "__main__":
    main()