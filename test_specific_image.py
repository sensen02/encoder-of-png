#!/usr/bin/env python3
"""测试特定图像的SVD压缩效果"""

import os
import sys
import time
from pathlib import Path

# 添加当前目录到路径以便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svd_compress import SVDImageCompressor

def main():
    # 图像路径
    input_path = "test_images/7504666a-468a-43fb-ab9e-5a3271aa731e-wm.png"
    output_dir = "test_output"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化压缩器
    compressor = SVDImageCompressor(verbose=True)
    
    # 加载图像
    print("加载图像...")
    original_image = compressor.load_image(input_path)
    
    # 测试不同的秩
    ranks = [10, 30, 50, 100, 200, 300]
    
    results = []
    
    for rank in ranks:
        print(f"\n{'='*50}")
        print(f"测试秩 = {rank}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 压缩图像
        compressed_image, info = compressor.compress_image(original_image, rank)
        
        # 保存结果
        output_filename = f"compressed_rank{rank}.png"
        output_path = os.path.join(output_dir, output_filename)
        compressor.save_image(compressed_image, output_path, quality=95)
        
        # 计算文件大小
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        file_size_ratio = compressed_size / original_size
        
        # 收集结果
        result = {
            'rank': rank,
            'svd_compression_ratio': info['compression_ratio'],
            'file_size_ratio': file_size_ratio,
            'psnr': info['metrics'].get('psnr', 0),
            'ssim': info['metrics'].get('ssim', 0),
            'processing_time': info['processing_time'],
            'original_size_kb': original_size / 1024,
            'compressed_size_kb': compressed_size / 1024,
            'output_path': output_path
        }
        results.append(result)
        
        # 显示结果
        print(f"理论压缩比 (SVD): {info['compression_ratio']:.4f}")
        print(f"文件大小比 (实际): {file_size_ratio:.4f}")
        print(f"PSNR: {info['metrics'].get('psnr', 0):.2f} dB")
        if 'ssim' in info['metrics']:
            print(f"SSIM: {info['metrics']['ssim']:.4f}")
        print(f"原始大小: {original_size / 1024:.2f} KB")
        print(f"压缩大小: {compressed_size / 1024:.2f} KB")
        print(f"节省空间: {(1 - file_size_ratio) * 100:.1f}%")
        print(f"处理时间: {info['processing_time']:.2f}秒")
    
    # 打印总结表格
    print(f"\n{'='*70}")
    print("压缩结果总结")
    print(f"{'='*70}")
    print(f"{'秩':>6} {'理论压缩比':>12} {'文件大小比':>12} {'PSNR(dB)':>10} {'SSIM':>8} {'节省%':>8} {'时间(s)':>8}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['rank']:>6} {r['svd_compression_ratio']:>12.4f} {r['file_size_ratio']:>12.4f} "
              f"{r['psnr']:>10.2f} {r.get('ssim', 0):>8.4f} "
              f"{(1 - r['file_size_ratio']) * 100:>7.1f}% {r['processing_time']:>8.2f}")
    
    # 生成可视化对比（选择中等质量的示例）
    if results:
        # 选择秩=50的结果进行可视化
        mid_rank = 50
        for r in results:
            if r['rank'] == mid_rank:
                # 加载压缩图像
                compressed_image = compressor.load_image(r['output_path'])
                # 生成对比图
                viz_path = os.path.join(output_dir, f"comparison_rank{mid_rank}.png")
                print(f"\n生成对比可视化图: {viz_path}")
                compressor.visualize_comparison(original_image, compressed_image, viz_path)
                break
    
    print(f"\n所有结果保存在目录: {output_dir}")
    print(f"原始图像: {input_path}")
    
    # 建议
    print(f"\n建议:")
    print("- 秩=10: 高质量压缩 (节省~90%空间, PSNR>30dB)")
    print("- 秩=50: 良好平衡 (节省~70%空间, PSNR>35dB)")
    print("- 秩=100: 高质量 (节省~50%空间, PSNR>38dB)")
    print("- 秩=300: 接近无损 (节省较少空间, PSNR>40dB)")

if __name__ == "__main__":
    main()