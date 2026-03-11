#!/usr/bin/env python3
"""测试JPEG格式的SVD压缩效果"""

import os
import sys
from PIL import Image
import numpy as np

# 添加当前目录到路径以便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svd_compress import SVDImageCompressor

def main():
    # 图像路径
    input_path = "test_images/7504666a-468a-43fb-ab9e-5a3271aa731e-wm.png"
    
    # 加载图像并转换为RGB（忽略alpha）
    print("加载图像并转换为RGB...")
    img = Image.open(input_path)
    
    # 检查是否需要转换
    if img.mode == 'RGBA':
        print("图像为RGBA模式，转换为RGB...")
        # 创建白色背景的RGB图像
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
        img_array = np.array(rgb_img)
    else:
        img_array = np.array(img.convert("RGB"))
    
    print(f"转换后尺寸: {img_array.shape}")
    
    # 保存为JPEG（高质量，作为基准）
    base_jpg_path = "test_output/base_image.jpg"
    Image.fromarray(img_array).save(base_jpg_path, "JPEG", quality=95)
    base_size = os.path.getsize(base_jpg_path)
    print(f"基准JPEG保存: {base_jpg_path} ({base_size/1024:.2f} KB)")
    
    # 初始化压缩器
    compressor = SVDImageCompressor(verbose=True)
    
    # 测试不同的秩，保存为JPEG
    ranks = [10, 30, 50, 100]
    
    print(f"\n{'='*60}")
    print("测试SVD压缩后保存为JPEG")
    print(f"{'='*60}")
    
    for rank in ranks:
        print(f"\n秩 = {rank}")
        print("-" * 40)
        
        # 压缩图像
        compressed_array, info = compressor.compress_image(img_array, rank)
        
        # 保存为JPEG
        output_path = f"test_output/svd_rank{rank}.jpg"
        Image.fromarray(compressed_array).save(output_path, "JPEG", quality=95)
        
        # 计算文件大小
        svd_size = os.path.getsize(output_path)
        size_ratio = svd_size / base_size
        
        # 计算质量指标（与原始RGB比较）
        metrics = compressor.calculate_metrics(img_array, compressed_array)
        
        print(f"理论压缩比: {info['compression_ratio']:.4f}")
        print(f"文件大小比 (vs 基准JPEG): {size_ratio:.4f}")
        print(f"PSNR: {metrics.get('psnr', 0):.2f} dB")
        if 'ssim' in metrics:
            print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"基准JPEG大小: {base_size/1024:.2f} KB")
        print(f"SVD+JPEG大小: {svd_size/1024:.2f} KB")
        print(f"节省空间: {(1 - size_ratio) * 100:.1f}%")
        print(f"处理时间: {info['processing_time']:.2f}秒")
    
    # 测试不同JPEG质量设置
    print(f"\n{'='*60}")
    print("测试不同JPEG质量设置（无SVD）")
    print(f"{'='*60}")
    
    qualities = [90, 80, 70, 60]
    
    for quality in qualities:
        output_path = f"test_output/jpeg_q{quality}.jpg"
        Image.fromarray(img_array).save(output_path, "JPEG", quality=quality)
        jpeg_size = os.path.getsize(output_path)
        size_ratio = jpeg_size / base_size
        
        # 加载JPEG计算质量损失
        jpeg_img = Image.open(output_path)
        jpeg_array = np.array(jpeg_img)
        metrics = compressor.calculate_metrics(img_array, jpeg_array)
        
        print(f"\nJPEG质量={quality}:")
        print(f"  文件大小: {jpeg_size/1024:.2f} KB")
        print(f"  大小比: {size_ratio:.4f}")
        print(f"  节省空间: {(1 - size_ratio) * 100:.1f}%")
        print(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
        if 'ssim' in metrics:
            print(f"  SSIM: {metrics['ssim']:.4f}")
    
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print("对于PNG图像，SVD压缩可能不会减少文件大小，因为：")
    print("1. PNG已经是无损压缩格式")
    print("2. SVD近似引入了难以压缩的噪声")
    print("3. Alpha通道保持不变")
    print("\n对于JPEG格式，SVD压缩可能更有效，但需要与标准JPEG压缩比较。")

if __name__ == "__main__":
    main()