#!/usr/bin/env python3
"""
SVD图像压缩示例用法
演示如何使用命令行工具和Python API
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加当前目录到路径以便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demonstrate_command_line():
    """演示命令行用法"""
    print("命令行用法示例")
    print("="*50)
    
    examples = [
        {
            "desc": "基本压缩（使用50个奇异值）",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_basic.jpg --rank 50"
        },
        {
            "desc": "按压缩比压缩（目标压缩比0.2）",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_ratio.jpg --compression_ratio 0.2"
        },
        {
            "desc": "按质量压缩（目标SSIM 0.9）",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_quality.jpg --quality 0.9"
        },
        {
            "desc": "计算质量指标",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_metrics.jpg --rank 30 --metrics"
        },
        {
            "desc": "生成对比可视化图",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_viz.jpg --rank 40 --visualize"
        },
        {
            "desc": "批量处理目录",
            "cmd": "python svd_compress.py --input_dir ./images --output_dir ./compressed --rank 50"
        },
        {
            "desc": "安静模式（减少输出）",
            "cmd": "python svd_compress.py --input test_image.png --output compressed_quiet.jpg --rank 60 --quiet"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['desc']}:")
        print(f"   {example['cmd']}")
    
    print("\n要运行这些示例，请先创建测试图像:")
    print("   python test_svd.py  # 这会创建test_image.png")

def demonstrate_python_api():
    """演示Python API用法"""
    print("\n\nPython API用法示例")
    print("="*50)
    
    code_examples = '''
# 1. 导入压缩器
from svd_compress import SVDImageCompressor

# 2. 创建压缩器实例
compressor = SVDImageCompressor(verbose=True)

# 3. 加载图像
image_path = "test_image.png"
original_image = compressor.load_image(image_path)

# 4. 方法1: 固定秩压缩
rank = 50
compressed_image, info = compressor.compress_image(original_image, rank)
compressor.save_image(compressed_image, "compressed_rank50.jpg")
print(f"压缩比: {info['compression_ratio']:.4f}")
print(f"PSNR: {info['metrics']['psnr']:.2f} dB")

# 5. 方法2: 按压缩比压缩
target_ratio = 0.3
compressed_image, info = compressor.compress_by_ratio(original_image, target_ratio)
compressor.save_image(compressed_image, "compressed_ratio0.3.jpg")
print(f"使用的秩: {info['rank']}")

# 6. 方法3: 按质量压缩
target_quality = 0.95
compressed_image, info = compressor.compress_by_quality(original_image, target_quality)
compressor.save_image(compressed_image, "compressed_quality0.95.jpg")
print(f"实际SSIM: {info['metrics'].get('ssim', 0):.4f}")

# 7. 计算质量指标
metrics = compressor.calculate_metrics(original_image, compressed_image)
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# 8. 生成对比可视化图
compressor.visualize_comparison(original_image, compressed_image, "comparison.png")
'''
    
    print(code_examples)

def demonstrate_advanced_usage():
    """演示高级用法"""
    print("\n\n高级用法示例")
    print("="*50)
    
    advanced_code = '''
# 1. 自定义质量评估
from svd_compress import SVDImageCompressor
import numpy as np

compressor = SVDImageCompressor(verbose=False)

# 加载图像
original_image = compressor.load_image("test_image.png")

# 2. 探索不同秩的效果
ranks = [10, 20, 30, 40, 50, 75, 100]
results = []

for rank in ranks:
    compressed, info = compressor.compress_image(original_image, rank)
    results.append({
        'rank': rank,
        'compression_ratio': info['compression_ratio'],
        'psnr': info['metrics']['psnr'],
        'ssim': info['metrics'].get('ssim', 0)
    })

# 3. 找到最佳压缩点（平衡压缩比和质量）
for r in results:
    score = r['psnr'] / r['compression_ratio']  # 简单的评分公式
    print(f"秩={r['rank']:3d}, 压缩比={r['compression_ratio']:.4f}, "
          f"PSNR={r['psnr']:.2f}dB, 评分={score:.2f}")

# 4. 处理大图像（分块压缩）
def compress_large_image(image_array, block_size=256, rank=50):
    """分块压缩大图像"""
    h, w = image_array.shape[:2]
    compressed_blocks = []
    
    for y in range(0, h, block_size):
        row_blocks = []
        for x in range(0, w, block_size):
            # 提取块
            block = image_array[y:y+block_size, x:x+block_size]
            # 压缩块
            compressed_block, _ = compressor.compress_image(block, rank)
            row_blocks.append(compressed_block)
        
        # 水平拼接块
        compressed_row = np.hstack(row_blocks)
        compressed_blocks.append(compressed_row)
    
    # 垂直拼接所有行
    return np.vstack(compressed_blocks)

# 5. 渐进式重建演示
def progressive_reconstruction(image_array, ranks=[10, 30, 50, 100]):
    """渐进式重建：逐步增加细节"""
    reconstructions = []
    
    for rank in ranks:
        compressed, _ = compressor.compress_image(image_array, rank)
        reconstructions.append(compressed)
    
    return reconstructions
'''
    
    print(advanced_code)

def create_sample_images():
    """创建示例图像（如果不存在）"""
    print("\n\n创建示例图像")
    print("="*50)
    
    # 检查测试图像是否存在
    if not os.path.exists("test_image.png"):
        print("创建测试图像...")
        from svd_compress import SVDImageCompressor
        import numpy as np
        from PIL import Image
        
        # 创建简单的测试图像
        width, height = 400, 300
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 渐变背景
        for y in range(height):
            for x in range(width):
                r = int(128 + 127 * np.sin(x / 50))
                g = int(128 + 127 * np.cos(y / 50))
                b = int(128 + 127 * np.sin((x + y) / 100))
                img_array[y, x] = [r, g, b]
        
        # 添加几何形状
        # 红色圆形
        center_x, center_y = 100, 100
        radius = 40
        for y in range(height):
            for x in range(width):
                if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                    img_array[y, x] = [255, 0, 0]
        
        # 绿色矩形
        img_array[150:250, 200:300, 1] = 255
        
        # 蓝色三角形
        for y in range(50, 150):
            for x in range(250, 350):
                if x - 250 < y - 50:
                    img_array[y, x, 2] = 255
        
        # 保存图像
        Image.fromarray(img_array).save("test_image.png")
        print("测试图像已创建: test_image.png")
    else:
        print("测试图像已存在: test_image.png")
    
    # 创建示例目录结构
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    print(f"\n示例目录已创建: {sample_dir}")
    print("可以放置图像文件在此目录中进行批量处理")

def main():
    """主函数"""
    print("SVD图像压缩示例用法")
    print("="*60)
    
    # 创建示例图像
    create_sample_images()
    
    # 演示各种用法
    demonstrate_command_line()
    demonstrate_python_api()
    demonstrate_advanced_usage()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print("\n主要功能:")
    print("1. 固定秩压缩 (--rank)")
    print("2. 按压缩比压缩 (--compression_ratio)")
    print("3. 按质量压缩 (--quality)")
    print("4. 质量评估 (--metrics)")
    print("5. 可视化对比 (--visualize)")
    print("6. 批量处理 (--input_dir, --output_dir)")
    
    print("\n下一步:")
    print("1. 运行测试: python test_svd.py")
    print("2. 尝试命令行: python svd_compress.py --input test_image.png --output test.jpg --rank 30")
    print("3. 查看帮助: python svd_compress.py --help")
    print("4. 探索Python API: python example_usage.py")

if __name__ == "__main__":
    main()