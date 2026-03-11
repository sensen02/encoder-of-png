#!/usr/bin/env python3
"""
基于奇异值分解（SVD）的图像压缩工具
使用低秩近似（类似LoRA原理）压缩图像
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import time
import warnings

# 尝试导入可选依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not installed, SSIM calculation will be disabled")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed, SSIM calculation will be disabled")

class SVDImageCompressor:
    """SVD图像压缩器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def load_image(self, image_path: str) -> np.ndarray:
        """加载图像为numpy数组"""
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            
            if self.verbose:
                print(f"加载图像: {image_path}")
                print(f"  尺寸: {img_array.shape}")
                print(f"  数据类型: {img_array.dtype}")
                print(f"  范围: [{img_array.min()}, {img_array.max()}]")
            
            return img_array
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {e}")
    
    def save_image(self, image_array: np.ndarray, output_path: str, quality: int = 95):
        """保存numpy数组为图像"""
        try:
            # 确保值在合理范围内
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(image_array)
            img.save(output_path, quality=quality)
            
            if self.verbose:
                file_size = os.path.getsize(output_path)
                print(f"保存图像: {output_path} ({file_size / 1024:.2f} KB)")
                
        except Exception as e:
            raise ValueError(f"无法保存图像 {output_path}: {e}")
    
    def svd_compress_channel(self, channel: np.ndarray, rank: int) -> Tuple[np.ndarray, float]:
        """
        对单通道图像进行SVD压缩
        
        Args:
            channel: 单通道图像矩阵 (H, W)
            rank: 保留的奇异值数量
            
        Returns:
            compressed_channel: 压缩后的通道
            compression_ratio: 压缩比
        """
        # 执行SVD分解
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        
        # 保留前rank个奇异值
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vt_k = Vt[:rank, :]
        
        # 重建近似矩阵
        compressed = U_k @ np.diag(S_k) @ Vt_k
        
        # 计算压缩比
        h, w = channel.shape
        original_size = h * w
        compressed_size = rank * (h + w + 1)  # U: h×k, Vt: k×w, S: k
        compression_ratio = compressed_size / original_size
        
        return compressed, compression_ratio
    
    def compress_image(self, image_array: np.ndarray, rank: int) -> Tuple[np.ndarray, Dict]:
        """
        压缩彩色图像
        
        Args:
            image_array: 图像数组 (H, W, C) 或 (H, W)
            rank: 保留的奇异值数量
            
        Returns:
            compressed_image: 压缩后的图像
            info: 压缩信息字典
        """
        start_time = time.time()
        
        # 检查图像维度
        if len(image_array.shape) == 2:
            # 灰度图像
            channels = [image_array]
            is_color = False
        elif len(image_array.shape) == 3:
            # 彩色图像
            h, w, c = image_array.shape
            if c == 3:  # RGB
                channels = [image_array[:, :, i] for i in range(3)]
                is_color = True
            elif c == 4:  # RGBA
                # 分离RGB和Alpha通道
                rgb_channels = [image_array[:, :, i] for i in range(3)]
                alpha_channel = image_array[:, :, 3]
                channels = rgb_channels
                has_alpha = True
                is_color = True
            else:
                raise ValueError(f"不支持的通道数: {c}")
        else:
            raise ValueError(f"不支持的图像维度: {len(image_array.shape)}")
        
        # 压缩每个通道
        compressed_channels = []
        channel_ratios = []
        
        for i, channel in enumerate(channels):
            if self.verbose and is_color:
                print(f"处理通道 {['R', 'G', 'B'][i] if i < 3 else i}...")
            
            compressed_channel, ratio = self.svd_compress_channel(channel, rank)
            compressed_channels.append(compressed_channel)
            channel_ratios.append(ratio)
        
        # 合并通道
        if is_color:
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # RGBA图像
                compressed_rgb = np.stack(compressed_channels, axis=2)
                compressed_image = np.dstack([compressed_rgb, alpha_channel])
            else:
                # RGB或灰度
                compressed_image = np.stack(compressed_channels, axis=2)
        else:
            compressed_image = compressed_channels[0]
        
        # 计算总体压缩比
        avg_compression_ratio = np.mean(channel_ratios)
        
        # 计算质量指标
        metrics = self.calculate_metrics(image_array, compressed_image)
        
        # 收集信息
        info = {
            'original_shape': image_array.shape,
            'compressed_shape': compressed_image.shape,
            'rank': rank,
            'compression_ratio': avg_compression_ratio,
            'channel_ratios': channel_ratios,
            'processing_time': time.time() - start_time,
            'metrics': metrics
        }
        
        if self.verbose:
            print(f"压缩完成!")
            print(f"  奇异值数量: {rank}")
            print(f"  压缩比: {avg_compression_ratio:.4f}")
            print(f"  处理时间: {info['processing_time']:.2f}秒")
            if 'psnr' in metrics:
                print(f"  PSNR: {metrics['psnr']:.2f} dB")
            if 'ssim' in metrics:
                print(f"  SSIM: {metrics['ssim']:.4f}")
        
        return compressed_image, info
    
    def compress_by_ratio(self, image_array: np.ndarray, target_ratio: float) -> Tuple[np.ndarray, Dict]:
        """
        按目标压缩比压缩图像
        
        Args:
            image_array: 图像数组
            target_ratio: 目标压缩比 (0-1之间)
            
        Returns:
            compressed_image: 压缩后的图像
            info: 压缩信息字典
        """
        if len(image_array.shape) == 2:
            h, w = image_array.shape
            max_rank = min(h, w)
        else:
            h, w, c = image_array.shape
            max_rank = min(h, w)
        
        # 根据压缩比计算需要的秩
        # 压缩比 = k * (h + w + 1) / (h * w)
        # 所以 k = 压缩比 * (h * w) / (h + w + 1)
        rank = int(target_ratio * h * w / (h + w + 1))
        
        # 确保秩在合理范围内
        rank = max(1, min(rank, max_rank))
        
        if self.verbose:
            print(f"目标压缩比: {target_ratio:.4f}")
            print(f"计算出的秩: {rank} (最大: {max_rank})")
        
        return self.compress_image(image_array, rank)
    
    def compress_by_quality(self, image_array: np.ndarray, target_quality: float, 
                          max_iterations: int = 20) -> Tuple[np.ndarray, Dict]:
        """
        按目标质量压缩图像（使用二分搜索找到合适的秩）
        
        Args:
            image_array: 图像数组
            target_quality: 目标质量 (0-1之间)
            max_iterations: 最大迭代次数
            
        Returns:
            compressed_image: 压缩后的图像
            info: 压缩信息字典
        """
        if len(image_array.shape) == 2:
            h, w = image_array.shape
            max_rank = min(h, w)
        else:
            h, w, c = image_array.shape
            max_rank = min(h, w)
        
        # 二分搜索找到满足质量要求的最小秩
        low, high = 1, max_rank
        best_image = None
        best_info = None
        
        for iteration in range(max_iterations):
            mid = (low + high) // 2
            
            if self.verbose:
                print(f"迭代 {iteration + 1}: 测试秩 = {mid}")
            
            compressed_image, info = self.compress_image(image_array, mid)
            ssim_value = info['metrics'].get('ssim', 0)
            
            if ssim_value >= target_quality:
                # 质量足够，尝试更小的秩
                high = mid
                best_image = compressed_image
                best_info = info
                
                if self.verbose:
                    print(f"  质量: {ssim_value:.4f} >= {target_quality:.4f}, 尝试更小秩")
            else:
                # 质量不足，需要更大的秩
                low = mid + 1
                
                if self.verbose:
                    print(f"  质量: {ssim_value:.4f} < {target_quality:.4f}, 尝试更大秩")
            
            if low >= high:
                break
        
        if best_image is None:
            # 如果没找到，使用最大秩
            if self.verbose:
                print("未找到满足质量要求的秩，使用最大秩")
            return self.compress_image(image_array, max_rank)
        
        return best_image, best_info
    
    def calculate_metrics(self, original: np.ndarray, compressed: np.ndarray) -> Dict:
        """计算图像质量指标"""
        metrics = {}
        
        # 确保图像类型一致
        if original.dtype != compressed.dtype:
            if original.dtype == np.uint8:
                compressed = np.clip(compressed, 0, 255).astype(np.uint8)
            else:
                original = original.astype(np.float32)
                compressed = compressed.astype(np.float32)
        
        # 计算PSNR
        if original.dtype == np.uint8:
            mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            metrics['psnr'] = psnr
        else:
            # 对于浮点图像
            max_val = max(original.max(), compressed.max())
            mse = np.mean((original - compressed) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(max_val / np.sqrt(mse))
            metrics['psnr'] = psnr
        
        # 计算SSIM（如果可用）
        if SKIMAGE_AVAILABLE and len(original.shape) == 2:
            # 灰度图像
            data_range = 255 if original.dtype == np.uint8 else original.max() - original.min()
            ssim_value = ssim(original, compressed, data_range=data_range)
            metrics['ssim'] = ssim_value
        elif CV2_AVAILABLE and len(original.shape) == 3:
            # 彩色图像，使用OpenCV
            if original.shape[2] == 3:
                original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
                compressed_bgr = cv2.cvtColor(compressed, cv2.COLOR_RGB2BGR)
                
                # 计算每个通道的SSIM然后平均
                ssim_values = []
                for i in range(3):
                    ssim_channel = ssim(original_bgr[:, :, i], compressed_bgr[:, :, i])
                    ssim_values.append(ssim_channel)
                metrics['ssim'] = np.mean(ssim_values)
        
        # 计算均方误差
        metrics['mse'] = float(np.mean((original - compressed) ** 2))
        
        # 计算平均绝对误差
        metrics['mae'] = float(np.mean(np.abs(original - compressed)))
        
        return metrics
    
    def visualize_comparison(self, original: np.ndarray, compressed: np.ndarray, 
                           output_path: Optional[str] = None):
        """可视化对比原始图像和压缩图像"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            if len(original.shape) == 2:
                axes[0].imshow(original, cmap='gray')
            else:
                axes[0].imshow(original)
            axes[0].set_title('原始图像')
            axes[0].axis('off')
            
            # 压缩图像
            if len(compressed.shape) == 2:
                axes[1].imshow(compressed, cmap='gray')
            else:
                axes[1].imshow(compressed)
            axes[1].set_title('压缩图像')
            axes[1].axis('off')
            
            # 差异图像
            difference = np.abs(original.astype(np.float32) - compressed.astype(np.float32))
            if len(difference.shape) == 3:
                difference = np.mean(difference, axis=2)
            
            im = axes[2].imshow(difference, cmap='hot')
            axes[2].set_title('差异图')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"对比图已保存: {output_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            warnings.warn("matplotlib not installed, visualization disabled")
        except Exception as e:
            warnings.warn(f"可视化失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="基于SVD的图像压缩工具")
    
    # 输入输出
    parser.add_argument("--input", type=str, help="输入图像文件路径")
    parser.add_argument("--output", type=str, help="输出图像文件路径")
    parser.add_argument("--input_dir", type=str, help="输入目录（批量处理）")
    parser.add_argument("--output_dir", type=str, help="输出目录（批量处理）")
    
    # 压缩参数
    parser.add_argument("--rank", type=int, help="保留的奇异值数量")
    parser.add_argument("--compression_ratio", type=float, 
                       help="目标压缩比（0-1之间）")
    parser.add_argument("--quality", type=float,
                       help="目标质量（0-1之间），基于SSIM")
    
    # 其他选项
    parser.add_argument("--metrics", action="store_true", 
                       help="计算并显示质量指标")
    parser.add_argument("--visualize", action="store_true",
                       help="生成对比可视化图")
    parser.add_argument("--format", type=str, default="jpg",
                       help="输出格式（jpg, png等）")
    parser.add_argument("--quiet", action="store_true",
                       help="减少输出信息")
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.input and not args.input_dir:
        parser.error("必须提供 --input 或 --input_dir 参数")
    
    if args.input and args.input_dir:
        parser.error("不能同时提供 --input 和 --input_dir 参数")
    
    if args.output and args.output_dir:
        parser.error("不能同时提供 --output 和 --output_dir 参数")
    
    # 初始化压缩器
    compressor = SVDImageCompressor(verbose=not args.quiet)
    
    # 处理单个图像
    if args.input:
        try:
            # 加载图像
            original_image = compressor.load_image(args.input)
            
            # 确定压缩方法
            if args.rank:
                # 固定秩压缩
                compressed_image, info = compressor.compress_image(original_image, args.rank)
            elif args.compression_ratio:
                # 按压缩比压缩
                compressed_image, info = compressor.compress_by_ratio(original_image, args.compression_ratio)
            elif args.quality:
                # 按质量压缩
                compressed_image, info = compressor.compress_by_quality(original_image, args.quality)
            else:
                # 默认使用50个奇异值
                compressed_image, info = compressor.compress_image(original_image, 50)
            
            # 保存压缩图像
            output_path = args.output or f"compressed_{Path(args.input).stem}.{args.format}"
            compressor.save_image(compressed_image, output_path)
            
            # 显示指标
            if args.metrics:
                print("\n质量指标:")
                for metric, value in info['metrics'].items():
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 生成可视化
            if args.visualize:
                viz_path = f"comparison_{Path(args.input).stem}.png"
                compressor.visualize_comparison(original_image, compressed_image, viz_path)
            
            print(f"\n压缩完成!")
            print(f"原始文件: {args.input}")
            print(f"压缩文件: {output_path}")
            print(f"压缩比: {info['compression_ratio']:.4f}")
            print(f"处理时间: {info['processing_time']:.2f}秒")
            
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 批量处理目录
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir or f"{input_dir}_compressed")
        output_dir.mkdir(exist_ok=True)
        
        # 获取图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            print(f"在 {input_dir} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件，开始批量处理...")
        
        for i, image_file in enumerate(image_files):
            try:
                print(f"\n处理 {i+1}/{len(image_files)}: {image_file.name}")
                
                # 加载图像
                original_image = compressor.load_image(str(image_file))
                
                # 使用固定秩或默认值
                rank = args.rank or 50
                compressed_image, info = compressor.compress_image(original_image, rank)
                
                # 保存压缩图像
                output_file = output_dir / f"{image_file.stem}_compressed.{args.format}"
                compressor.save_image(compressed_image, str(output_file))
                
                print(f"  压缩比: {info['compression_ratio']:.4f}, PSNR: {info['metrics'].get('psnr', 0):.2f} dB")
                
            except Exception as e:
                print(f"  处理失败: {e}")
                continue
        
        print(f"\n批量处理完成! 结果保存在: {output_dir}")

if __name__ == "__main__":
    main()