#!/usr/bin/env python3
"""检查图像信息"""
from PIL import Image
import os

image_path = "test_images/7504666a-468a-43fb-ab9e-5a3271aa731e-wm.png"

# 打开图像
img = Image.open(image_path)

print(f"图像信息:")
print(f"  文件名: {os.path.basename(image_path)}")
print(f"  尺寸: {img.size} (宽×高)")
print(f"  模式: {img.mode}")
print(f"  格式: {img.format}")
print(f"  文件大小: {os.path.getsize(image_path) / 1024:.2f} KB")

# 如果是RGBA，检查是否有透明通道
if img.mode == 'RGBA':
    print(f"  通道数: 4 (RGBA)")
    # 检查alpha通道
    img_array = img.convert("RGBA")
    alpha = img_array.getchannel('A')
    print(f"  Alpha通道范围: [{min(alpha.getdata())}, {max(alpha.getdata())}]")
elif img.mode == 'RGB':
    print(f"  通道数: 3 (RGB)")
else:
    print(f"  通道数: {len(img.mode)} ({img.mode})")