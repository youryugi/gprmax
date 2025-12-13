import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

# ================= 配置区域 =================
# 1. 图片文件夹路径
IMG_DIR = r"/home/yang/gprmax/gengpr/cut16bluered50"

# 2. 想要检查的图片文件名 (请确保该文件存在于 IMG_DIR 中)
TARGET_IMAGE_NAME = "bscan_0101_v0.png"  # 修改为您想看的文件名

# 3. 目标尺寸 (必须与训练代码一致)
RSSI_height = 128
RSSI_width = 128
# ===========================================

def denormalize_image(tensor):
    """
    反归一化: 从 [-1, 1] 映射回 [0, 1] 用于显示
    """
    return (tensor + 1) / 2

def main():
    img_path = os.path.join(IMG_DIR, TARGET_IMAGE_NAME)
    
    if not os.path.exists(img_path):
        print(f"错误: 找不到文件 {img_path}")
        return

    # 1. 读取原图
    # 注意：训练代码中使用了 .convert("L") 转为灰度
    original_img = Image.open(img_path).convert("L")
    print(f"原图尺寸: {original_img.size}")

    # 2. 定义与训练代码一致的 Transform
    transform = transforms.Compose([
        transforms.Resize((RSSI_height, RSSI_width)), # 强制缩放
        transforms.ToTensor(),                        # [0, 255] -> [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,))          # [0.0, 1.0] -> [-1.0, 1.0]
    ])

    # 3. 应用变换
    transformed_tensor = transform(original_img)
    print(f"变换后 Tensor 形状: {transformed_tensor.shape}")
    print(f"变换后数值范围: min={transformed_tensor.min():.2f}, max={transformed_tensor.max():.2f}")

    # 4. 准备显示
    # 反归一化回 [0, 1] 以便 matplotlib 显示
    display_tensor = denormalize_image(transformed_tensor)
    
    # [C, H, W] -> [H, W, C] -> 去掉单通道维度变成 [H, W]
    display_img = display_tensor.squeeze().numpy()

    # 5. 绘图对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 左图：原图
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f"Original ({original_img.size[0]}x{original_img.size[1]})")
    axes[0].axis('off')

    # 右图：训练时的输入
    axes[1].imshow(display_img, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f"Network Input ({RSSI_width}x{RSSI_height})")
    axes[1].axis('off')

    plt.tight_layout()
    
    # 保存对比图
    save_path = "check_resize_result.png"
    plt.savefig(save_path)
    print(f"\n对比图已保存至: {save_path}")
    print("请打开该图片查看缩放效果。")

if __name__ == "__main__":
    main()