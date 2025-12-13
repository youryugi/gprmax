import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from diffusion_utilities_3d_1212 import *  # 必须导入工具库
from SD_1212_gpu1_3d import ContextUnet, denoise_add_noise  # 导入模型定义

# ================= 配置区域 =================
# 1. 模型权重路径 (请修改为您实际训练好的模型路径)
# 例如: 'weights_16_experiments_img/excluded_1_0_1_0/context_model_99.pth'
MODEL_PATH = '/home/yang/gprmax/gengpr/stabled1213/weights_16_experiments_img_20251213_110538/excluded_1_0_1_0/context_model_99.pth'

# 2. 想要生成的标签列表
# 可以写多个，例如 [[1, 0, 1, 0], [0, 0, 0, 0]]
TARGET_LABELS = [
    [1, 0, 1, 0],  # 生成 1010
    [1, 0, 1, 1]   # 生成 1111
]

# 3. 每个标签生成多少张图
N_SAMPLES = 1

# 4. 输出文件夹
OUTPUT_DIR = './generated_results'

# 5. GPU 设置
DEVICE_ID = "0"
# ===========================================

# -------------------------------------------
#  以下代码通常不需要修改
# -------------------------------------------

# 扩散模型参数 (必须与训练时一致)
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02
d_RSSI = 3
RSSI_height = 128
RSSI_width = 128
n_feat = 64
n_cfeat = 4 # 标签长度

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 预计算扩散参数
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

@torch.no_grad()
def sample_ddpm_context(model, n_sample, context):
    """
    采样函数
    """
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)
    
    for i in range(timesteps, 0, -1):
        print(f'Sampling timestep {i:3d}', end='\r')
        
        # 1. 构造模型需要的 float 时间嵌入 (归一化到 0~1)
        t_float = torch.tensor([i / timesteps]).to(device).view(1, 1, 1, 1)
        
        z = torch.randn_like(samples) if i > 1 else 0
        
        # 2. 预测噪声 (传入 float 时间)
        eps = model(samples, t_float, c=context)
        
        # 3. 计算去噪参数 (使用整数索引 i 获取对应的 alpha/beta)
        # b_t, a_t, ab_t 是预定义的参数表
        beta_t = b_t[i]
        alpha_t = a_t[i]
        alpha_bar_t = ab_t[i]
        
        # 4. 还原一步 (DDPM 公式)
        # mean = 1/sqrt(alpha) * (x - (1-alpha)/sqrt(1-alpha_bar) * eps)
        mean = (samples - eps * ((1 - alpha_t) / (1 - alpha_bar_t).sqrt())) / alpha_t.sqrt()
        
        # 加上噪声项 sigma * z
        noise = beta_t.sqrt() * z
        
        mean = torch.clamp(mean, -1, 1)
        samples = mean + noise

    return samples

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading model from {MODEL_PATH}...")

    # 初始化模型
    model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        print("请检查 MODEL_PATH 是否正确，或者模型结构是否与训练时一致。")
        return
        
    model.eval()
    print("Model loaded successfully.")

    for label in TARGET_LABELS:
        label_str = ''.join(map(str, label))
        print(f"\nGenerating {N_SAMPLES} images for label: {label_str} ...")
        
        # 构造 Context
        # shape: [N_SAMPLES, 4]
        context = torch.tensor([label] * N_SAMPLES).float().to(device)
        
        # 生成
        samples = sample_ddpm_context(model, N_SAMPLES, context)
        
        # === 保存结果 ===
        
        # 1. 拼图 (Grid)
        grid = make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        save_path = os.path.join(OUTPUT_DIR, f"gen_{label_str}.png")
        
        plt.figure(figsize=(10, 5))
        plt.imshow(grid_np, cmap='gray')
        plt.axis('off')
        plt.title(f"Generated: {label_str}")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Saved image to: {save_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()