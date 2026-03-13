import time
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities_1212 import *  # 引入刚才修改的工具文件
from datetime import datetime
import itertools
import os

starttime = time.time()

# -----------------------------
#        模型与函数定义
# -----------------------------

# === 新增：正弦位置编码 (标准 Diffusion 做法) ===
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
# ==============================================

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        # === 核心修改 1: 输入通道数增加 ===
        # 我们不再把标签当成风格嵌入中间层，而是直接拼接到输入层
        # 输入通道 = 图片通道(1) + 标签通道(4) = 5
        self.init_conv = ResidualConvBlock(in_channels + n_cfeat, n_feat, is_res=True)
        # ==============================

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(n_feat),
            nn.Linear(n_feat, 2 * n_feat),
            nn.GELU()
        )
        
        self.time_proj1 = nn.Linear(2 * n_feat, 2 * n_feat)
        self.time_proj2 = nn.Linear(2 * n_feat, n_feat)

        # === 核心修改 2: 移除 EmbedFC ===
        # 既然标签已经在输入层作为“地图”给进去了，中间就不需要再加了
        # self.contextembed1 = ... (移除)
        # self.contextembed2 = ... (移除)
        # ==============================

        self.up0 = ResidualConvBlock(2 * n_feat, 2 * n_feat, is_res=True)
        self.up1 = UnetUp(2 * n_feat, n_feat, skip_channels=2 * n_feat)
        self.up2 = UnetUp(n_feat, n_feat, skip_channels=n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        # x: [Batch, 1, 128, 128]
        # c: [Batch, 4]
        
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # === 核心修改 3: 空间拼接 (Spatial Concatenation) ===
        # 1. 将标签 c 从 [Batch, 4] 扩展为 [Batch, 4, 1, 1]
        c_expanded = c.view(-1, self.n_cfeat, 1, 1)
        
        # 2. 将标签广播到全图大小 [Batch, 4, 128, 128]
        # 这样，如果 c[0]=1，那么第一张特征图全都是 1
        c_map = c_expanded.expand(-1, -1, x.shape[2], x.shape[3])
        
        # 3. 拼接到输入图像上 -> [Batch, 1+4, 128, 128]
        x_in = torch.cat([x, c_map], dim=1)
        # ================================================

        # 1. 初始卷积 (现在处理 5 个通道)
        x_init = self.init_conv(x_in)
        
        # 2. 下采样
        down1_skip, down1 = self.down1(x_init) 
        down2_skip, down2 = self.down2(down1)
        
        hiddenvec = down2

        # 时间嵌入
        t = t.view(-1) 
        t_emb = self.time_mlp(t * 1000) 
        temb1 = self.time_proj1(t_emb).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.time_proj2(t_emb).view(-1, self.n_feat, 1, 1)

        # 5. 上采样
        up0 = self.up0(hiddenvec)
        
        # === 核心修改 4: 移除 cemb 加法 ===
        # 只加时间嵌入，不加 context 嵌入
        up2 = self.up1(up0 + temb1, down2_skip)     
        up3 = self.up2(up2 + temb2, down1_skip)     
        
        out = self.out(torch.cat((up3, x_init), 1))
        return out


def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    mean = torch.clamp(mean, -1, 1)
    return mean + noise

@torch.no_grad()
def sample_ddpm_context(model, n_sample, context, save_rate=20, guide_w=0.0): # <--- 新增 guide_w 参数
    # 注意：这里使用全局变量 d_RSSI, RSSI_height, RSSI_width
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)
    intermediate = []
    
    # 准备无条件生成的 context (全部设为 -1)
    if guide_w > 0:
        context_uncond = torch.ones_like(context).to(device) * -1

    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        
        # === 修复开始：调整 t 的形状以匹配 batch size ===
        # 原代码: t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        # 修改为: 先构造标量，再扩展到 [n_sample, 1, 1, 1]
        t = torch.tensor([i / timesteps]).to(device)
        t = t.view(1, 1, 1, 1).repeat(n_sample, 1, 1, 1)
        # === 修复结束 ===

        z = torch.randn_like(samples) if i > 1 else 0
        
        # === 修改：支持 CFG 引导 ===
        if guide_w > 0:
            # 复制输入，一份给有条件，一份给无条件
            samples_double = samples.repeat(2, 1, 1, 1)
            t_double = t.repeat(2, 1, 1, 1)
            context_double = torch.cat([context, context_uncond], dim=0)
            
            # 一次预测两个
            eps_double = model(samples_double, t_double, c=context_double)
            eps_cond = eps_double[:n_sample]
            eps_uncond = eps_double[n_sample:]
            
            # 引导公式: uncond + w * (cond - uncond)
            # w 越大，生成的图像越符合标签，但多样性会降低
            eps = eps_uncond + (1 + guide_w) * (eps_cond - eps_uncond)
        else:
            eps = model(samples, t, c=context) 
        # ==========================
        
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

def save_samples_as_npy(samples, save_path):
    """
    保存为 .npy, 形状 [batch, H, W, C]
    """
    samples_np = samples.detach().cpu().numpy()
    # [batch, C, H, W] -> [batch, H, W, C]
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))
    np.save(save_path, samples_np)
    print(f"Saved samples to {save_path} with shape {samples_np.shape}")

if __name__ == "__main__":
    # =====================
    #     0. GPU 配置与检查
    # =====================
    #在此处指定要使用的 GPU ID (例如 "0", "1", "2")
    TARGET_GPU_ID = "0" 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID
    
    if torch.cuda.is_available():
        # 注意：设置了 CUDA_VISIBLE_DEVICES 后，PyTorch 内部的 device id 总是从 0 开始
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"\n{'='*40}")
        print(f"   Running on GPU: {props.name}")
        print(f"   Total Memory:   {props.total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Device ID: {TARGET_GPU_ID} (Mapped to cuda:0)")
        print(f"{'='*40}\n")
    else:
        device = torch.device("cpu")
        print("\nWARNING: CUDA is not available. Running on CPU!\n")

    # =====================
    #     1. 加载数据
    # =====================
    # 请修改为您的图片文件夹路径
    img_dir = r"/home/yang/gprmax/gengpr/cutmore16gpr50grey"  
    
    # 图像参数设置
    RSSI_height = 128
    RSSI_width = 128
    d_RSSI = 1  # 灰度图是 1 通道

    # Transform: Resize -> Grayscale -> Tensor(0~1) -> Normalize(-1~1)
    transform = transforms.Compose([
        transforms.Resize((RSSI_height, RSSI_width)),
        transforms.Grayscale(num_output_channels=1),  # Add this line to force grayscale
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])

    # 使用新的 Dataset
    # 注意：请确保 diffusion_utilities_1212.py 中已经更新了 BScanImageDataset 类
    try:
        dataset = BScanImageDataset(img_dir, transform=transform)
    except NameError:
        print("Error: 请先更新 diffusion_utilities_1212.py 添加 BScanImageDataset 类")
        exit()

    if len(dataset) == 0:
        print(f"Error: No images found in {img_dir}")
        print("Please check the directory path and ensure it contains .png files.")
        exit()

    # 获取所有标签用于后续筛选
    labels = dataset.labels 
    print("Data loaded. Total images:", len(dataset))
    print("Labels shape:", labels.shape)
    print("Image size:", (d_RSSI, RSSI_height, RSSI_width))

    # =====================
    #     2. 设置超参数
    # =====================
    timesteps = 1000
    beta1 = 1e-4
    beta2 = 0.02

    # device 已经在上面设置好了，这里不需要重复设置
    # device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    
    n_feat = 128 
    n_cfeat = labels.shape[1] # 4
    batch_size = 64 # 图片变大了，Batch Size 可能需要调小一点，防止显存溢出
    n_epoch = 500
    lrate = 1e-4

    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    def perturb_input(x, t, noise):
        # 修复：添加 .sqrt()。标准公式是 sqrt(bar_alpha) * x + sqrt(1 - bar_alpha) * noise
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise

    # === 修改开始：给输出文件夹加上时间戳 ===
    # 获取当前时间，例如 20251212_153000
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 权重保存文件夹：weights_16_experiments_img_20251212_153000/
    multi_exp_root = f'./weights_16_experiments_img_{current_time}/'
    os.makedirs(multi_exp_root, exist_ok=True)
    
    # 结果可视化文件夹：result_vis_img_20251212_153000/
    save_vis_dir = f'./result_vis_img_{current_time}/'
    os.makedirs(save_vis_dir, exist_ok=True)
    
    print(f"Experiment results will be saved to:\n  Weights: {multi_exp_root}\n  Vis:     {save_vis_dir}")
    # === 修改结束 ===

    all_16_combos = list(itertools.product([0, 1], repeat=4))

    # === 修改开始 ===
    # 如果只想测试排除 (1, 0, 1, 0) 这一种情况：
    target_exclude = (1, 0, 1, 0) 
    # 过滤列表，只包含这一个元素
    loop_combos = [c for c in all_16_combos if c == target_exclude]
    
    # 或者，如果您想跑完所有 16 种，就保持原样：
    # loop_combos = all_16_combos 
    
    for excluded_combo in loop_combos:
    # === 修改结束 ===
        exp_dir = os.path.join(multi_exp_root, f"excluded_{'_'.join(map(str, excluded_combo))}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n=== Training (multi-run) excluding {excluded_combo} ===")

        train_combos_15 = [combo for combo in all_16_combos if combo != excluded_combo]
        
        # 筛选数据索引
        train_indices_15 = []
        for i_lbl, lbl_arr in enumerate(labels):
            lbl_tuple = tuple(lbl_arr.tolist())
            if lbl_tuple in train_combos_15:
                train_indices_15.append(i_lbl)

        if len(train_indices_15) == 0:
            print("Warning: No training data found for this combination set. Skipping.")
            continue

        train_dataset_15 = Subset(dataset, train_indices_15)
        dataloader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True, num_workers=4)

        multi_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
        multi_optim = torch.optim.Adam(multi_model.parameters(), lr=lrate)

        multi_model.train()
        for ep in range(n_epoch):
            print(f'   Multi-run epoch {ep} (exclude {excluded_combo})')
            multi_optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
            
            for x, c in tqdm(dataloader_15, mininterval=2):
                multi_optim.zero_grad()
                x = x.to(device)
                c = c.to(device).float()

                # === 修改：使用 -1 作为掩码，避免与 0000 冲突 ===
                # 即使您不想要 "0.5标签"，为了让 0000 清晰，
                # 训练时必须偶尔让模型学会“不看标签生成”，
                # 这样在测试时加强标签权重 (guide_w) 才能生效。
                
                # 10% 的概率将标签设为 -1 (代表未知)
                if np.random.random() < 0.1:
                    c = torch.ones_like(c) * -1
                # ==============================================

                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
                x_pert = perturb_input(x, t, noise)

                pred_noise = multi_model(x_pert, t / timesteps, c=c)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                multi_optim.step()

            if ep % 10 == 0 or ep == n_epoch - 1:
                model_path = os.path.join(exp_dir, f"context_model_{ep}.pth")
                torch.save(multi_model.state_dict(), model_path)

        # 测试
        multi_model.eval()
        final_model_path = os.path.join(exp_dir, f"context_model_{n_epoch - 1}.pth")
        multi_model.load_state_dict(torch.load(final_model_path, map_location=device))

        print(f"   Loaded final model for testing (excluded={excluded_combo})")

        for test_combo in all_16_combos:
            context_batch = torch.tensor([test_combo] * 10).float().to(device)
            
            # === 修改：加入 guide_w=2.0 (通常 2.0 效果很好) ===
            # 这会强制模型严格遵守标签，0000 就会变干净，不再是叠加图
            test_samples, _ = sample_ddpm_context(multi_model, context_batch.shape[0], context_batch, guide_w=2.0)
            
            trained_dir = "trained" if test_combo in train_combos_15 else "untrained"
            combo_dir = os.path.join(exp_dir, trained_dir, f"context_{'_'.join(map(str, test_combo))}")
            os.makedirs(combo_dir, exist_ok=True)

            # 保存 npy
            npy_path = os.path.join(combo_dir, "samples.npy")
            test_samples_t = test_samples.clone().detach() # 避免重复 wrap tensor
            save_samples_as_npy(test_samples_t, npy_path)

            # 可视化 (修改部分)
            # === 修改：不再取平均，而是拼成网格图 ===
            # make_grid 会自动处理反归一化: value_range=(-1, 1) -> [0, 1]
            # nrow=5 表示每行放 5 张图 (共 10 张，即 2 行)
            grid = make_grid(test_samples_t, nrow=5, normalize=True, value_range=(-1, 1))
            
            # 维度转换: [C, H, W] -> [H, W, C] 用于 matplotlib 显示
            img_show = grid.permute(1, 2, 0).cpu().numpy()
            
            # 画布设置大一点，适应网格
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(img_show) 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            
            # === 建议修改：文件名加上 excluded 信息，防止覆盖 ===
            vis_fname = f"vis_excl_{'_'.join(map(str, excluded_combo))}_test_{'_'.join(map(str, test_combo))}.png"
            vis_path = os.path.join(save_vis_dir, vis_fname)
            fig.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"[Saved visualization to {vis_path}]")

        print(f"=== Finished multi-run experiment excluding {excluded_combo} ===")

    print("\nAll multi-run experiments done!")
    endtime = time.time()
    print(f"Total time: {endtime - starttime} seconds")