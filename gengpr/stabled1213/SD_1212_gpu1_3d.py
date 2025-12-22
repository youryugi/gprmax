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
from diffusion_utilities_3d_1212 import *  # 引入刚才修改的工具文件
from datetime import datetime
import itertools
import os

starttime = time.time()

# -----------------------------
#        模型与函数定义
# -----------------------------

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  
        super(ContextUnet, self).__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h // 4)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        
        # === 修改开始 ===
        # up1: 输入128(2*n), 输出64(n). Skip来自down2(128通道)
        self.up1 = UnetUp(2 * n_feat, n_feat, skip_channels=2 * n_feat)
        
        # up2: 输入64(n), 输出64(n). Skip来自down1(64通道)
        # 注意：这里输入改为 n_feat，因为 up1 的输出是 n_feat
        self.up2 = UnetUp(n_feat, n_feat, skip_channels=n_feat)
        # === 修改结束 ===

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        # 1. 初始卷积
        x_init = self.init_conv(x)
        
        # 2. 下采样 (注意这里需要解包)
        # UnetDown 返回 (skip_connection, pooled_output)
        down1_skip, down1 = self.down1(x_init) 
        down2_skip, down2 = self.down2(down1)
        
        # 3. 瓶颈层
        hiddenvec = self.to_vec(down2)

        # 4. 嵌入向量处理
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # 5. 上采样
        up1 = self.up0(hiddenvec)
        
        # 注意：这里使用解包出来的 skip connection (down2_skip, down1_skip)
        up2 = self.up1(cemb1 * up1 + temb1, down2_skip)
        up3 = self.up2(cemb2 * up2 + temb2, down1_skip)
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
def sample_ddpm_context(model, n_sample, context, save_rate=20):
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t_batch = torch.full((n_sample,), i / timesteps, device=device).unsqueeze(-1)
        z = torch.randn_like(samples) if i > 1 else 0

        # 只用条件分支
        eps = model(samples, t_batch, c=context)

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
    img_dir = r"/home/yang/gprmax/gengpr/cut16bluered50"  
    
    # 图像参数设置
    RSSI_height = 128
    RSSI_width = 128
    d_RSSI = 3  # 从灰度(1通道)改为RGB(3通道)

    # Transform: Resize -> Tensor(0~1) -> Normalize(-1~1)
    transform = transforms.Compose([
        transforms.Resize((RSSI_height, RSSI_width)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3通道归一化
    ])

    # 使用新的 Dataset
    # 注意：请确保 diffusion_utilities_1212.py 中已经更新了 BScanImageDataset 类
    try:
        dataset = BScanImageDataset(img_dir, transform=transform)
    except NameError:
        print("Error: 请先更新 diffusion_utilities_1212.py 添加 BScanImageDataset 类")
        exit()

    # 获取所有标签用于后续筛选
    labels = dataset.labels 
    print("Data loaded. Total images:", len(dataset))
    print("Labels shape:", labels.shape)
    print("Image size:", (d_RSSI, RSSI_height, RSSI_width))

    # =====================
    #     2. 设置超参数
    # =====================
    timesteps = 1000    # [修改] 从 400 改为 1000，提升生成细节质量
    beta1 = 1e-4
    beta2 = 0.02

    # device 已经在上面设置好了，这里不需要重复设置
    # device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    
    n_feat = 64 
    n_cfeat = labels.shape[1] # 4
    batch_size = 64 
    n_epoch = 300      # [修改] 增加训练轮数，建议 1000~2000 起步
    lrate = 1e-4

    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

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
            
            # 学习率线性衰减 (保持不变即可，适配新的 n_epoch)
            multi_optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
            
            # [修改] 使用 enumerate 获取 batch 索引，以便打印验证信息
            for batch_idx, (x, c) in enumerate(tqdm(dataloader_15, mininterval=2)):
                
                # === 验证代码：打印标签 ===
                if ep == 0 and batch_idx == 0:
                    print(f"\n\n[DEBUG CHECK] Training Labels (Batch 0):")
                    print(f"  Shape: {c.shape}")
                    print(f"  First 5 labels in this batch:\n{c[:5].numpy()}")
                    print("  (请检查上方打印出的标签是否为你预期的 0/1 组合)\n")
                # ========================

                multi_optim.zero_grad()
                x = x.to(device)
                c = c.to(device).float()

                # 不再做无条件mask
                # mask_prob = 0
                # mask = torch.bernoulli(torch.zeros(c.shape[0]) + (1 - mask_prob)).to(device)
                # mask = mask.unsqueeze(-1).expand_as(c)
                # c = c * mask + (1 - mask) * -1
                # 直接用真实标签
                # c = c

                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
                x_pert = perturb_input(x, t, noise)

                pred_noise = multi_model(x_pert, t / timesteps, c=c)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                multi_optim.step()

            # [修改] 保存频率：改为每 100 个 epoch 保存一次，或者是最后一次
            if ep % 100 == 0 or ep == n_epoch - 1:
                model_path = os.path.join(exp_dir, f"context_model_{ep}.pth")
                torch.save(multi_model.state_dict(), model_path)

        # 测试
        multi_model.eval()
        final_model_path = os.path.join(exp_dir, f"context_model_{n_epoch - 1}.pth")
        multi_model.load_state_dict(torch.load(final_model_path, map_location=device))

        print(f"   Loaded final model for testing (excluded={excluded_combo})")

        for test_combo in all_16_combos:
            context_batch = torch.tensor([test_combo] * 10).float().to(device)
            
            # [修改] 传入 multi_model 进行采样
            test_samples, _ = sample_ddpm_context(multi_model, context_batch.shape[0], context_batch)
            
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
            
            vis_fname = f"vis_{'_'.join(map(str, test_combo))}.png"
            vis_path = os.path.join(save_vis_dir, vis_fname)
            fig.savefig(vis_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(f"[Saved visualization to {vis_path}]")

        print(f"=== Finished multi-run experiment excluding {excluded_combo} ===")

    print("\nAll multi-run experiments done!")
    endtime = time.time()
    print(f"Total time: {endtime - starttime} seconds")