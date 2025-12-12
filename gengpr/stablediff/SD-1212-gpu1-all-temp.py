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
        
        # 注意：height=128时，down2后是32x32。AvgPool(32)会变成1x1向量。
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
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        x_init = self.init_conv(x)
        down1 = self.down1(x_init)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x_init), 1))
        return out

def denormalize_image(data):
    """ 
    反归一化: 从 [-1, 1] 映射回 [0, 1] 用于显示 
    """
    return (data + 1) / 2

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    mean = torch.clamp(mean, -1, 1)
    return mean + noise

@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    # 注意：这里使用全局变量 d_RSSI, RSSI_height, RSSI_width
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t, c=context)
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
    #     1. 加载数据
    # =====================
    # 请修改为您的图片文件夹路径
    img_dir = r"./data_images/"  
    
    # 图像参数设置
    RSSI_height = 128
    RSSI_width = 128
    d_RSSI = 1  # 灰度图是 1 通道

    # Transform: Resize -> Tensor(0~1) -> Normalize(-1~1)
    transform = transforms.Compose([
        transforms.Resize((RSSI_height, RSSI_width)),
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

    # 获取所有标签用于后续筛选
    labels = dataset.labels 
    print("Data loaded. Total images:", len(dataset))
    print("Labels shape:", labels.shape)
    print("Image size:", (d_RSSI, RSSI_height, RSSI_width))

    # =====================
    #     2. 设置超参数
    # =====================
    timesteps = 400
    beta1 = 1e-4
    beta2 = 0.02

    device = torch.device("cuda:1" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64 
    n_cfeat = labels.shape[1] # 4
    batch_size = 64 # 图片变大了，Batch Size 可能需要调小一点，防止显存溢出
    n_epoch = 100
    lrate = 1e-4

    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    multi_exp_root = './weights_16_experiments_img/'
    os.makedirs(multi_exp_root, exist_ok=True)
    save_vis_dir = './result_vis_img/'
    os.makedirs(save_vis_dir, exist_ok=True)

    all_16_combos = list(itertools.product([0, 1], repeat=4))

    for excluded_combo in all_16_combos:
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

                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
                c = c * context_mask.unsqueeze(-1)

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
            test_samples, _ = sample_ddpm_context(context_batch.shape[0], context_batch)
            
            trained_dir = "trained" if test_combo in train_combos_15 else "untrained"
            combo_dir = os.path.join(exp_dir, trained_dir, f"context_{'_'.join(map(str, test_combo))}")
            os.makedirs(combo_dir, exist_ok=True)

            # 保存 npy
            npy_path = os.path.join(combo_dir, "samples.npy")
            test_samples_t = torch.tensor(test_samples, dtype=torch.float32, device=device)
            save_samples_as_npy(test_samples_t, npy_path)

            # 可视化 (修改部分)
            # 取平均值
            test_samples_t_mean = torch.mean(test_samples_t, dim=0, keepdim=True) # [1, 1, 128, 128]
            
            # 反归一化 [-1, 1] -> [0, 1]
            img_show = denormalize_image(test_samples_t_mean)
            
            # 转为 numpy [H, W] (去掉 batch 和 channel 维度，因为是灰度)
            img_show = img_show[0, 0, :, :].cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(5, 5))
            # 使用 gray colormap
            ax.imshow(img_show, cmap="gray", vmin=0, vmax=1)
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