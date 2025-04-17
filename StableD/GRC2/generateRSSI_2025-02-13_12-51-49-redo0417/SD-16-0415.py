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
from diffusion_utilities_0201 import *  # 你自己的工具函数/类
from datetime import datetime
import itertools
import os
starttime=time.time()
#print(dir())  # 列出当前作用域的所有变量和函数

# -----------------------------
#        模型与函数定义
# -----------------------------

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=28):  # cfeat - context features
        super(ContextUnet, self).__init__()
        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  # assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)  # down1
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h // 4)), nn.GELU())

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)

        # Initialize the up-sampling path of the U-Net with three levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),  # up-sample
            nn.GroupNorm(8, 2 * n_feat),  # normalize
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),  # reduce number of feature maps
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, in_channels, h, w) : input image
        t : (batch, 1)                 : time step (already normalized to [0,1])
        c : (batch, n_cfeat)           : context label
        """
        # pass the input image through the initial convolutional layer
        x_init = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x_init)
        down2 = self.down2(down1)

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # if context is None, set to zero
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x_init), 1))
        return out


def normalize_rssi(data):
    """ 归一化到 [-1, 1] """
    return (data + 60) / 40

def denormalize_rssi(data):
    """ 反归一化回原始 RSSI 范围，可根据实际需要调节；此处假设 [-60, -20] """
    return data * 40 - 60

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    # 限制一下范围
    mean = torch.clamp(mean, -1, 1)
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    """
    无条件采样（仅时间，不含context），保留原示例。
    """
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_height).to(device)
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    """
    根据给定的 context 采样。
    context.shape = (batch_size, n_cfeat)
    """
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
    # 反归一化
    samples = denormalize_rssi(samples)
    return samples, intermediate

def show_images(imgs, nrow=2):
    """
    简单可视化函数，如不需要可不使用
    """
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4, 2))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()

def save_samples_as_npy(samples, save_path):
    """
    将推理结果保存为 [batch, RSSI_height, RSSI_width, d_RSSI] 的 .npy 文件
    :param samples: 推理结果，形状为 [batch, d_RSSI, RSSI_height, RSSI_width]
    :param save_path: 保存路径
    """
    samples_np = samples.detach().cpu().numpy()
    # 调整形状为 [batch, RSSI_height, RSSI_width, d_RSSI]
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))
    np.save(save_path, samples_np)
    print(f"Saved samples to {save_path} with shape {samples_np.shape}")


if __name__ == "__main__":
    # =====================
    #     1. 加载数据
    # =====================
    data_file = r"X_data.npy"
    label_file = r"Y_data.npy"
    data = np.load(data_file)   # [N, H, W, d_RSSI]
    labels = np.load(label_file)  # [N, 4] 假设4维二进制标签

    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    RSSI_height = data.shape[1]
    RSSI_width = data.shape[2]
    d_RSSI = data.shape[3]
    print("d_RSSI:", d_RSSI, 'height', RSSI_height, 'width', RSSI_width)

    # 自定义 transform，用于将数据范围 [-60, -20] 大致映射到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-60,), (40,))  # 将 RSSI (-100, -20) 大致映射到 [-1, 1]
    ])

    # 构造数据集
    dataset = CustomDataset(data_file, label_file, transform, null_context=False)

    # =====================
    #     2. 设置超参数
    # =====================
    timesteps = 400#论文中是400
    #beta1 = 1e-3
    beta1 = 1e-4#论文中是10-4
    #终止噪声系数，即扩散过程的最大噪声方差，默认为beta2 = 0.05
    beta2 = 0.02 #论文是0.02

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64           # U-Net中间通道数
    n_cfeat = labels.shape[1]  # context维度 (这里是4)
    batch_size = 100
    n_epoch = 100
    lrate = 1e-4#lrate	学习率（learning rate），初始设定为 1e-3（即 0.001） 论文中是10-4和10-5

    # 构造扩散噪声表(全局变量, 方便在函数里使用)
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # 初始化网络与优化器
    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    def perturb_input(x, t, noise):
        """
        将干净数据 x 扩散到第 t 步
        x : shape [B, d_RSSI, H, W]
        t : scalar or shape [B]
        noise : 随机噪声
        """
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    nn_model.train()
    # ===============================================
    #     4. 循环16次，每次排除一个组合进行训练
    # ===============================================
    # 如果不需要这个多轮排除实验，可以将此段注释掉

    # -- 在新的目录下保存每次排除结果 --
    multi_exp_root = './weights_16_experiments/'
    os.makedirs(multi_exp_root, exist_ok=True)

    # 所有16种标签组合
    all_16_combos = list(itertools.product([0, 1], repeat=4))

    for excluded_combo in all_16_combos:
        # 4.1 当前实验文件夹
        exp_dir = os.path.join(multi_exp_root, f"excluded_{'_'.join(map(str, excluded_combo))}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n=== Training (multi-run) excluding {excluded_combo} ===")

        # 4.2 只保留除 excluded_combo 之外的15种标签
        train_combos_15 = [combo for combo in all_16_combos if combo != excluded_combo]
        # 找到对应的数据索引
        train_indices_15 = []
        for i_lbl, lbl_arr in enumerate(labels):
            lbl_tuple = tuple(lbl_arr.tolist())
            if lbl_tuple in train_combos_15:
                train_indices_15.append(i_lbl)

        train_dataset_15 = Subset(dataset, train_indices_15)
        dataloader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True, num_workers=1)

        # 4.3 重新初始化一个新模型 (独立于前面的 nn_model)
        multi_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
        multi_optim = torch.optim.Adam(multi_model.parameters(), lr=lrate)

        # ---- 开始训练 (排除1个combo) ----
        multi_model.train()
        for ep in range(n_epoch):
            print(f'Multi-run epoch {ep} (exclude {excluded_combo})')
            multi_optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
            for x, c in tqdm(dataloader_15, mininterval=2):
                multi_optim.zero_grad()
                x = x.to(device)
                c = c.to(device).float()

                # 随机mask
                context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
                c = c * context_mask.unsqueeze(-1)

                noise = torch.randn_like(x)
                t = torch.randint(1, timesteps + 1, (x.shape[0],), device=device)
                x_pert = perturb_input(x, t, noise)

                pred_noise = multi_model(x_pert, t / timesteps, c=c)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                multi_optim.step()

            # 每隔 10 epoch或最后一个epoch 保存
            if ep % 10 == 0 or ep == n_epoch - 1:
                model_path = os.path.join(exp_dir, f"context_model_{ep}.pth")
                torch.save(multi_model.state_dict(), model_path)
                print(f"   [Saved model at {model_path}]")

        # # 4.4 测试：对16种组合分别采样
        # multi_model.eval()
        # final_model_path = os.path.join(exp_dir, f"context_model_{n_epoch - 1}.pth")
        # multi_model.load_state_dict(torch.load(final_model_path, map_location=device))
        # print(f"   Loaded final model for testing (excluded={excluded_combo})")

        # 遍历所有16种标签进行采样
        for test_combo in all_16_combos:
            context_batch = torch.tensor([test_combo] * 10).float().to(device)
            # 采样
            test_samples, _ = sample_ddpm_context(context_batch.shape[0], context_batch)

            # 判断 test_combo 是否被当前模型训练过
            # 如果 test_combo 不等于 excluded_combo, 那么就曾经在训练集(除非在数据中根本没有该组合)
            trained_dir = "trained" if test_combo in train_combos_15 else "untrained"

            # 存储路径
            combo_dir = os.path.join(exp_dir, trained_dir, f"context_{'_'.join(map(str, test_combo))}")
            os.makedirs(combo_dir, exist_ok=True)

            # 保存
            npy_path = os.path.join(combo_dir, "samples.npy")
            test_samples_t = torch.tensor(test_samples, dtype=torch.float32, device=device)
            save_samples_as_npy(test_samples_t, npy_path)

        print(f"=== Finished multi-run experiment excluding {excluded_combo} ===")

    print("\nAll multi-run experiments done!")
endtime=time.time()
alltime=endtime-starttime
print(alltime)