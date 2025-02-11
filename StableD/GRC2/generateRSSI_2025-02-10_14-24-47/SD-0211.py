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

print(dir())  # 列出当前作用域的所有变量和函数

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
        self.to_vec = nn.Sequential(nn.AvgPool2d((self.h//4)), nn.GELU())

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
    """ 反归一化回原始 RSSI 范围 [-100, -20] 可根据实际需要调节，这里示例假设 [-60, -20] 等范围 """
    return data * 40 - 60

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    # 这里可以考虑对z做一定范围限制，不过一般不做也行
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    # 限制一下范围
    mean = torch.clamp(mean, -1, 1)
    return mean + noise

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_height).to(device)

    # array to keep track of generated steps for plotting (如果需要的话)
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
    # 初始噪声
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
    # 加载数据
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

    # 自定义的 transform，用于将数据范围 [-60, -20] 大致映射到 [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-60,), (40,))  # 让 RSSI (-100, -20) → [-1, 1]
    ])

    # 构造数据集
    dataset = CustomDataset(data_file, label_file, transform, null_context=False)

    # 选择“部分”训练标签组合（示例）。这里只训练其中 4 种组合
    train_combinations = [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 1, 1),
        (1, 0, 0, 0),
        (1, 0, 0, 1),
        (1, 0, 1, 0),
        (1, 0, 1, 1),
        (1, 1, 0, 0),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ]

    # 根据 train_combinations 过滤出仅包含这些标签的索引
    all_indices = list(range(len(labels)))
    train_indices = []
    for i in all_indices:
        lbl_tuple = tuple(labels[i].tolist())  # 把 [0,1,0,0] 这种转成 (0,1,0,0)
        if lbl_tuple in train_combinations:
            train_indices.append(i)

    # 使用 Subset 只训练这些组合的数据
    train_dataset = Subset(dataset, train_indices)

    # 其他超参数
    timesteps = 500
    beta1 = 1e-3
    beta2 = 0.05

    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64
    n_cfeat = labels.shape[1]  # 4
    save_dir = './weights/'

    batch_size = 100
    n_epoch = 100
    lrate = 1e-3

    # 构造扩散噪声表
    global b_t, a_t, ab_t  # 为了在函数里使用
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # 初始化网络
    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)

    # 注意，这里我们用刚才的 train_dataset 而不是 dataset
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    # 训练
    nn_model.train()
    for ep in range(n_epoch):
        print(f'epoch {ep}')

        # linearly decay learning rate
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader, mininterval=2)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device).float()

            # 随机mask掉一部分 context
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)

            # 扩散步骤 t
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)

            # 预测噪声
            pred_noise = nn_model(x_pert, t / timesteps, c=c)

            # 损失
            # 也可以用L1或者smooth_l1看效果
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optim.step()

        # 保存模型
        if ep % 10 == 0 or ep == n_epoch - 1:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(nn_model.state_dict(), os.path.join(save_dir, f"context_model_{ep}.pth"))
            print('saved model at ' + save_dir + f"context_model_{ep}.pth")

    # 测试（推理）阶段：加载最后一个模型
    nn_model.load_state_dict(torch.load(os.path.join(save_dir, f"context_model_{n_epoch - 1}.pth"),
                                        map_location=device))
    nn_model.eval()
    print("Loaded in Context Model for testing.")

    # == 测试时我们希望对所有 16 种标签都做采样 ==
    all_combinations = list(itertools.product([0, 1], repeat=4))  # 全部 16 种 (0/1) 组合
    # 每种组合重复 10 次
    expanded_data = np.repeat(all_combinations, 10, axis=0)
    ctx = torch.tensor(expanded_data).float().to(device)

    # 一次性采样
    samples, _ = sample_ddpm_context(ctx.shape[0], ctx)

    # 将采样结果拆分成 16 组，每组 10 个
    samples_per_context = np.split(samples, len(all_combinations))

    # 分别存到不同文件夹；如果该标签在训练集里则进 trained_contexts，否则进 untrained_contexts
    trained_context_dir = os.path.join(save_dir, "trained_contexts")
    untrained_context_dir = os.path.join(save_dir, "untrained_contexts")
    os.makedirs(trained_context_dir, exist_ok=True)
    os.makedirs(untrained_context_dir, exist_ok=True)

    for i, combo in enumerate(all_combinations):
        combo_samples = samples_per_context[i]  # 形状 [10, d_RSSI, H, W]
        combo_str = "_".join(map(str, combo))
        # 判断是否在训练标签组合里
        if combo in train_combinations:
            out_path = os.path.join(trained_context_dir, f"context_{combo_str}_samples.npy")
        else:
            out_path = os.path.join(untrained_context_dir, f"context_{combo_str}_samples.npy")

        # 这里复用之前的 save_samples_as_npy 函数，但要把 combo_samples 转回 tensor
        combo_samples_tensor = torch.tensor(combo_samples, dtype=torch.float32, device=device)
        save_samples_as_npy(combo_samples_tensor, out_path)

    print("All test samples saved, separated by trained/untrained contexts.")
