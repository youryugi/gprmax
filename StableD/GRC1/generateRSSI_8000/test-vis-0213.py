from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from IPython.display import HTML
from diffusion_utilities_0201 import *
from datetime import datetime
import itertools
#16种都测试
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
        self.down1 = UnetDown(n_feat, n_feat)  # down1 #[10, 256, 8, 8]
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 #[10, 256, 4,  4]

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())

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
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  # [10, 256, 8, 8]
        down2 = self.down2(down1)  # [10, 256, 4, 4]

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down2)

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def normalize_rssi(data):
    return (data + 100) / 80  # 归一化到 [0, 1]，因为 -100 ~ -20

def denormalize_rssi(data):
    return data * 80 - 100  # 反归一化回 RSSI 真实值

# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        #z = torch.randn_like(x)
        z = torch.randn_like(x) * 0.5  # 限制噪声范围 [-1,1]
#TODO: 此处要适应RSSI的数据范围
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    mean = torch.clamp(mean, -1, 1)  # 限制在 RSSI 范围内  这也是新加的
    return mean + noise
# helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
# sample using standard algorithm
@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_height).to(device)#因为是正方形，所以是两个height

    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t)    # predict noise e_(x_t,t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate ==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate
# sample with context using standard algorithm
@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    #samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)
    # TODO：1. 生成符合 [-1,1] 范围的初始噪声
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device) * 0.5 - 0.5
    # array to keep track of generated steps for plotting
    intermediate = []
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t, ctx)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    #TODO
    # 3. 反归一化数据 [-1,1] → [-100, -20]
    samples = denormalize_rssi(samples)
    return samples, intermediate
def denormalize_rssi(x_norm):
    """ 将归一化数据转换回原始 RSSI 范围 """
    return x_norm * 40 - 60
def show_images(imgs, nrow=2):
    _, axs = plt.subplots(nrow, imgs.shape[0] // nrow, figsize=(4,2 ))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        img = (img.permute(1, 2, 0).clip(-1, 1).detach().cpu().numpy() + 1) / 2
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img)
    plt.show()

# 在推理完成后保存输出为 .npy 文件
def save_samples_as_npy(samples, save_path):
    """
    将推理结果保存为 [x, RSSI_height, RSSI_width, d_RSSI] 的 .npy 文件
    :param samples: 推理结果，形状为 [x, d_RSSI, RSSI_height, RSSI_width]
    :param save_path: 保存路径
    """
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    samples_np = samples.detach().cpu().numpy()

    # 调整形状为 [x, RSSI_height, RSSI_width, d_RSSI]
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))

    # 保存为 .npy 文件
    np.save(save_path, samples_np)
    print(f"Saved samples to {save_path} with shape {samples_np.shape}")

if __name__ == "__main__":
    data_file=r".\X_data.npy"
    label_file=r".\Y_data.npy"
    dataset = CustomDataset(data_file,label_file, transform, null_context=False)
    data = np.load(data_file)
    labels = np.load(label_file)
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)
    RSSI_height=data.shape[1]
    RSSI_width=data.shape[2]

    d_RSSI=data.shape[3]#datashape的第四个维度是  输入的通道数
    print("d_RSSI:",d_RSSI,'height',RSSI_height,'width',RSSI_width)
    # hyperparameters
    timesteps = 500
    #beta1 = 1e-4
    #beta2 = 0.02
    #TODO:
    # 调整 beta 以适应 RSSI 数据范围
    beta1 = 1e-3  # 增大 beta1 适应 RSSI 信号
    beta2 = 0.05  # 增大 beta2 让扩散过程适应更大信号变化

    # network hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
    n_feat = 64  # 64 hidden dimension feature
    n_cfeat = labels.shape[1]  # context vector is of size是label的shape的第二个维度
    save_dir = './weights/'

    # training hyperparameters
    batch_size = 100
    n_epoch = 100
    lrate = 1e-3

    # construct DDPM noise schedule
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1

    # reset neural network
    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)

    # re setup optimizer
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    # load dataset and construct optimizer
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-60,), (40,))  # 让 RSSI (-100, -20) → [-1, 1]
        #transforms.Normalize((0.5,), (0.5,))
    ])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    # helper function: perturbs an image to a specified noise level
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    # training with context code
    # load in pretrain model weights and set to eval mode
    nn_model.load_state_dict(torch.load(f"{save_dir}/context_model_99.pth", map_location=device))
    nn_model.eval()
    print("Loaded in Context Model")
    # 在推理完成后调用保存函数
    # 示例：保存随机上下文生成的样本
    # 示例：保存用户定义上下文生成的样本
    combinations = list(itertools.product([0, 1], repeat=4))

    # 每个组合重复10次
    expanded_data = np.repeat(combinations, 10, axis=0)
    ctx = torch.tensor(expanded_data).float().to(device)
    samples, _ = sample_ddpm_context(ctx.shape[0], ctx)
    save_samples_as_npy(samples, f"{save_dir}/user_defined_context_samples_all16x10.npy")

import matplotlib.pyplot as plt


def plot_key_inference_steps(
        intermediate: np.ndarray,
        denormalize_fn,
        steps_to_show=[500, 400, 300, 200],
        sample_idx=0,
        d_rssi=1,
        vmin=-100,
        vmax=-20
):
    """
    可视化推理时在 4 个关键步骤生成的 RSSI map。（不修改原有的 sample_ddpm_context 函数）

    参数说明：
    -----------
    intermediate: shape 为 [num_saved_steps, batch_size, d_RSSI, height, width]
                  来自 sample_ddpm_context 返回的第二个值。

    denormalize_fn: 用于反归一化的函数，比如您代码中定义的 denormalize_rssi(x)。

    steps_to_show: 要展示的 4 个关键步骤(从大到小)，例如 [500, 400, 300, 200]。
                   注意必须是和 sample_ddpm_context 中 if i % save_rate==0 (默认 20)
                   能对应上的步数，否则在 intermediate 里找不到。
                   例如 i=500→index=0, i=480→index=1, i=460→index=2, i=1→最后几帧……

    sample_idx:    当一次采样(batch)生成多张时，这里指定可视化第几张图，默认 0 即第一张。

    d_rssi:        通道数，如果是单通道(1)则可用 imshow 直接显示，如果是多通道也可以自行改写。

    vmin, vmax:    画热力图时的数值范围，对应 RSSI 的最小值(-100)和最大值(-20)，
                   仅在绘制热力图时用于控制颜色映射区间。
    """

    # intermediate[0] 对应 i=timesteps；中间依次每 save_rate 步存一次；直到 i<8 时每步都存
    # 如果 steps_to_show 不在这个保存列表里，则无法索引到正确步骤。

    # 先构造“步数 i”到 intermediate 下标的映射
    # 默认 save_rate=20 且从 500 到 1，i=500 => index=0, i=480 => index=1, ...
    def step_to_index(i):
        # 因为 i 每隔 20 保存一次，故可以用 (timesteps - i)//save_rate
        # 注意 i==timesteps时是 intermediate[0]
        return (500 - i) // 20

    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(5 * len(steps_to_show), 5))

    for col, step in enumerate(steps_to_show):
        idx = step_to_index(step)
        # intermediate[idx] 的 shape: [batch_size, d_RSSI, height, width]
        # 这里默认只显示指定 sample_idx 的第 0 通道
        img_tensor = intermediate[idx, sample_idx, 0, :, :] if d_rssi == 1 else intermediate[idx, sample_idx]

        # 将其从 [-1,1] 的归一化范围反归一化到真实 RSSI 值
        img_rssi = denormalize_fn(img_tensor)

        # 可视化
        ax = axes[col]
        im = ax.imshow(img_rssi, cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(f"Step {step}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


