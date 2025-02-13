import time
from typing import Dict, Tuple
import itertools
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 假设你有一个自定义的 utilities 文件，其中包括 CustomDataset 等
# from diffusion_utilities_0201 import CustomDataset  # 如果你之前的文件里已经写好这个类
# 这里给出一个简单示例，若你已有实现可直接替换
class CustomDataset(torch.utils.data.Dataset):
    """
    假设 __init__ 中加载了 data_file (X_data.npy) 和 label_file (Y_data.npy)
    transform: 对RSSI数据做 [-60, -20]->[-1, 1] 等操作
    null_context: 如果为 True, 则 label 都返回 0; 否则返回真实标签
    """
    def __init__(self, data_file, label_file, transform=None, null_context=False):
        super().__init__()
        self.data = np.load(data_file)     # [N, H, W, d_RSSI]
        self.labels = np.load(label_file)  # [N, 4]
        self.transform = transform
        self.null_context = null_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # shape [H, W, d_RSSI]
        c = self.labels[idx] # shape [4]

        # 转为tensor([d_RSSI, H, W]) 并做 transform
        x = torch.from_numpy(x).permute(2,0,1).float()  # [d_RSSI, H, W]

        if self.transform:
            x = self.transform(x)

        if self.null_context:
            c = np.zeros_like(c)
        c = torch.from_numpy(c).float()

        return x, c

# -----------------------------------------------------
#  1. 一些辅助函数
# -----------------------------------------------------

def normalize_rssi(data):
    """将 RSSI 从大约 [-60, -20] 线性映射到 [-1, 1]。若你数据范围不同，可自行修改。"""
    return (data + 60) / 40.0 * 2.0 - 1.0

def denormalize_rssi(data):
    """把 [-1, 1] 的数据反映射回到大约 [-60, -20]。"""
    return (data + 1.0) / 2.0 * 40.0 - 60.0

def save_samples_as_npy(samples, save_path):
    """
    将生成结果保存为 [batch, RSSI_height, RSSI_width, d_RSSI] 的 .npy 文件
    :param samples: 形状 [batch, d_RSSI, H, W] 的张量
    """
    samples_np = samples.detach().cpu().numpy()
    # 转为 [batch, H, W, d_RSSI]
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))
    np.save(save_path, samples_np)
    print(f"Saved samples to {save_path} with shape {samples_np.shape}")

# -----------------------------------------------------
#  2. 定义 cGAN 的 Generator 与 Discriminator
# -----------------------------------------------------

class CGANGenerator(nn.Module):
    """
    一个简易的 cGAN Generator。
    - 输入：噪声 z (shape=[batch, z_dim])，标签 c (shape=[batch, n_label_dim])
    - 输出：形状 [batch, d_RSSI, H, W] 的张量
    """
    def __init__(self, z_dim, label_dim, d_rssi=1, out_size=28, base_channels=64):
        super(CGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.label_dim = label_dim
        self.d_rssi = d_rssi
        self.out_size = out_size
        self.base_channels = base_channels

        # 将 label 做一个简单的嵌入层(全连接)，映射到某个维度，这里随意设 16
        embed_dim = 16
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, embed_dim),
            nn.ReLU()
        )

        # 生成器的主干：将 (z_dim + embed_dim) -> 再映射到 [base_channels*4, 7, 7] -> 上采样到 [d_rssi, 28, 28]
        self.fc = nn.Linear(z_dim + embed_dim, base_channels*4*7*7)

        # 这里用简单的上采样 + 卷积，也可用 ConvTranspose2d
        # 先 reshape 到 (base_channels*4, 7, 7)
        # 上采样到 (base_channels*2, 14,14)
        # 再上采样到 (base_channels, 28,28)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, d_rssi, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 让输出范围到 [-1, 1]
        )

    def forward(self, z, c):
        """
        :param z: shape = [batch, z_dim]
        :param c: shape = [batch, label_dim]
        """
        # 对 label 做嵌入
        c_emb = self.label_emb(c)  # [batch, embed_dim]

        # 与z拼接
        x = torch.cat([z, c_emb], dim=1)  # [batch, z_dim + embed_dim]

        x = self.fc(x)  # -> [batch, base_channels*4*7*7]
        x = x.view(x.size(0), self.base_channels*4, 7, 7)  # reshape

        out = self.conv_blocks(x)  # -> [batch, d_rssi, 28, 28]
        return out


class CGANDiscriminator(nn.Module):
    """
    简易 cGAN Discriminator。将图像与标签一起判别真或假。
    """
    def __init__(self, label_dim, d_rssi=1, in_size=28, base_channels=64):
        super(CGANDiscriminator, self).__init__()
        self.label_dim = label_dim
        self.d_rssi = d_rssi
        self.in_size = in_size
        self.base_channels = base_channels

        # 同样给 label 一个嵌入层
        embed_dim = 16
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 判别器输入的通道维度 = d_rssi + 1（或者用 embedding 的某种拼接方式）
        # 这里的简易做法：把标签嵌入后复制到整张 feature map 上再拼接
        # 你也可以采用 Projection Discriminator 等更高级方法
        self.label_to_feature = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)

        # 卷积主干
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(d_rssi + 1, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base_channels*2*(in_size//4)*(in_size//4), 1)
        )

    def forward(self, x, c):
        """
        :param x: 图像, shape=[batch, d_rssi, in_size, in_size]
        :param c: 标签, shape=[batch, label_dim]
        :return: 判别器输出, shape=[batch, 1]
        """
        # 先对标签做嵌入
        c_emb = self.label_emb(c)  # [batch, embed_dim]

        # 把 c_emb reshape 成 [batch, embed_dim, 1, 1]，然后扩展到 [batch, embed_dim, in_size, in_size]
        c_map = c_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, c_emb.shape[1], x.shape[2], x.shape[3])
        # c_map 通过一个 1x1 conv 或 3x3 conv 投影到单通道
        c_map = self.label_to_feature(c_map)  # [batch, 1, in_size, in_size]

        # 拼接到 x 的通道维度
        x_cat = torch.cat([x, c_map], dim=1)  # [batch, d_rssi+1, in_size, in_size]

        out = self.conv_blocks(x_cat)
        return out

# -----------------------------------------------------
#  3. 训练循环 & 采样函数
# -----------------------------------------------------

def train_cgan_one_epoch(gen, disc, dataloader, g_opt, d_opt, device, z_dim=100):
    """
    cGAN 单个 epoch 的训练过程。
    :param gen: CGANGenerator
    :param disc: CGANDiscriminator
    :param dataloader: 训练数据
    :param g_opt: 生成器优化器
    :param d_opt: 判别器优化器
    :param device: ...
    :param z_dim: 噪声维度
    """
    gen.train()
    disc.train()

    criterion = nn.BCEWithLogitsLoss()

    for real_x, c in tqdm(dataloader, mininterval=2):
        real_x = real_x.to(device)  # [B, d_RSSI, H, W]
        c = c.to(device)

        # ========== 1. 训练判别器 =============
        d_opt.zero_grad()

        # (a) 判别器判别真实图像
        b_size = real_x.size(0)
        real_label = torch.ones((b_size, 1), device=device)  # 真
        fake_label = torch.zeros((b_size, 1), device=device) # 假

        d_real = disc(real_x, c)
        loss_d_real = criterion(d_real, real_label)

        # (b) 判别器判别生成图像
        z = torch.randn(b_size, z_dim, device=device)
        fake_x = gen(z, c).detach()  # 不让生成器回传梯度
        d_fake = disc(fake_x, c)
        loss_d_fake = criterion(d_fake, fake_label)

        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        d_opt.step()

        # ========== 2. 训练生成器 =============
        g_opt.zero_grad()
        z = torch.randn(b_size, z_dim, device=device)
        gen_x = gen(z, c)  # [B, d_RSSI, H, W]
        d_gen = disc(gen_x, c)
        # 生成器希望判别器输出 1（即真）
        loss_g = criterion(d_gen, real_label)
        loss_g.backward()
        g_opt.step()

    # 这里只是简单返回最后一次batch的loss做参考
    return loss_d.item(), loss_g.item()


@torch.no_grad()
def sample_cgan(gen, context, n_sample=1, z_dim=100):
    """
    使用 cGAN 生成若干图像。
    :param gen: 训练好的生成器
    :param context: shape=[batch, label_dim] 的标签（可一次性传入多个）
    :param n_sample: 每个标签要生成多少张
    :param z_dim: 噪声维度
    :return: 生成图像张量, shape=[batch*n_sample, d_RSSI, H, W]
    """
    device = next(gen.parameters()).device

    # 如果 context 本身已经包含了 batch 维度，假设 context.shape = [B, label_dim]
    # 我们为每一个 context 生成 n_sample 张，则重复 context
    context_expanded = context.repeat_interleave(n_sample, dim=0)

    b_size = context_expanded.size(0)
    z = torch.randn(b_size, z_dim, device=device)
    samples = gen(z, context_expanded)  # -> [b_size, d_RSSI, H, W]

    # 反归一化到 [-60, -20] 区间
    samples = denormalize_rssi(samples)
    return samples


# -----------------------------------------------------
#  4. 主逻辑示例
# -----------------------------------------------------
if __name__ == "__main__":
    starttime = time.time()

    # ========== 4.1 数据准备 ================
    data_file = r"X_data.npy"
    label_file = r"Y_data.npy"
    data = np.load(data_file)   # [N, H, W, d_RSSI]
    labels = np.load(label_file)  # [N, 4]
    print("Data shape:", data.shape)
    RSSI_height = data.shape[1]
    RSSI_width  = data.shape[2]
    d_RSSI = data.shape[3]
    print("d_RSSI:", d_RSSI, 'height:', RSSI_height, 'width:', RSSI_width)

    # transform: [-60, -20] -> [-1, 1]
    transform = transforms.Lambda(lambda x: normalize_rssi(x))

    dataset = CustomDataset(data_file, label_file, transform=transform, null_context=False)

    # ========== 4.2 cGAN 网络 & 超参数 ============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_dim = 100       # 噪声维度 (可调)
    label_dim = 4     # 你这里标签是4维二进制
    lr = 1e-4         # 学习率
    batch_size = 64
    n_epoch = 50

    # 实例化生成器 & 判别器
    gen = CGANGenerator(z_dim=z_dim, label_dim=label_dim, d_rssi=d_RSSI, out_size=RSSI_height, base_channels=64).to(device)
    disc = CGANDiscriminator(label_dim=label_dim, d_rssi=d_RSSI, in_size=RSSI_height, base_channels=64).to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # 用于演示：只训练 15 种标签组合
    train_combinations = [
        (0,0,0,0),
        (0,0,0,1),
        (0,0,1,0),
        (0,0,1,1),
        (0,1,0,0),
        (0,1,0,1),
        (0,1,1,0),
        (0,1,1,1),
        (1,0,0,0),
        (1,0,0,1),
        (1,0,1,0),
        (1,0,1,1),
        (1,1,0,0),
        (1,1,0,1),
        (1,1,1,0),
    ]
    # 找到对应索引
    train_indices = []
    for i, lbl in enumerate(labels):
        if tuple(lbl.tolist()) in train_combinations:
            train_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # ========== 4.3 训练 (单次示例) ============
    save_dir = './cgan_weights/'
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(n_epoch):
        print(f"\n==== Epoch {ep}/{n_epoch} ====")
        d_loss, g_loss = train_cgan_one_epoch(gen, disc, dataloader, g_opt, d_opt, device, z_dim=z_dim)
        print(f"Epoch {ep}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

        # 可按需衰减 lr (与之前扩散示例类似)
        # g_opt.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)
        # d_opt.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

        # 间隔保存模型
        if ep % 10 == 0 or ep == n_epoch - 1:
            torch.save(gen.state_dict(),  os.path.join(save_dir, f"gen_{ep}.pth"))
            torch.save(disc.state_dict(), os.path.join(save_dir, f"disc_{ep}.pth"))
            print("Model saved.")

    # ========== 4.4 测试采样 (单次) ============
    # 加载最后的生成器
    gen_path = os.path.join(save_dir, f"gen_{n_epoch - 1}.pth")
    gen.load_state_dict(torch.load(gen_path, map_location=device))
    gen.eval()
    print("Loaded generator for testing...")

    # 对所有16种标签组合进行采样，每种组合生成10张
    all_combinations = list(itertools.product([0, 1], repeat=4))
    expanded_data = np.repeat(all_combinations, 10, axis=0)  # [16*10, 4]
    ctx = torch.tensor(expanded_data).float().to(device)

    # 一次性生成 (16种组合 * 10张 = 160张)
    samples = sample_cgan(gen, ctx, n_sample=1, z_dim=z_dim)  # 这里 n_sample=1，因为 ctx 已经手工 repeat
    # 切分成 16 组，每组 10 张
    samples_per_context = torch.split(samples, 10, dim=0)

    # 分别保存到 trained / untrained
    trained_context_dir = os.path.join(save_dir, "trained_contexts")
    untrained_context_dir = os.path.join(save_dir, "untrained_contexts")
    os.makedirs(trained_context_dir, exist_ok=True)
    os.makedirs(untrained_context_dir, exist_ok=True)

    for i, combo in enumerate(all_combinations):
        combo_samples = samples_per_context[i]  # [10, d_RSSI, H, W]
        combo_str = "_".join(map(str, combo))
        if combo in train_combinations:
            out_path = os.path.join(trained_context_dir, f"context_{combo_str}_samples.npy")
        else:
            out_path = os.path.join(untrained_context_dir, f"context_{combo_str}_samples.npy")
        save_samples_as_npy(combo_samples, out_path)

    print("Single partial-training test finished. Samples saved.")

    # ==========================================================
    #     4.5 循环16次，每次排除一个组合进行训练 (可选)
    # ==========================================================
    multi_exp_root = './cgan_weights_16_experiments/'
    os.makedirs(multi_exp_root, exist_ok=True)

    # 所有16种标签
    all_16_combos = all_combinations  # 与上面相同

    for excluded_combo in all_16_combos:
        exp_dir = os.path.join(multi_exp_root, f"excluded_{'_'.join(map(str, excluded_combo))}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n=== cGAN Training (multi-run) excluding {excluded_combo} ===")

        # 筛选除 excluded_combo 之外的15种标签
        train_combos_15 = [c for c in all_16_combos if c != excluded_combo]
        train_indices_15 = []
        for i_lbl, lbl_arr in enumerate(labels):
            if tuple(lbl_arr.tolist()) in train_combos_15:
                train_indices_15.append(i_lbl)

        train_dataset_15 = Subset(dataset, train_indices_15)
        dataloader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True, num_workers=1)

        # 重新初始化一个新的 cGAN
        gen_15 = CGANGenerator(z_dim=z_dim, label_dim=label_dim, d_rssi=d_RSSI, out_size=RSSI_height).to(device)
        disc_15 = CGANDiscriminator(label_dim=label_dim, d_rssi=d_RSSI, in_size=RSSI_height).to(device)
        g_opt_15 = torch.optim.Adam(gen_15.parameters(), lr=lr, betas=(0.5, 0.999))
        d_opt_15 = torch.optim.Adam(disc_15.parameters(), lr=lr, betas=(0.5, 0.999))

        # 开始训练
        for ep in range(n_epoch):
            print(f"   Multi-run epoch {ep} (exclude {excluded_combo})")
            d_loss, g_loss = train_cgan_one_epoch(gen_15, disc_15, dataloader_15, g_opt_15, d_opt_15, device, z_dim=z_dim)

            # 可自行衰减 lr
            # g_opt_15.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)
            # d_opt_15.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

            # 间隔保存
            if ep % 10 == 0 or ep == n_epoch - 1:
                torch.save(gen_15.state_dict(),  os.path.join(exp_dir, f"gen_{ep}.pth"))
                torch.save(disc_15.state_dict(), os.path.join(exp_dir, f"disc_{ep}.pth"))
                print(f"   [Saved model at epoch {ep}]")

        # 测试阶段：对16种组合分别采样
        gen_15.eval()
        final_gen_path = os.path.join(exp_dir, f"gen_{n_epoch - 1}.pth")
        gen_15.load_state_dict(torch.load(final_gen_path, map_location=device))

        for test_combo in all_16_combos:
            context_batch = torch.tensor([test_combo]*10).float().to(device)
            test_samples = sample_cgan(gen_15, context_batch, n_sample=1, z_dim=z_dim)  # 10张

            # 判断 test_combo 是否在训练集里
            trained_dir = "trained" if test_combo in train_combos_15 else "untrained"
            combo_dir = os.path.join(exp_dir, trained_dir, f"context_{'_'.join(map(str, test_combo))}")
            os.makedirs(combo_dir, exist_ok=True)

            npy_path = os.path.join(combo_dir, "samples.npy")
            save_samples_as_npy(test_samples, npy_path)

        print(f"=== Finished multi-run experiment excluding {excluded_combo} ===")

    endtime = time.time()
    print("All done! Total time:", endtime - starttime)
