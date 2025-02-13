import time
import itertools
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# ========== 如果你有自己的 utilities 文件，可在此 import ==========
# from diffusion_utilities_0201 import CustomDataset  # 假设你在此文件中有定义
# 这里示例简单实现一个 CustomDataset
class CustomDataset(torch.utils.data.Dataset):
    """
    data_file -> X_data.npy: shape [N, H, W, d_RSSI]
    label_file -> Y_data.npy: shape [N, 4]
    transform: 对 x 做 [-60,-20]->[-1,1] 等映射
    null_context: 若为 True, 置标签为0
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
        x = self.data[idx]   # [H, W, d_RSSI]
        c = self.labels[idx] # [4]

        # 转为 [d_RSSI, H, W]
        x = torch.from_numpy(x).permute(2, 0, 1).float()
        if self.transform:
            x = self.transform(x)

        if self.null_context:
            c = np.zeros_like(c)
        c = torch.from_numpy(c).float()

        return x, c


# ========== 1. RSSI 归一化 / 反归一化工具函数 ==========
def normalize_rssi(data):
    """假设原始RSSI约在[-60,-20]之间，映射到[-1,1]。"""
    return (data + 60.0) / 40.0 * 2.0 - 1.0

def denormalize_rssi(data):
    """从[-1,1]反映射回[-60,-20]左右。"""
    return (data + 1.0) / 2.0 * 40.0 - 60.0

def save_samples_as_npy(samples, save_path):
    """
    将 [batch, d_RSSI, H, W] 保存为 .npy 文件 (转成[batch,H,W,d_RSSI])
    """
    samples_np = samples.detach().cpu().numpy()
    samples_np = np.transpose(samples_np, (0, 2, 3, 1))  # => [batch, H, W, d_RSSI]
    np.save(save_path, samples_np)
    print(f"Saved samples to {save_path} with shape {samples_np.shape}")


# ========== 2. 定义 cVAE ==========
class EncoderCVAE(nn.Module):
    """
    将 (x, c) -> (mu, logvar)
    - x: [batch, d_RSSI, H, W]
    - c: [batch, label_dim], label_dim=4
    - 输出: mu, logvar [batch, z_dim]
    """
    def __init__(self, d_rssi=1, label_dim=4, in_size=28, n_feat=64, z_dim=128):
        super(EncoderCVAE, self).__init__()
        self.d_rssi = d_rssi
        self.label_dim = label_dim
        self.in_size = in_size
        self.n_feat = n_feat
        self.z_dim = z_dim

        # 把标签 c 做一个简单嵌入(线性)再扩展到特征图的方式
        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 16),
            nn.ReLU()
        )
        # 用卷积下采样 x + label_emb(扩展为 feature map)
        # 这里演示：先把 label_emb => [B,16] => [B,16,H,W] => cat到 x => [B, d_rssi+16, H, W]
        # 再经几层卷积 => [B, n_feat*4, H//4, W//4] => flatten => fc => mu, logvar
        self.conv_down = nn.Sequential(
            nn.Conv2d(d_rssi + 16, n_feat, kernel_size=4, stride=2, padding=1),  # => [B, n_feat, H/2, W/2]
            nn.BatchNorm2d(n_feat),
            nn.ReLU(True),

            nn.Conv2d(n_feat, n_feat*2, kernel_size=4, stride=2, padding=1),     # => [B, 2*n_feat, H/4, W/4]
            nn.BatchNorm2d(n_feat*2),
            nn.ReLU(True),
        )
        # 最后映射到均值/对数方差
        # channel = n_feat*2 * (in_size//4) * (in_size//4)
        self.out_size = n_feat * 2 * (in_size//4) * (in_size//4)
        self.fc_mu    = nn.Linear(self.out_size, z_dim)
        self.fc_logvar= nn.Linear(self.out_size, z_dim)

    def forward(self, x, c):
        bsz = x.size(0)
        # c_emb => [B, 16]
        c_emb = self.label_emb(c)
        # 扩展到 [B, 16, H, W]
        c_map = c_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, 16, x.shape[2], x.shape[3])
        # 与 x 拼接
        inp = torch.cat([x, c_map], dim=1)  # => [B, d_rssi+16, H, W]

        h = self.conv_down(inp)  # => [B, n_feat*2, H/4, W/4]
        h = h.view(bsz, -1)      # => [B, out_size]

        mu = self.fc_mu(h)       # => [B, z_dim]
        logvar = self.fc_logvar(h)
        return mu, logvar


class DecoderCVAE(nn.Module):
    """
    将 (z, c) -> x_recon
    - z: [batch, z_dim]
    - c: [batch, label_dim]
    - 输出: x_recon [batch, d_rssi, H, W]
    """
    def __init__(self, d_rssi=1, label_dim=4, out_size=28, n_feat=64, z_dim=128):
        super(DecoderCVAE, self).__init__()
        self.d_rssi = d_rssi
        self.label_dim = label_dim
        self.out_size = out_size
        self.n_feat = n_feat
        self.z_dim = z_dim

        self.label_emb = nn.Sequential(
            nn.Linear(label_dim, 16),
            nn.ReLU()
        )
        # 解码过程:
        # 1) (z + c_emb) => fc => [B, n_feat*2*(out_size//4)*(out_size//4)]
        # 2) reshape => 转置卷积/上采样 => [B, d_rssi, out_size, out_size]
        self.fc = nn.Linear(z_dim + 16, n_feat*2 * (out_size//4) * (out_size//4))

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=4, stride=2, padding=1),   # => [B, n_feat, out_size/2, out_size/2]
            nn.BatchNorm2d(n_feat),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_feat, d_rssi, kernel_size=4, stride=2, padding=1),     # => [B, d_rssi, out_size, out_size]
            nn.Tanh()  # 输出到 [-1,1]
        )

    def forward(self, z, c):
        bsz = z.size(0)
        c_emb = self.label_emb(c)  # => [B,16]
        zc = torch.cat([z, c_emb], dim=1)  # => [B, z_dim+16]

        h = self.fc(zc)  # => [B, n_feat*2*(out_size//4)*(out_size//4)]
        h = h.view(bsz, self.n_feat*2, self.out_size//4, self.out_size//4)

        x_recon = self.conv_up(h)  # => [B, d_rssi, out_size, out_size]
        return x_recon


class CVAE(nn.Module):
    """
    封装 Encoder+Decoder，方便训练/推断
    """
    def __init__(self, d_rssi=1, label_dim=4, in_size=28, n_feat=64, z_dim=128):
        super(CVAE, self).__init__()
        self.encoder = EncoderCVAE(d_rssi, label_dim, in_size, n_feat, z_dim)
        self.decoder = DecoderCVAE(d_rssi, label_dim, in_size, n_feat, z_dim)

    def reparameterize(self, mu, logvar):
        """
        reparameterization trick:
          z = mu + std * eps
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, c):
        """
        前向同时返回重建图 x_recon 以及 (mu, logvar) 便于计算 VAE loss
        """
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, c)
        return x_recon, mu, logvar


# ========== 3. 训练 & 采样函数 ==========

def vae_loss_function(x_recon, x, mu, logvar):
    """
    VAE 损失 = 重构误差 + KL 散度
    这里用 MSE 作为重构误差，可按需换成 BCE。
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))，可对 batch/通道做平均
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl
    return loss, recon_loss, kl


def train_cvae_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    for x, c in tqdm(dataloader, mininterval=2):
        x = x.to(device)   # [B, d_RSSI, H, W]
        c = c.to(device)   # [B, 4]
        optimizer.zero_grad()

        x_recon, mu, logvar = model(x, c)
        loss, r_loss, kl_loss = vae_loss_function(x_recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item()  * x.size(0)
        total_recon += r_loss.item()* x.size(0)
        total_kl    += kl_loss.item()*x.size(0)

    N = len(dataloader.dataset)
    return total_loss/N, total_recon/N, total_kl/N


@torch.no_grad()
def sample_cvae(model, context, n_sample=1, z_dim=128):
    """
    给定标签 context（[batch,4]），从 N(0,I) 中采样若干 z，生成 x。
    最终输出形状 [batch*n_sample, d_RSSI, H, W]。
    """
    device = next(model.parameters()).device
    # 扩展 context
    context_expanded = context.repeat_interleave(n_sample, dim=0)  # [batch*n_sample, 4]
    bsz = context_expanded.size(0)

    # 采样 z
    z = torch.randn(bsz, z_dim).to(device)
    # 解码
    x_gen = model.decoder(z, context_expanded)  # => [bsz, d_RSSI, H, W]
    # 反归一化到 RSSI
    x_gen = denormalize_rssi(x_gen)
    return x_gen


# ========== 4. 主逻辑示例 ==========

if __name__ == "__main__":
    starttime = time.time()

    # 4.1 数据准备
    data_file = r"X_data.npy"
    label_file = r"Y_data.npy"

    data = np.load(data_file)   # [N, H, W, d_RSSI]
    labels = np.load(label_file)  # [N, 4]
    print("Data shape:", data.shape)
    RSSI_height = data.shape[1]
    RSSI_width  = data.shape[2]
    d_RSSI = data.shape[3]
    print("d_RSSI:", d_RSSI, "height:", RSSI_height, "width:", RSSI_width)

    # transform: [-60,-20]->[-1,1]
    transform = transforms.Lambda(lambda x: normalize_rssi(x))
    dataset = CustomDataset(data_file, label_file, transform=transform, null_context=False)

    # 4.2 定义 cVAE & 超参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_dim = labels.shape[1]  # =4
    z_dim = 128
    n_feat = 64
    lr = 1e-4
    batch_size = 64
    n_epoch = 50

    cvae_model = CVAE(d_rssi=d_RSSI, label_dim=label_dim,
                      in_size=RSSI_height, n_feat=n_feat, z_dim=z_dim).to(device)

    optimizer = torch.optim.Adam(cvae_model.parameters(), lr=lr)

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
    train_indices = []
    for i, lbl in enumerate(labels):
        if tuple(lbl.tolist()) in train_combinations:
            train_indices.append(i)

    train_dataset = Subset(dataset, train_indices)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # 4.3 训练示例 (单次)
    save_dir = './cvae_weights/'
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(n_epoch):
        print(f"\n=== Epoch {ep}/{n_epoch} ===")
        avg_loss, avg_recon, avg_kl = train_cvae_one_epoch(cvae_model, dataloader, optimizer, device)
        print(f"Loss={avg_loss:.4f}  Recon={avg_recon:.4f}  KL={avg_kl:.4f}")

        # 可选：衰减 lr
        # optimizer.param_groups[0]['lr'] = lr * (1 - ep / n_epoch)

        # 间隔保存
        if ep % 10 == 0 or ep == n_epoch - 1:
            torch.save(cvae_model.state_dict(), os.path.join(save_dir, f"cvae_{ep}.pth"))
            print("Model saved.")

    # 4.4 测试采样 (单次)
    last_model_path = os.path.join(save_dir, f"cvae_{n_epoch-1}.pth")
    cvae_model.load_state_dict(torch.load(last_model_path, map_location=device))
    cvae_model.eval()
    print("Loaded final cVAE model for testing...")

    # 对16种标签组合各生成10张
    all_combinations = list(itertools.product([0,1], repeat=4))  # 16
    expanded_data = np.repeat(all_combinations, 10, axis=0)      # => [16*10,4]
    ctx = torch.tensor(expanded_data).float().to(device)

    samples = sample_cvae(cvae_model, ctx, n_sample=1, z_dim=z_dim)  # => [160, d_RSSI, H, W]
    samples_per_context = torch.split(samples, 10, dim=0)  # 切成16组

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

    # ================================================
    #  4.5 循环16次，每次排除一个组合进行训练 (可选)
    # ================================================
    multi_exp_root = './cvae_weights_16_experiments/'
    os.makedirs(multi_exp_root, exist_ok=True)

    all_16_combos = all_combinations  # 16
    for excluded_combo in all_16_combos:
        exp_dir = os.path.join(multi_exp_root, f"excluded_{'_'.join(map(str,excluded_combo))}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n=== Training cVAE excluding {excluded_combo} ===")

        # 只保留剩余15种
        train_combos_15 = [c for c in all_16_combos if c != excluded_combo]
        train_indices_15 = []
        for i_lbl, lbl_arr in enumerate(labels):
            if tuple(lbl_arr.tolist()) in train_combos_15:
                train_indices_15.append(i_lbl)

        train_dataset_15 = Subset(dataset, train_indices_15)
        dataloader_15 = DataLoader(train_dataset_15, batch_size=batch_size, shuffle=True, num_workers=1)

        # 重新初始化 cVAE
        cvae_15 = CVAE(d_rssi=d_RSSI, label_dim=label_dim, in_size=RSSI_height, n_feat=n_feat, z_dim=z_dim).to(device)
        optimizer_15 = torch.optim.Adam(cvae_15.parameters(), lr=lr)

        # 训练
        for ep in range(n_epoch):
            print(f"   epoch {ep} (exclude={excluded_combo})")
            avg_loss, avg_recon, avg_kl = train_cvae_one_epoch(cvae_15, dataloader_15, optimizer_15, device)
            # 可选：衰减
            # optimizer_15.param_groups[0]['lr'] = lr * (1 - ep/n_epoch)

            # 间隔保存
            if ep % 10 == 0 or ep == n_epoch-1:
                torch.save(cvae_15.state_dict(), os.path.join(exp_dir, f"cvae_{ep}.pth"))
                print(f"   Model saved at epoch {ep}.")

        # 测试：对16种组合采样
        cvae_15.eval()
        final_cvae_path = os.path.join(exp_dir, f"cvae_{n_epoch-1}.pth")
        cvae_15.load_state_dict(torch.load(final_cvae_path, map_location=device))

        for test_combo in all_16_combos:
            ctx_batch = torch.tensor([test_combo]*10).float().to(device)
            test_samples = sample_cvae(cvae_15, ctx_batch, n_sample=1, z_dim=z_dim)  # => [10, d_RSSI, H, W]

            # 判断 test_combo 是否在训练集
            trained_dir = "trained" if test_combo in train_combos_15 else "untrained"
            combo_dir = os.path.join(exp_dir, trained_dir, f"context_{'_'.join(map(str,test_combo))}")
            os.makedirs(combo_dir, exist_ok=True)

            npy_path = os.path.join(combo_dir, "samples.npy")
            save_samples_as_npy(test_samples, npy_path)

        print(f"=== Finished cVAE experiment excluding {excluded_combo} ===")

    endtime = time.time()
    print("All done! Total time:", endtime - starttime)
