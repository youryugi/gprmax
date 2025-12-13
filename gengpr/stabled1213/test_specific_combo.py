import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from diffusion_utilities_3d_1212 import *

# ==========================================
#   配置区域 (请修改这里)
# ==========================================
MODEL_PATH = "/home/yang/gprmax/gengpr/stabled1213/weights_16_experiments_img_20251213_152133/excluded_1_0_1_0/context_model_299.pth"
TARGET_LABEL = [0, 0, 0, 0]
GUIDE_W = 2.0
DEVICE_ID = "0"

# ==========================================
#   模型定义 (必须与训练时一致)
# ==========================================
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
        self.up1 = UnetUp(2 * n_feat, n_feat, skip_channels=2 * n_feat)
        self.up2 = UnetUp(n_feat, n_feat, skip_channels=n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        x_init = self.init_conv(x)
        down1_skip, down1 = self.down1(x_init)
        down2_skip, down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2_skip)
        up3 = self.up2(cemb2 * up2 + temb2, down1_skip)
        out = self.out(torch.cat((up3, x_init), 1))
        return out

# ==========================================
#   采样函数 (支持 CFG)
# ==========================================
timesteps = 1000
beta1 = 1e-4
beta2 = 0.02
device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")

b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    mean = torch.clamp(mean, -1, 1)
    return mean + noise

@torch.no_grad()
def sample_ddpm_context(model, n_sample, context, guide_w=0.0):
    d_RSSI = 3
    RSSI_height = 128
    RSSI_width = 128

    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device)

    context_uncond = torch.ones_like(context) * -1
    # context_uncond = torch.zeros_like(context)  # 若旧模型用 0 mask，请改用此行

    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t_batch = torch.full((n_sample,), i / timesteps, device=device)
        t_batch = torch.cat([t_batch, t_batch], dim=0).unsqueeze(-1)

        z = torch.randn_like(samples) if i > 1 else 0

        samples_repeat = samples.repeat(2, 1, 1, 1)
        c_concat = torch.cat([context, context_uncond], dim=0)

        eps_concat = model(samples_repeat, t_batch, c=c_concat)
        eps_cond, eps_uncond = eps_concat.chunk(2, dim=0)
        eps = eps_uncond + guide_w * (eps_cond - eps_uncond)

        samples = denoise_add_noise(samples, i, eps, z)

    return samples

# ==========================================
#   主程序
# ==========================================
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    print(f"Using device: {device}")

    n_feat = 64
    n_cfeat = 4
    height = 128
    model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("请在代码顶部修改 MODEL_PATH 变量！")
        exit()

    print(f"Loading model from {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    num_samples = 1
    target_tensor = torch.tensor([TARGET_LABEL] * num_samples).float().to(device)

    print(f"Generating {num_samples} images for label: {TARGET_LABEL}")
    print(f"Guidance Weight (CFG): {GUIDE_W}")

    samples = sample_ddpm_context(model, num_samples, target_tensor, guide_w=GUIDE_W)

    grid = make_grid(samples, nrow=5, normalize=True, value_range=(-1, 1))
    img_show = grid.permute(1, 2, 0).cpu().numpy()

    label_str = "_".join(map(str, TARGET_LABEL))
    save_name = f"test_result_{label_str}_cfg{GUIDE_W}.png"

    plt.figure(figsize=(12, 6))
    plt.imshow(img_show)
    plt.axis('off')
    plt.title(f"Generated Result for {TARGET_LABEL} (CFG={GUIDE_W})")
    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0.1)
    print(f"\nResult saved to: {save_name}")
    plt.show()