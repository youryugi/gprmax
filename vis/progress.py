import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


# 加载数据
def load_intermediate_results(save_path):
    return np.load(save_path)  # 形状: (steps, samples, H, W, channels)


# 可视化扩散去噪过程
def visualize_diffusion_process(data, selected_sample=0, selected_channel=0, save_gif="ddpm_radio_map_evolution.gif",
                                save_mp4="ddpm_radio_map_evolution.mp4"):
    timesteps, num_samples, H, W, channels = data.shape
    frames = data[:, selected_sample, :, :, selected_channel]

    fig, ax = plt.subplots(figsize=(5, 5))
    heatmap = ax.imshow(frames[0], cmap="viridis", interpolation="nearest")
    ax.set_title(f"Timestep 0")

    def update(frame):
        heatmap.set_data(frames[frame])
        ax.set_title(f"Timestep {frame}")

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save(save_gif, writer="pillow", fps=10)  # 保存为 GIF
    ani.save(save_mp4, writer="ffmpeg", fps=10)  # 保存为 MP4
    plt.show()


# 归一化和反归一化函数
def normalize_rssi(data):
    return (data + 100) / 80  # 归一化到 [0, 1]


def denormalize_rssi(data):
    return data * 80 - 100  # 反归一化回 RSSI 真实值


@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20, device="cuda"):
    samples = torch.randn(n_sample, d_RSSI, RSSI_height, RSSI_width).to(device) * 0.5 - 0.5  # 初始化噪声
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
    samples = denormalize_rssi(samples)
    return samples, intermediate


# 运行并保存中间结果
if __name__ == "__main__":
    save_dir = "./weights/"
    combinations = list(itertools.product([0, 1], repeat=4))
    expanded_data = np.repeat(combinations, 10, axis=0)
    ctx = torch.tensor(expanded_data).float().to(device)
    samples, intermediate = sample_ddpm_context(ctx.shape[0], ctx)
    np.save(f"{save_dir}/user_defined_context_samples_all16x10.npy", intermediate)
    print(f"Saved intermediate results with shape {intermediate.shape}")

    # 可视化扩散过程
    visualize_diffusion_process(intermediate)
