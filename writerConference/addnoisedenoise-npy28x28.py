import numpy as np
import matplotlib.pyplot as plt
import os

# ========== 1. 加载 28x28 的 RSSI 数据 ==========
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-04-15_16-12-30\X_data.npy"
data = np.load(file1)
data = data[4799][:,:,1]  # 提取你需要的那个通道，得到 (28, 28)

assert data.shape == (28, 28), f"数据形状错误：{data.shape}"

# ========== 2. 数据归一化 ==========
data_min, data_max = np.min(data), np.max(data)
normalized_data = (data - data_min) / (data_max - data_min)

# ========== 3. 定义扩散过程参数 ==========
total_steps = 100
alpha_values = np.linspace(0.99, 0.001, total_steps)
beta_values = 1 - alpha_values
save_indices = [0, total_steps // 3, (2 * total_steps) // 3, total_steps - 1]  # 四个关键步骤

# ========== 4. 创建保存文件夹 ==========
output_dir = "noised_heatmaps_with_noise"
os.makedirs(output_dir, exist_ok=True)

# ========== 5. 添加噪声、绘制 RSSI 和 噪声图 ==========
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 两行：上 RSSI 下 噪声

for i, step in enumerate(save_indices):
    alpha_t = alpha_values[step]
    beta_t = beta_values[step]

    # 加入噪声
    noise = np.random.normal(loc=0, scale=1.0, size=normalized_data.shape)
    noisy_data = np.sqrt(alpha_t) * normalized_data + np.sqrt(beta_t) * noise

    # 上排：带噪 RSSI 热图
    im1 = axes[0, i].imshow(noisy_data, cmap="viridis", interpolation="nearest")
    axes[0, i].set_title(f"Noisy RSSI\nStep {step}, α={alpha_t:.3f}")
    axes[0, i].axis('off')

    # 下排：噪声图（直接可视化 ε）
    im2 = axes[1, i].imshow(noise, cmap="bwr", interpolation="nearest")  # bwr 表示正负值
    axes[1, i].set_title(f"Noise ε\nStep {step}")
    axes[1, i].axis('off')

    # 保存数据（可选）
    np.save(os.path.join(output_dir, f"noisy_data_step_{i+1}.npy"), noisy_data)
    np.save(os.path.join(output_dir, f"noise_step_{i+1}.npy"), noise)
    plt.imsave(os.path.join(output_dir, f"heatmap_rssi_step_{i+1}.png"), noisy_data, cmap="viridis")
    plt.imsave(os.path.join(output_dir, f"heatmap_noise_step_{i+1}.png"), noise, cmap="bwr")

# 添加 colorbar 到右下角最后一个图
fig.colorbar(im1, ax=axes[0, :].ravel().tolist(), shrink=0.6, orientation='horizontal', pad=0.05)
fig.colorbar(im2, ax=axes[1, :].ravel().tolist(), shrink=0.6, orientation='horizontal', pad=0.05)

plt.tight_layout()
plt.show()

print("已完成带噪 RSSI 与对应噪声图的生成与保存！")
