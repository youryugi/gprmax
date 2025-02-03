import numpy as np
import matplotlib.pyplot as plt
import os

# 设定数据路径
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC1\generateRSSI_8000\mean_X_data.npy"
file2 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC1\generateRSSI_8000\weights\mean_user_defined_context_samples_all16x10.npy"

# 读取数据 (n, 16, 16, 4)
data1 = np.load(file1)
data2 = np.load(file2)

# 确保数据形状相同
assert data1.shape == data2.shape, f"数据形状不匹配: {data1.shape} vs {data2.shape}"
assert len(data1.shape) == 4 and data1.shape[1:] == (16, 16, 4), f"数据形状不匹配: {data1.shape}"

# 计算差值
diff = data1 - data2

# 设定保存目录
save_dir = r"C:\Users\79152\Desktop\3rdtopic\vis\16x4_16x4"
os.makedirs(save_dir, exist_ok=True)

# 遍历 `n` 个样本
for i in range(data1.shape[0]):
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 3行4列：data1, data2, diff

    for j in range(4):
        # 可视化 Data1
        im1 = axes[0, j].imshow(data1[i, :, :, j], cmap="viridis", interpolation="nearest")
        axes[0, j].set_title(f"Data1 - Sample {i + 1} - Channel {j + 1}")
        fig.colorbar(im1, ax=axes[0, j])

        # 可视化 Data2
        im2 = axes[1, j].imshow(data2[i, :, :, j], cmap="viridis", interpolation="nearest")
        axes[1, j].set_title(f"Data2 - Sample {i + 1} - Channel {j + 1}")
        fig.colorbar(im2, ax=axes[1, j])

        # 可视化差值
        im3 = axes[2, j].imshow(diff[i, :, :, j], cmap="coolwarm", interpolation="nearest")
        axes[2, j].set_title(f"Difference - Sample {i + 1} - Channel {j + 1}")
        fig.colorbar(im3, ax=axes[2, j])

    plt.suptitle(f"Visualization of Sample {i + 1} (Data1, Data2, and Their Difference)")

    # 保存图片
    save_path = os.path.join(save_dir, f"sample_{i + 1}.png")
    plt.savefig(save_path)
    plt.close(fig)

print(f"所有可视化结果已保存到 {save_dir}")
