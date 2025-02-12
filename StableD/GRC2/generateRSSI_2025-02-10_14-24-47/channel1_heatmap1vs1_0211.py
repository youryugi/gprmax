import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\averaged_rssi_per_label\mean_1_1_1_1.npy"
file2 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\weights\untrained_contexts\mean_context_1_1_1_1_samples.npy"

# 读取数据
data1 = np.load(file1)  # 形状 (1, 28, 28, 4)
data2 = np.load(file2)  # 形状 (1, 28, 28, 4)

# 确保数据形状匹配
assert data1.shape == (1, 28, 28, 4), f"数据1形状不匹配: {data1.shape}"
assert data2.shape == (1, 28, 28, 4), f"数据2形状不匹配: {data2.shape}"

# 选择通道进行可视化
channel_index = 0  # 可以改成 1, 2, 3 选择不同通道

# 取出对应通道的数据
data1 = data1[0, :, :, channel_index]  # (28, 28)
data2 = data2[0, :, :, channel_index]  # (28, 28)

# 计算差值
diff = data1 - data2

# 计算 data1 和 data2 的最小最大值，确保颜色条相同
vmin = min(data1.min(), data2.min())
vmax = max(data1.max(), data2.max())

# 创建 3 个子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 可视化 data1
im1 = axes[0].imshow(data1, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Heatmap of Ground Truth (router {channel_index+1})")
fig.colorbar(im1, ax=axes[0])

# 可视化 data2
im2 = axes[1].imshow(data2, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[1].set_title(f"Heatmap of Generated Data (router {channel_index+1})")
fig.colorbar(im2, ax=axes[1])

# 可视化差值
im3 = axes[2].imshow(diff, cmap="coolwarm", interpolation="nearest")
axes[2].set_title(f"Difference Heatmap (router {channel_index+1})")
fig.colorbar(im3, ax=axes[2])

# 设置整体标题
plt.suptitle(f"Visualization of Channel {channel_index}: Ground Truth, Generated Data, and Differences")
plt.show()
