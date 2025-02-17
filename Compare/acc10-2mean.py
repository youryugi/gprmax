import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-13_12-51-49\averaged_rssi_per_label\mean_0_0_0_0.npy"
file2 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-13_12-51-49\weights_16_experiments\excluded_0_0_0_0\untrained\context_0_0_0_0\samples.npy"

# 读取数据
data1 = np.load(file1)  # 形状 (1, 28, 28, 4)
data2 = np.load(file2)  # 形状 (1, 28, 28, 4)

data2 = np.mean(data2, axis=0, keepdims=True)

# 确保数据形状匹配
assert data1.shape == (1, 28, 28, 4), f"数据1形状不匹配: {data1.shape}"
assert data2.shape == (1, 28, 28, 4), f"数据2形状不匹配: {data2.shape}"

# 选择通道进行可视化
channel_index = 3  # 可以改成 1, 2, 3 选择不同通道

# 取出对应通道的数据
data1 = data1[0, :, :, channel_index]  # (28, 28)
data2 = data2[0, :, :, channel_index]  # (28, 28)

# 计算误差百分比
error_percentage = np.abs(data1 - data2) / (np.abs(data1) + 1e-6) * 100  # 避免除零

# 计算准确率（误差在 10% 以内的点）
accuracy_map = error_percentage <= 10
accuracy_ratio = np.sum(accuracy_map) / (28 * 28)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始 RSSI 可视化
im1 = axes[0].imshow(data1, cmap="viridis", interpolation="nearest")
axes[0].set_title(f"Ground Truth (router {channel_index+1})")
fig.colorbar(im1, ax=axes[0])

# 生成的 RSSI 可视化
im2 = axes[1].imshow(data2, cmap="viridis", interpolation="nearest")
axes[1].set_title(f"Generated Data (router {channel_index+1})")
fig.colorbar(im2, ax=axes[1])

# 准确度可视化（准确为白色，误差超过10%为黑色）
im3 = axes[2].imshow(accuracy_map, cmap="gray", interpolation="nearest")
axes[2].set_title(f"Accuracy Map (White: Accurate, Black: Inaccurate)")
fig.colorbar(im3, ax=axes[2])

# 设置整体标题
plt.suptitle(f"Accuracy Comparison: Channel {channel_index}\nAccuracy Ratio: {accuracy_ratio:.2%}")
plt.show()
