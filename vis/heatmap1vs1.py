import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC1\generateRSSI_2025-02-01_22-50-06\mean_chanel1_mean_X_data.npy"  # 你的基准文件
file2 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC1\generateRSSI_2025-02-01_22-50-06\weights\mean_chanel1_user_defined_context_samples.npy"  # 另一个文件

# 读取数据
data1 = np.load(file1)  # 形状应为 (16, 16, 1)
print(data1.shape)
data2 = np.load(file2)  # 形状应为 (16, 16, 1)
print(data2.shape)
# 确保数据形状正确

#data1 = np.flipud(data1)
#data2 = np.flipud(data2)  #数据翻转

assert data1.shape == (16, 16), f"数据1形状不匹配: {data1.shape}"
assert data2.shape == (16, 16), f"数据2形状不匹配: {data2.shape}"

# 计算差值
diff = data1[:, :] - data2[:, :]  # 去掉最后一个维度，使其变成 (16,16)
# 计算 data1 和 data2 的最小最大值，确保颜色条相同
vmin = min(data1.min(), data2.min())  # 取最小值
vmax = max(data1.max(), data2.max())  # 取最大值

# 创建 3 个子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 可视化 data1，设置相同的 vmin 和 vmax
im1 = axes[0].imshow(data1, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[0].set_title("Data1 Heatmap")
fig.colorbar(im1, ax=axes[0])

# 可视化 data2，设置相同的 vmin 和 vmax
im2 = axes[1].imshow(data2, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
axes[1].set_title("Data2 Heatmap")
fig.colorbar(im2, ax=axes[1])

# 可视化差值图
im3 = axes[2].imshow(diff, cmap="coolwarm", interpolation="nearest")
axes[2].set_title("Difference Heatmap")
fig.colorbar(im3, ax=axes[2])

# 设置整体标题
plt.suptitle("Visualization of Data1, Data2, and Their Differences")
plt.show()