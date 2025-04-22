import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-04-15_16-12-30\X_data.npy"

# 读取数据
data1 = np.load(file1)
data1 = data1[4790][:, :, 1]

assert data1.shape == (28, 28), f"数据1形状不匹配: {data1.shape}"

# 创建干净的图
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(data1, cmap="viridis", interpolation="nearest")
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)  # 去掉边框

plt.tight_layout()
plt.show()
