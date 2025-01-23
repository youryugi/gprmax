import numpy as np

# 加载 context 标签文件
labels = np.load("./sprite_labels_nc_1788_16x16.npy")

# 检查数据的形状
print(f"Shape of labels: {labels.shape}")

# 查看前几个样本
print("First 5 labels:")
print(labels[:100])
