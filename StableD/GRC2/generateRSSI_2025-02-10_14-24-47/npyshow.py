import numpy as np

# 加载文件
data = np.load('./generateRSSI_2025-01-23_21-25-38/X_data.npy')
labels = np.load('./generateRSSI_2025-01-23_21-25-38/Y_data.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

print("Labels shape:", labels.shape)
print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[4])  # 查看第一个样本
print("Labels sample:", labels[4])  # 查看第一个标签
