import numpy as np

# 加载文件
data = np.load(r'C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\weights_16_experiments\excluded_0_0_0_0\trained\context_0_0_0_1\samples.npy')
labels = np.load(r'C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\weights_16_experiments\excluded_0_0_0_0\untrained\context_0_0_0_0\samples.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

print("Labels shape:", labels.shape)
print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[0])  # 查看第一个样本
print("Labels sample:", labels[0])  # 查看第一个标签
