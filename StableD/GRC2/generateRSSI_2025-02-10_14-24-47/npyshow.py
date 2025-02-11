import numpy as np

# 加载文件
data = np.load(r'C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\averaged_rssi_per_label\mean_1_1_1_1.npy')
labels = np.load(r'C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-02-10_14-24-47\weights\trained_contexts\context_1_1_1_0_samples.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

print("Labels shape:", labels.shape)
print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[0])  # 查看第一个样本
print("Labels sample:", labels[0])  # 查看第一个标签
