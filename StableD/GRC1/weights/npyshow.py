import numpy as np

# 加载文件
data = np.load('./user_defined_context_samples.npy')
#labels = np.load('./sprite_labels_nc_1788_16x16.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

#print("Labels shape:", labels.shape)
#print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[4])  # 查看第一个样本
#print("Labels sample:", labels[0])  # 查看第一个标签
