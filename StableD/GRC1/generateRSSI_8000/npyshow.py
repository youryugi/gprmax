import numpy as np
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量
# 加载文件
n=500
data = np.load('./X_data.npy')
labels = np.load('./Y_data.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

print("Labels shape:", labels.shape)
print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[n])  # 查看第一个样本
print("Labels sample:", labels[n])  # 查看第一个标签
