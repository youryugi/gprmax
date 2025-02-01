import numpy as np
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量

# 加载文件
data = np.load('./user_defined_context_samples.npy')
#labels = np.load('./sprite_labels_nc_1788_16x16.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

#print("Labels shape:", labels.shape)
#print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[0])  # 查看第一个样本
#print("Labels sample:", labels[0])  # 查看第一个标签
print('=======================')
dataxyz=data[0]
print(dataxyz[0])
print('=======================')
print(data[1][0][0])
print('=======================')
print(data[2][0][0])