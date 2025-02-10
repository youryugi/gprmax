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
# Reshape the data into groups of 500


# 2. 确保数据可以整除 500 进行分组
num_groups = data.shape[0] // 500
reshaped_data = data[:num_groups * 500].reshape(num_groups, 500, 16, 16, 4)

# 3. 计算均值
mean_data = reshaped_data.mean(axis=1)

# 4. 打印结果形状
print("均值数据的形状:", mean_data.shape)  # 应该是 (16, 16, 16, 4)

# 5. 可选：保存计算后的数据
np.save("./mean_X_data.npy", mean_data)