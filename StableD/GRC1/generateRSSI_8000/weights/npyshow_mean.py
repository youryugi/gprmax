import numpy as np

# 加载文件
data = np.load(r'C:\Users\79152\Desktop\3rdtopic\StableD\GRC1\generateRSSI_8000\weights\user_defined_context_samples_all16x10.npy')

print("Data shape:", data.shape)
print('=======================')
print("Data sample:", data[4])  # 查看第一个样本

# Reshape the data into groups of 500

# 2. 确保数据可以整除 10 进行分组
num_groups = data.shape[0] // 10
reshaped_data = data[:num_groups * 10].reshape(num_groups, 10, 16, 16, 4)

# 3. 计算均值
mean_data = reshaped_data.mean(axis=1)

# 4. 打印结果形状
print("均值数据的形状:", mean_data.shape)  # 应该是 (16, 16, 16, 4)

# 5. 可选：保存计算后的数据
np.save("./mean_user_defined_context_samples_all16x10.npy", mean_data)