import numpy as np
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量
# 加载文件
n=0
data = np.load('./user_defined_context_samples.npy')

print("Data shape:", data.shape)
print("Data type:", data.dtype)

print('=======================')
#print("Data sample:", data[n])  # 查看第一个样本

print()

num_groups = data.shape[0] // 8
reshaped_data = data[:num_groups * 8].reshape(num_groups, 8, 16, 16, 4)
# 3. 计算均值
mean_data = reshaped_data.mean(axis=1)

# 4. 打印结果形状
print("均值数据的形状:", mean_data.shape)  # 应该是 (16, 16, 16, 4)

# 5. 可选：保存计算后的数据
np.save("./mean_user_defined_context_samples.npy", mean_data)
sample = mean_data[0]

# 确保数据形状符合 (8, 16, 16, 4)
assert sample.shape == (16, 16, 4), "数据形状不符合预期"

# 按 16x16 显示 4 次
for i in range(4):
    print(f"第 {i+1} 个通道数据:")
    print(sample[:, :, i])  # 输出第 i 个通道的 16x16 数据
    print("\n" + "-"*50 + "\n")

np.save("./mean_chanel1_user_defined_context_samples.npy", sample[:, :, 1])