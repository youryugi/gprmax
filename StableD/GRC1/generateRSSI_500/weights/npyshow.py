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
#print(dataxyz[0])
print('=======================')
print(data[1][0][0])
print('=======================')
print(data[2][0][0])
print(data[3][0][0])
print(data[4][0][0])
print(data[5][0][0])
print(data[6][0][0])
print(data[7][0][0])


sample = data[0]

# 确保数据形状符合 (8, 16, 16, 4)
assert sample.shape == (16, 16, 4), "数据形状不符合预期"

# 按 16x16 显示 4 次
for i in range(4):
    print(f"第 {i+1} 个通道数据:")
    print(sample[:, :, i])  # 输出第 i 个通道的 16x16 数据
    print("\n" + "-"*50 + "\n")
