import numpy as np
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量
# 加载文件
n=0
data = np.load('./mean_X_data.npy')
labels = np.load('./Y_data.npy')
print("Data shape:", data.shape)
print("Data type:", data.dtype)

print("Labels shape:", labels.shape)
print("Labels type:", labels.dtype)
print('=======================')
print("Data sample:", data[n])  # 查看第一个样本
print("Labels sample:", labels[n])  # 查看第一个标签
print()

sample = data[0]

# 确保数据形状符合 (8, 16, 16, 4)
assert sample.shape == (16, 16, 4), "数据形状不符合预期"

# 按 16x16 显示 4 次
for i in range(4):
    print(f"第 {i+1} 个通道数据:")
    print(sample[:, :, i])  # 输出第 i 个通道的 16x16 数据
    print("\n" + "-"*50 + "\n")

np.save("./mean_chanel1_mean_X_data.npy", sample[:, :, 1])
