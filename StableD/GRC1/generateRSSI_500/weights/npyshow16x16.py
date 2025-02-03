import numpy as np
np.set_printoptions(threshold=np.inf)  # 设置 NumPy 不限制打印数量
# 加载文件
n=0
data = np.load('./user_defined_context_samples.npy')

print("Data shape:", data.shape)
print("Data type:", data.dtype)

print('=======================')
print("Data sample:", data[n])  # 查看第一个样本

print()

sample = data[0]

# 确保数据形状符合 (8, 16, 16, 4)
assert sample.shape == (16, 16, 4), "数据形状不符合预期"

# 按 16x16 显示 4 次
for i in range(4):
    print(f"第 {i+1} 个通道数据:")
    print(sample[:, :, i])  # 输出第 i 个通道的 16x16 数据
    print("\n" + "-"*50 + "\n")

