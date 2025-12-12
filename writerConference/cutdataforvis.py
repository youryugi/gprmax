import numpy as np
import matplotlib.pyplot as plt

# 加载数据
file1 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-04-15_16-12-30\X_data.npy"  # 你的基准文件

# 读取数据
data1 = np.load(file1)  # 形状应为 (16, 16, 1)
datatemp=data1[4799,:,:,:]
datatemp=datatemp[None,:,:,:]
print(datatemp.shape)



# 假设 datatemp 是一个 NumPy 数组，shape 为 (1, 16, 16, 4)
np.save('data1111/datatemp_1_1_1_1.npy', datatemp)

data1=data1[4799]
data1=data1[:,:,0]
print(data1.shape)


assert data1.shape == (28, 28), f"数据1形状不匹配: {data1.shape}"

# 创建 3 个子图
fig, ax = plt.subplots(figsize=(5, 5))

# 绘制 heatmap
im = ax.imshow(data1, cmap="viridis", interpolation="nearest")
ax.set_title("[0,0,1,0] Heatmap")


# 设置整体标题
#plt.suptitle("Visualization of Data1, Data2, and Their Differences")
plt.show()