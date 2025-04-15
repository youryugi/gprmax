# 重新导入必要的库
import numpy as np
import os

# 加载数据
data_file = "X_data.npy"  # RSSI 数据
label_file = "Y_data.npy"  # 标签数据

data = np.load(data_file)  # Shape: (8000, 28, 28, 4)
labels = np.load(label_file)  # Shape: (8000, 4)

# 计算所有唯一的标签组合
unique_labels = np.unique(labels, axis=0)

# 存储平均值数据
output_dir = "averaged_rssi_per_label"
os.makedirs(output_dir, exist_ok=True)

# 计算每个标签下的平均 RSSI
for label in unique_labels:
    # 找到属于当前标签的数据索引
    indices = np.all(labels == label, axis=1)
    subset_data = data[indices]  # 选出当前标签的数据

    # 计算平均值
    avg_rssi = np.mean(subset_data, axis=0, keepdims=True)  # [1, 28, 28, 4]

    # 生成文件名，如 "label_0_0_0_0.npy"
    label_str = "_".join(map(str, label))
    file_name = f"mean_{label_str}.npy"

    # 保存到文件
    np.save(os.path.join(output_dir, file_name), avg_rssi)

# 返回生成的文件列表
os.listdir(output_dir)
