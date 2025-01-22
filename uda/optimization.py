import pandas as pd
import numpy as np

# 读取xlsx文件
file_path = r"C:\Users\79152\Desktop\myData\review1sd\D.xlsx" # 替换为你的文件路径
data = pd.read_excel(file_path, header=None)

# 提取标签和特征
labels = data.iloc[:, 0]  # 第一列为标签
features = data.iloc[:, 1:]  # 剩余列为特征

# 计算每个标签的特征均值
unique_labels = labels.unique()
label_means = {
    label: features[labels == label].mean(axis=0).values for label in unique_labels
}

# 构建距离矩阵
distance_matrix = pd.DataFrame(
    index=unique_labels, columns=unique_labels, dtype=float
)

for label_a, mean_a in label_means.items():
    for label_b, mean_b in label_means.items():
        distance = np.linalg.norm(mean_a - mean_b)
        distance_matrix.loc[label_a, label_b] = distance

# 打印结果
print("距离矩阵：")
print(distance_matrix)

# 保存结果为新的xlsx文件
distance_matrix.to_excel('label_distance_matrixD.xlsx')
