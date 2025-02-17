import os
import numpy as np
import pandas as pd

# 设置两个文件夹路径
folder1 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\truedata"
folder2 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\predicteddata"

# 设置允许的误差百分比
acc_threshold = 15  # 可以修改此值来改变允许的误差范围

# 获取两个文件夹中的所有 npy 文件名
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# 取交集，确保只处理两个文件夹中都存在的文件
common_files = sorted(files1.intersection(files2), key=lambda x: int(x.split('.')[0]))

# 存储结果
results = []

# 遍历所有相同文件
for file_name in common_files:
    file1_path = os.path.join(folder1, file_name)
    file2_path = os.path.join(folder2, file_name)

    # 读取 npy 数据
    data1 = np.load(file1_path)  # 形状 (1, 28, 28, 4)
    data2 = np.load(file2_path)  # 形状 (1, 28, 28, 4)

    # 取均值保持一致形状
    data2 = np.mean(data2, axis=0, keepdims=True)

    # 确保形状匹配
    if data1.shape != (1, 28, 28, 4) or data2.shape != (1, 28, 28, 4):
        print(f"跳过 {file_name}，数据形状不匹配: {data1.shape} vs {data2.shape}")
        continue

    # 计算误差并取平均值
    error_percentage = np.abs(data1 - data2) / (np.abs(data1) + 1e-6) * 100  # 避免除零
    accuracy_map = error_percentage <= acc_threshold  # 误差在允许范围内的点
    accuracy_ratio = np.sum(accuracy_map) / (28 * 28 * 4)  # 计算整个数组的准确率

    # 存储结果
    results.append([file_name, round(accuracy_ratio, 4)])  # 保留 4 位小数

# 转换为 DataFrame
df_results = pd.DataFrame(results, columns=["File Name", "Accuracy Ratio"])

# 生成 CSV 文件名，包含误差阈值
output_csv = rf".\accuracy_results_{acc_threshold}.csv"
df_results.to_csv(output_csv, index=False, float_format="%.4f")  # 确保精度为 4 位小数

print(f"结果已保存至 {output_csv}")
