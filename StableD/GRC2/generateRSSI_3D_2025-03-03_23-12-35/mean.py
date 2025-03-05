import numpy as np
import os

# 指定目标文件夹
target_dir = "your_folder_path_here"  # 替换为目标文件夹路径

# 获取指定文件夹下所有 .npy 文件
npy_files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]

# 遍历所有 .npy 文件并计算平均值
for npy_file in npy_files:
    # 加载数据
    file_path = os.path.join(target_dir, npy_file)
    data = np.load(file_path)  # 预期 shape: (10, 28, 28, 4)

    # 确保数据的形状符合预期
    if data.shape == (10, 28, 28, 4):
        # 计算平均值，保持维度为 (1, 28, 28, 4)
        mean_data = np.mean(data, axis=0, keepdims=True)

        # 生成新文件名，前面加 "mean"
        mean_filename = f"mean_{npy_file}"
        mean_file_path = os.path.join(target_dir, mean_filename)

        # 保存新文件
        np.save(mean_file_path, mean_data)

        print(f"Processed {npy_file} → Saved {mean_filename}")

print("All files processed.")
