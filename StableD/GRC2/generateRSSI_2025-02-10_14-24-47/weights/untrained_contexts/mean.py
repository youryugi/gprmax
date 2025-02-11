import numpy as np
import os

# 获取当前文件夹下所有 .npy 文件
current_dir = os.getcwd()
npy_files = [f for f in os.listdir(current_dir) if f.endswith(".npy")]

# 遍历所有 .npy 文件并计算平均值
for npy_file in npy_files:
    # 加载数据
    data = np.load(npy_file)  # 预期 shape: (10, 28, 28, 4)

    # 确保数据的形状符合预期
    if data.shape == (10, 28, 28, 4):
        # 计算平均值，保持维度为 (1, 28, 28, 4)
        mean_data = np.mean(data, axis=0, keepdims=True)

        # 生成新文件名，前面加 "mean"
        mean_filename = f"mean_{npy_file}"

        # 保存新文件
        np.save(os.path.join(current_dir, mean_filename), mean_data)

        print(f"Processed {npy_file} → Saved {mean_filename}")

print("All files processed.")
