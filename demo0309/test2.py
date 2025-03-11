import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============== 1. 环境与路径设置 ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_folder = "results"
model_timestamp = "20250309_160201"
model_path = os.path.join(results_folder, f"{model_timestamp}_train", f"trained_model_{model_timestamp}.pth")
scaler_load_path = r"C:\Users\79152\Desktop\3rdtopic\demo0309\results\20250309_160201_train\scaler_20250309_160201.npy"
location_list_path = r"C:\Users\79152\Desktop\3rdtopic\demo0309\location-list.xlsx"
location_data = pd.read_excel(location_list_path)

label_to_coords = {idx: (row["x"], row["y"]) for idx, row in location_data.iterrows()}

target_ssids = ["ytest1-2.4g", "ytest2-2.4g", "ytest3-2.4g", "ytest4-2.4g"]


# =============== 2. 定义并加载模型 ===============
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


input_size = 4
num_classes = len(location_data)
model = NeuralNet(input_size, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# =============== 3. 加载训练好的标准化参数 ===============
from sklearn.preprocessing import StandardScaler

scaler_params = np.load(scaler_load_path, allow_pickle=True).item()
scaler = StandardScaler()
scaler.mean_ = scaler_params["mean"]
scaler.scale_ = scaler_params["scale"]

# =============== 4. 位置可视化参数 ===============
space_size_x = 10.35
space_size_y = 14.8
walls = [
    [(0, 6.3), (10.35, 6.3)],  # 水平墙
    [(7.1, 0), (7.1, 6.3)]  # 垂直墙
]

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, space_size_x)
ax.set_ylim(0, space_size_y)
ax.set_xlabel("X (m)",fontsize=20)
ax.set_ylabel("Y (m)",fontsize=20)
ax.set_title("Predicted Location in Lab  ⬆North",fontsize=20)
ax.grid(False)

# 画墙壁
for wall in walls:
    (x1, y1), (x2, y2) = wall
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

plt.ion()  # 使绘图窗口保持打开状态
plt.show()


def update_location(pred_coords):
    ax.scatter(pred_coords[0], pred_coords[1], c='red', s=100, label='Predicted Location')
    plt.draw()
    plt.pause(0.5)


# =============== 5. 进入循环，不断进行扫描和推理 ===============
# =============== 5. 使用指定 RSSI 数据进行推测 ===============
custom_rssi = [-55,	-42	,-58,-57]
X_new = np.array(custom_rssi).reshape(1, -1)
X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)

pred_label = predicted.item()
coords = label_to_coords.get(pred_label, ("未知", "未知"))
print(f"预测标签: {pred_label}，位置坐标: {coords}\n")

if coords != ("未知", "未知"):
    update_location(coords)

plt.ioff()
plt.show()

