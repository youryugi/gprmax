# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:25:20 2024
@author: 79152
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import os
from datetime import datetime
from sklearn.metrics import accuracy_score

# 设置随机种子，保证实验可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 超参数
num_epochs = 200  # 训练轮数
batch_size = 64   # 批量大小
learning_rate = 0.001  # 学习率

# 数据路径
source_data_path = r'C:\Users\79152\Desktop\3rdtopic\demo0309\close0309_all.xlsx'  # Source 数据集
location_list_path = r'C:\Users\79152\Desktop\3rdtopic\demo0309\location-list.xlsx'  # 位置标签

# 结果保存路径
results_folder = "results"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
script_name = os.path.basename(__file__).split('.')[0]
folder_name = os.path.join(results_folder, f"{current_time}_{script_name}")
os.makedirs(folder_name, exist_ok=True)

# 模型保存路径
model_save_path = os.path.join(folder_name, f"trained_model_{current_time}.pth")

# 读取数据
df = pd.read_excel(source_data_path, header=0)
location_data = pd.read_excel(location_list_path)

# 创建标签到坐标的映射
label_to_coords = {row['loc']: (row['x'], row['y']) for _, row in location_data.iterrows()}

# 分离特征和标签
X = df.iloc[:, 1:].values  # 特征
y = df.iloc[:, 0].values  # 标签

# 数据集拆分（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# 标准化
scaler = StandardScaler()
# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存 scaler 参数
scaler_params = {
    "mean": scaler.mean_,
    "scale": scaler.scale_
}
scaler_save_path = os.path.join(folder_name, f"scaler_{current_time}.npy")
np.save(scaler_save_path, scaler_params)
print(f"标准化参数已保存到 {scaler_save_path}")


# 转换为 PyTorch Tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 定义神经网络模型
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

# 初始化模型
num_classes = len(np.unique(y))
model = NeuralNet(X_train.shape[1], num_classes)

# 选择 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# 训练模型
print("开始训练模型...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# 训练完成后保存模型
torch.save(model.state_dict(), model_save_path)
print(f"训练完成，模型已保存到 {model_save_path}")

# ----------------------测试模型-----------------------
model.eval()  # 切换为评估模式
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = 100 * correct / total

    print(f"模型在测试集上的准确率: {accuracy:.2f}%")

# ----------------------保存测试结果-----------------------
true_labels = y_test_tensor.cpu().numpy()
predicted_labels = predicted.cpu().numpy()

# 计算欧几里得误差
true_coords = np.array([label_to_coords[label] for label in true_labels])
pred_coords = np.array([label_to_coords[label] for label in predicted_labels])
mean_distance = np.mean(np.linalg.norm(true_coords - pred_coords, axis=1))

# 结果保存到 Excel
results_df = pd.DataFrame({
    "True Label": true_labels,
    "Predicted Label": predicted_labels,
    "True X": true_coords[:, 0],
    "True Y": true_coords[:, 1],
    "Pred X": pred_coords[:, 0],
    "Pred Y": pred_coords[:, 1],
    "Error Distance": np.linalg.norm(true_coords - pred_coords, axis=1)
})
results_file = os.path.join(folder_name, f"test_results_{current_time}.xlsx")
results_df.to_excel(results_file, index=False)

print(f"测试结果已保存到 {results_file}")
print(f"平均误差距离: {mean_distance:.2f} m")
