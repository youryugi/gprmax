from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
import itertools
import os
import time

# 计时开始
start_time = time.time()

# 读取数据
data_file = r"X_data.npy"
label_file = r"Y_data.npy"
data = np.load(data_file)
labels = np.load(label_file)

RSSI_height = data.shape[1]
RSSI_width = data.shape[2]
d_RSSI = data.shape[3]

# 生成所有 16 种标签组合
all_combinations = list(itertools.product([0, 1], repeat=4))

# 其他超参数
timesteps = 500
beta1 = 1e-3
beta2 = 0.05
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64
n_cfeat = labels.shape[1]  # 4
base_save_dir = './results_exclude_one/'

batch_size = 100
n_epoch = 100
lrate = 1e-3

# 计算扩散噪声参数
global b_t, a_t, ab_t
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# 遍历 16 种情况，每次排除一种
for excluded_combo in all_combinations:
    excluded_str = "_".join(map(str, excluded_combo))
    save_dir = os.path.join(base_save_dir, f"excluded_{excluded_str}")
    os.makedirs(save_dir, exist_ok=True)

    # 设定当前训练用的组合（排除一个）
    train_combinations = [combo for combo in all_combinations if combo != excluded_combo]

    # 过滤出训练数据
    all_indices = list(range(len(labels)))
    train_indices = [i for i in all_indices if tuple(labels[i].tolist()) in train_combinations]

    # 训练数据集
    train_dataset = Subset(dataset, train_indices)

    # 初始化网络
    nn_model = ContextUnet(in_channels=d_RSSI, n_feat=n_feat, n_cfeat=n_cfeat, height=RSSI_height).to(device)
    optim = torch.optim.Adam(nn_model.parameters(), lr=lrate)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # 训练
    nn_model.train()
    for ep in range(n_epoch):
        print(f'Excluded: {excluded_str}, Epoch {ep}')

        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
        pbar = tqdm(dataloader, mininterval=2)
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device).float()

            # 随机 mask 掉部分 context
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)

            # 扩散步骤 t
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
            x_pert = ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

            # 预测噪声
            pred_noise = nn_model(x_pert, t / timesteps, c=c)

            # 计算损失
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optim.step()

        # 保存模型
        if ep % 10 == 0 or ep == n_epoch - 1:
            torch.save(nn_model.state_dict(), os.path.join(save_dir, f"context_model_{ep}.pth"))

    # 测试阶段
    nn_model.load_state_dict(torch.load(os.path.join(save_dir, f"context_model_{n_epoch - 1}.pth"),
                                        map_location=device))
    nn_model.eval()
    print(f"Loaded model for excluded: {excluded_str}")

    # 采样 16 组，每种标签 10 个
    expanded_data = np.repeat(all_combinations, 10, axis=0)
    ctx = torch.tensor(expanded_data).float().to(device)

    samples, _ = sample_ddpm_context(ctx.shape[0], ctx)

    # 分别存到不同文件夹
    trained_context_dir = os.path.join(save_dir, "trained_contexts")
    untrained_context_dir = os.path.join(save_dir, "untrained_contexts")
    os.makedirs(trained_context_dir, exist_ok=True)
    os.makedirs(untrained_context_dir, exist_ok=True)

    for i, combo in enumerate(all_combinations):
        combo_samples = samples[i * 10:(i + 1) * 10]
        combo_str = "_".join(map(str, combo))

        out_path = os.path.join(trained_context_dir if combo in train_combinations else untrained_context_dir,
                                f"context_{combo_str}_samples.npy")
        save_samples_as_npy(torch.tensor(combo_samples, dtype=torch.float32, device=device), out_path)

    print(f"Finished training for excluded: {excluded_str}")

# 计时结束
end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
