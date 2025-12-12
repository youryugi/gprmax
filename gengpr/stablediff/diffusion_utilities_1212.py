import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob




class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            if self.same_channels:
                out = x + x1
            else:
                out = x1
            return out / 1.414
        else:
            return self.conv1(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        )

    def forward(self, x, skip):
        x = self.layers[0](x)
        x = torch.cat((x, skip), 1)
        x = self.layers[1](x)
        return x


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            ),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        skip = self.layers[0](x)
        x = self.layers[1](skip)
        return skip, x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.layers(x)


# TODO: 反归一化 RSSI 数据，使其从 [-1,1] 变回 [-100,-20]
def denormalize_rssi(x_norm):
    return x_norm * 40 - 60




# 新增：专门用于读取 B-scan 图片的 Dataset
class BScanImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图片所在的文件夹路径
        transform: 预处理
        文件名格式示例: bscan_0010_v6.png
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png"))) # 假设是png，如果是jpg请修改
        self.transform = transform
        
        # 预先解析所有标签，方便后续 Subset 使用
        self.labels = []
        for path in self.img_paths:
            filename = os.path.basename(path)
            # 解析文件名: bscan_0010_v6.png -> parts=['bscan', '0010', 'v6.png']
            parts = filename.split('_')
            label_str = parts[1] # 获取 '0010'
            # 将字符串 '0010' 转为列表 [0, 0, 1, 0]
            label_vec = [int(c) for c in label_str]
            self.labels.append(label_vec)
        
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 打开图片并转为灰度 (L模式)
        image = Image.open(img_path).convert("L")
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label).float()
