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
            return out
        else:
            return self.conv1(x)


# === 请用这段代码替换原来的 UnetUp 类 ===
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None):
        """
        in_channels: 输入特征图的通道数 (来自下一层)
        out_channels: 输出特征图的通道数
        skip_channels: 跳跃连接特征图的通道数。如果不填，默认等于 out_channels
        """
        super(UnetUp, self).__init__()
        
        # 1. 转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        
        # 2. 计算拼接后的通道数
        if skip_channels is None:
            skip_channels = out_channels
            
        concat_channels = out_channels + skip_channels
        
        # 3. 卷积层处理拼接后的特征
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), 1) # 拼接
        x = self.conv(x)
        return x
# ========================================


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





# 新增：专门用于读取 B-scan 图片的 Dataset
class BScanImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        """
        img_dir: 图片所在的文件夹路径
        transform: 预处理
        文件名格式示例: bscan_0010_v6.png
        """
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png"))) 
        self.transform = transform
        
        self.labels = []
        for path in self.img_paths:
            filename = os.path.basename(path)
            parts = filename.split('_')
            label_str = parts[1] 
            label_vec = [int(c) for c in label_str]
            self.labels.append(label_vec)
        
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # 必须转为灰度 'L'，否则可能是 'RGB' 导致通道数不对
        image = Image.open(img_path).convert("L")
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # 关键点：确保只返回 image 和 label，不要返回 idx 或其他东西
        # label 必须转为 float tensor 才能进入网络计算
        return image, torch.tensor(label).float()
