import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为 RGB
image = cv2.imread(r"C:\Users\79152\Downloads\pixel cat.jpg")  # 读取图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 提取各通道
red_channel = image_rgb.copy()
red_channel[:, :, [1, 2]] = 0  # 保留红色通道，其他通道置为0

green_channel = image_rgb.copy()
green_channel[:, :, [0, 2]] = 0  # 保留绿色通道，其他通道置为0

blue_channel = image_rgb.copy()
blue_channel[:, :, [0, 1]] = 0  # 保留蓝色通道，其他通道置为0

# 显示原图及各通道
fig, axes = plt.subplots(1, 4, figsize=(12, 4))  # 调整 figsize，使图像更紧凑

titles = ["Original Image", "Red Channel", "Green Channel", "Blue Channel"]
images = [image_rgb, red_channel, green_channel, blue_channel]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=20)
    ax.axis("off")

# 调整子图间距，减少空白
plt.subplots_adjust(wspace=0.1, hspace=0)

plt.show()
