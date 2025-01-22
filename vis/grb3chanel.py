import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为 RGB
image = cv2.imread(r"C:\Users\79152\Desktop\3rdtopic\StableD\uribo.jpg")  # 读取图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 提取各通道
red_channel = image_rgb.copy()
red_channel[:, :, [1, 2]] = 0  # 保留红色通道，其他通道置为0

green_channel = image_rgb.copy()
green_channel[:, :, [0, 2]] = 0  # 保留绿色通道，其他通道置为0

blue_channel = image_rgb.copy()
blue_channel[:, :, [0, 1]] = 0  # 保留蓝色通道，其他通道置为0

# 显示原图及各通道
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Red Channel")
plt.imshow(red_channel)
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Green Channel")
plt.imshow(green_channel)
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Blue Channel")
plt.imshow(blue_channel)
plt.axis('off')

plt.show()
