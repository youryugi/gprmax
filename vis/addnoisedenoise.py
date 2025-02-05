import cv2
import numpy as np
import os

# 创建存储结果的文件夹
output_dir = "noised_images"
os.makedirs(output_dir, exist_ok=True)

# 读取原始图片（将图片转换为浮点类型）
image_path = r"C:\Users\79152\Pictures\3nd\cat.jpg"  # 替换为你的图片路径
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]

# 定义扩散过程的总步数（较大步数，确保加噪充分）
total_steps = 100

# 设定 alpha_t 逐步衰减到 0.001，保证最后一步几乎全是噪声
alpha_values = np.linspace(0.99, 0.001, total_steps)
beta_values = 1 - alpha_values  # 噪声比例

# 选择 4 个关键步（例如：0%、33%、66%、100% 位置）
save_indices = [0, total_steps // 3, (2 * total_steps) // 3, total_steps - 1]

for i, step in enumerate(save_indices):
    alpha_t = alpha_values[step]
    beta_t = beta_values[step]

    # 生成高斯噪声
    noise = np.random.normal(loc=0, scale=1.0, size=image.shape)

    # 计算加噪后的图像（扩散过程）
    noisy_image = np.sqrt(alpha_t) * image + np.sqrt(beta_t) * noise

    # 还原到 [0, 255] 并转换回 uint8
    noisy_image_uint8 = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

    # 保存加噪后的图片
    output_path = os.path.join(output_dir, f"diffused_step_{i + 1}.jpg")
    cv2.imwrite(output_path, noisy_image_uint8)
    print(f"已保存: {output_path}（步数: {step}）")

print("所有扩散加噪图片已生成并保存！")
