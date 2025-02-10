import numpy as np
import os
from datetime import datetime

np.random.seed(42)

# =============================
# 1. 定义基础参数与工具函数
# =============================

# 定义空间大小 (米)
space_size_x = 16
space_size_y = 16

# 四个路由器坐标 (AP)
# 1是左下, 2是左上, 3是右下, 4是右上
routers = [
    (1, 1),
    (1, space_size_y - 1),
    (space_size_x - 1, 1),
    (space_size_x - 1, space_size_y - 1)
]

# 穿墙损耗与人体损耗
wall_loss = 3   # (dB)
person_loss = 6 # (dB)

# 定义人的影响范围（半径）
person_effect_radius = 0.3

# 四个门（墙）的位置
walls_base = [
    [(space_size_x / 2, 0), (space_size_x / 2, space_size_y / 2)],  # 门1（左下）
    [(space_size_x / 2, space_size_y / 2), (space_size_x / 2, space_size_y)],  # 门2（左上）
    [(0, space_size_y / 2), (space_size_x / 2, space_size_y / 2)],  # 门3（右上）
    [(space_size_x / 2, space_size_y / 2), (space_size_x, space_size_y / 2)],  # 门4（右下）
]

# 生成 16 种门的状态
walls_layouts = []
layout_labels_4bits = []

for i in range(16):  # 0000 ~ 1111 共 16 种
    bits = [(i >> b) & 1 for b in range(4)]  # 解析成二进制列表
    bits.reverse()  # 让最左边是最高位
    layout_labels_4bits.append(bits)

    # 生成该状态下的墙壁（如果 bit == 0 代表墙存在）
    walls = [walls_base[j] for j in range(4) if bits[j] == 0]
    walls_layouts.append(walls)

layout_labels_4bits = np.array(layout_labels_4bits)

# 线段相交判断
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

# 线段-圆相交判断
def line_circle_intersect(x1, y1, x2, y2, cx, cy, radius):
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - radius**2
    discriminant = b**2 - 4*a*c

    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
    return False

# 计算穿墙次数
def walls_crossed(x, y, rx, ry, walls):
    crossed = 0
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        if lines_intersect(rx, ry, x, y, x1, y1, x2, y2):
            crossed += 1
    return crossed

# RSSI 计算
def signal_strength(x, y, rx, ry, walls, path_loss=2, noise_level=2):
    distance = np.sqrt((x - rx)**2 + (y - ry)**2)
    if distance == 0:
        distance = 0.1  # 防止 log(0) 问题
    crossed = walls_crossed(x, y, rx, ry, walls)
    noise = np.random.normal(0, noise_level)
    rssi = -30 - 10 * path_loss * np.log10(distance) - wall_loss * crossed + noise
    return round(rssi)

# =============================
# 2. 定义网格并生成数据
# =============================
x_min, x_max = 1.5, 15  # 维度 = 28
y_min, y_max = 1.5, 15  # 维度 = 28
# x_min, x_max = 1.5, 9  # 维度是16
# y_min, y_max = 1.5, 9  # 维度是16
interval = 0.5

x_coords = np.arange(x_min, x_max + interval, interval)
y_coords = np.arange(y_min, y_max + interval, interval)
num_x = len(x_coords)
num_y = len(y_coords)

# 每种布局，采集 num_samples 张“图像”
num_samples = 500

# 存储数据
X_list = []
Y_list = []

# 生成输出文件夹
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_folder = f'./generateRSSI_{current_time}'
os.makedirs(output_folder, exist_ok=True)

for layout_idx, walls in enumerate(walls_layouts):
    layout_label_4d = layout_labels_4bits[layout_idx]

    for sample_idx in range(num_samples):
        image_rssi = np.zeros((num_y, num_x, 4), dtype=np.float32)

        for iy, vy in enumerate(y_coords):
            for ix, vx in enumerate(x_coords):
                for ap_i, (rx, ry) in enumerate(routers):
                    rssi = signal_strength(vx, vy, rx, ry, walls)
                    image_rssi[iy, ix, ap_i] = rssi

        X_list.append(image_rssi)
        Y_list.append(layout_label_4d)

X_data = np.array(X_list, dtype=np.float32)
Y_data = np.array(Y_list, dtype=np.int32)

print("X_data shape:", X_data.shape)
print("Y_data shape:", Y_data.shape)

# 保存到 .npy 文件
x_save_path = os.path.join(output_folder, 'X_data.npy')
y_save_path = os.path.join(output_folder, 'Y_data.npy')
np.save(x_save_path, X_data)
np.save(y_save_path, Y_data)

print(f"数据已保存：\n  {x_save_path}\n  {y_save_path}")
