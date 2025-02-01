import numpy as np
import os
from datetime import datetime

# =============================
# 1. 定义基础参数与工具函数
# =============================

np.random.seed(42)

# 定义空间大小 (米)
space_size_x = 10.35
space_size_y = 14.8

# 四个路由器坐标 (AP)
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

# 下面的 16 种墙壁布局按照原始示例给出
walls_layouts = [
    # 1（全开）
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ]

]

# 假设用 4 位二进制表示布局，从 [0,0,0,0] ~ [1,1,1,1]
layout_labels_4bits = []
for i in range(16):
    # i 的二进制表示，保持 4 位
    bits = [(i >> b) & 1 for b in range(4)]
    bits.reverse()  # 让最左边是最高位
    layout_labels_4bits.append(bits)
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

# 计算该段路由->测点路径穿过多少面墙 (可选是否考虑人)
def walls_crossed(x, y, rx, ry, walls, person_position=None, consider_person=False):
    crossed = 0
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        if lines_intersect(rx, ry, x, y, x1, y1, x2, y2):
            crossed += 1

    # 如果考虑人体，判断是否穿过圆形区域
    if consider_person and person_position is not None:
        px, py = person_position
        if line_circle_intersect(rx, ry, x, y, px, py, person_effect_radius):
            # 增加相当于一次“穿墙”的额外损耗
            crossed += person_loss / wall_loss
    return crossed

# RSSI 计算模型 (只计算路由器->测点，不计算路由器间)
def signal_strength(x, y, rx, ry, walls, path_loss=2, noise_level=2,
                    person_position=None, consider_person=False):
    distance = np.sqrt((x - rx)**2 + (y - ry)**2)
    if distance == 0:
        distance = 0.1  # 防止距离为0造成 log(0) 问题
    crossed = walls_crossed(x, y, rx, ry, walls, person_position, consider_person)
    noise = np.random.normal(0, noise_level)
    # 简单模型：初始 -30 dBm，路径损耗 10*path_loss*log10(distance)，每穿一堵墙损耗 wall_loss
    rssi = -30 - 10 * path_loss * np.log10(distance) - wall_loss * crossed + noise
    return round(rssi)

# =============================
# 2. 定义网格并生成数据
# =============================

# 构建 (x,y) 网格
#x_min, x_max = 1.5, 9.5 #维度是17
#y_min, y_max = 1.5, 9.5 #维度是17
#y_min, y_max = 2, 12.5 #维度是22
x_min, x_max = 1.5, 9 #维度是16
y_min, y_max = 1.5, 9 #维度是16
interval = 0.5

x_coords = np.arange(x_min, x_max + interval, interval)  # 1.5,2.0,...,9.5
y_coords = np.arange(y_min, y_max + interval, interval)  # 2.0,2.5,...,12.5
num_x = len(x_coords)  # 图像宽度
num_y = len(y_coords)  # 图像高度

# 每种布局，采集 num_samples 张“图像”
num_samples = 500

# 最终结果列表
X_list = []  # 存放 [480, num_y, num_x, 4]
Y_list = []  # 存放 [480, 4]

# 生成输出文件夹
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_folder = f'./generateRSSI_{current_time}'
os.makedirs(output_folder, exist_ok=True)

for layout_idx, walls in enumerate(walls_layouts):
    # 获取该布局对应的 4 维标签
    layout_label_4d = layout_labels_4bits[layout_idx]  # shape = (4,)

    for sample_idx in range(num_samples):
        # 构建一张“图像”，大小 [num_y, num_x, 4]
        image_rssi = np.zeros((num_y, num_x, 4), dtype=np.float32)

        # 遍历网格点 (先行后列 => (iy, ix) => (y, x))
        for iy, vy in enumerate(y_coords):
            for ix, vx in enumerate(x_coords):
                # 4 个 AP 分别计算 RSSI
                for ap_i, (rx, ry) in enumerate(routers):
                    # 这里不考虑人体对测点的影响 => consider_person=False
                    rssi = signal_strength(vx, vy, rx, ry, walls, consider_person=False)
                    image_rssi[iy, ix, ap_i] = rssi

        # 记录数据
        X_list.append(image_rssi)
        Y_list.append(layout_label_4d)

# 转成 numpy 数组
X_data = np.array(X_list, dtype=np.float32)  # [480, num_y, num_x, 4]
Y_data = np.array(Y_list, dtype=np.int32)    # [480, 4]

print("X_data shape:", X_data.shape)  # 期望: (480, num_y, num_x, 4)
print("Y_data shape:", Y_data.shape)  # 期望: (480, 4)

# 保存到 .npy 文件
x_save_path = os.path.join(output_folder, 'X_data.npy')
y_save_path = os.path.join(output_folder, 'Y_data.npy')
np.save(x_save_path, X_data)
np.save(y_save_path, Y_data)

print(f"数据已保存：\n  {x_save_path}\n  {y_save_path}")
