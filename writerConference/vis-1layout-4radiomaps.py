import numpy as np
import matplotlib.pyplot as plt

# =========== 参数设置与辅助函数 ===========
np.random.seed(42)  # 设置随机种子，保证可重复

space_size_x = 10.35
space_size_y = 14.8

routers = [(1, 1),
           (1, space_size_y - 1),
           (space_size_x - 1, 1),
           (space_size_x - 1, space_size_y - 1)]

wall_loss = 3  # 墙的穿透损耗(dB)
person_loss = 6  # 人体损耗系数 (dB)
person_effect_radius = 0.3  # 人体影响范围

# 仅使用第一个布局 (walls_layouts[0])
walls_initial = [
    [(0, 6.3), (1.3, 6.3)],  # 水平墙
    [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
    [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
    [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
    [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
    [(7.1, 0), (7.1, 4.9)],  # 垂直墙
    [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
]


def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """判断两条线段是否相交"""

    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)

    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))


def line_circle_intersect(x1, y1, x2, y2, cx, cy, radius):
    """判断线段是否与圆(人体模型)相交"""
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx ** 2 + dy ** 2
    b = 2 * (fx * dx + fy * dy)
    c = fx ** 2 + fy ** 2 - radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        disc_sqrt = np.sqrt(discriminant)
        t1 = (-b - disc_sqrt) / (2 * a)
        t2 = (-b + disc_sqrt) / (2 * a)
        # 检查交点是否在线段上
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
    return False


def walls_crossed(x, y, router_x, router_y, walls,
                  person_position=None, consider_person=False):
    """计算从路由器到测点信号路径穿过了多少堵墙；可选考虑人体影响"""
    crossed = 0
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        if lines_intersect(router_x, router_y, x, y, x1, y1, x2, y2):
            crossed += 1

    if consider_person and person_position is not None:
        px, py = person_position
        if line_circle_intersect(router_x, router_y, x, y, px, py, person_effect_radius):
            # 把人体损耗折算成“等效墙”数量
            crossed += person_loss / wall_loss
    return crossed


def signal_strength(x, y, router_x, router_y, walls,
                    path_loss=2, noise_level=2,
                    person_position=None, consider_person=False,
                    previous_rssi=None, stability_threshold=0.8):
    """
    计算 RSSI，若给定 previous_rssi 且随机值小于 stability_threshold，则复用 previous_rssi。
    """
    # 稳定性模拟（只是一种示例）
    if previous_rssi is not None and np.random.rand() < stability_threshold:
        return previous_rssi

    distance = np.sqrt((x - router_x) ** 2 + (y - router_y) ** 2)
    # 避免 distance=0
    distance = 0.1 if distance == 0 else distance

    n_walls = walls_crossed(x, y, router_x, router_y, walls,
                            person_position=person_position,
                            consider_person=consider_person)

    noise = np.random.normal(0, noise_level)
    rssi = -30 - 10 * path_loss * np.log10(distance) - wall_loss * n_walls + noise
    return round(rssi)


# =========== 生成测点 ===========

x_min, x_max = 1.5, 9.5
y_min, y_max = 2.0, 12.5
interval = 0.5

x_coords = np.arange(x_min, x_max + interval, interval)
y_coords = np.arange(y_min, y_max + interval, interval)

# 用于快速索引
Nx = len(x_coords)
Ny = len(y_coords)

# 生成网格点 (x, y)
reference_points = [(x, y) for x in x_coords for y in y_coords]

# 给每个参考点一个编号
labels = {point: idx for idx, point in enumerate(reference_points)}

# =========== 多次测量后求平均 RSSI ===========

num_samples = 30

# 准备 4 个路由器的二维矩阵存放 RSSI (Ny, Nx)
router_rssi_grids = [np.zeros((Ny, Nx)) for _ in range(len(routers))]

# 为了模拟稳定 RSSI，可额外记录 previous_rssi
previous_rssi_values = [[None] * len(reference_points) for _ in range(len(routers))]

for idx, (x, y) in enumerate(reference_points):
    # index 转化成 row, col 用于后面 imshow 填值
    # 例如 col = x_coords.index(x), row = y_coords.index(y)
    # 也可用以下更快的办法（由于是均匀间隔）
    col = int(round((x - x_min) / interval))
    row = int(round((y - y_min) / interval))

    # 多次测量
    router_rssi_accum = [[] for _ in range(len(routers))]
    for _ in range(num_samples):
        for i, (rx, ry) in enumerate(routers):
            rssi = signal_strength(x, y, rx, ry, walls_initial,
                                   consider_person=False,
                                   previous_rssi=previous_rssi_values[i][idx])
            router_rssi_accum[i].append(rssi)
            previous_rssi_values[i][idx] = rssi

    # 取平均并写入网格
    for i in range(len(routers)):
        mean_val = np.mean(router_rssi_accum[i])
        router_rssi_grids[i][row, col] = mean_val

# =========== 为绘制“正方形像素”准备：计算全局 RSSI 范围，统一 color scale ===========
all_vals = np.concatenate([arr.flatten() for arr in router_rssi_grids])
global_min, global_max = all_vals.min(), all_vals.max()

# 定义一个归一化器（所有子图共享）
norm = plt.Normalize(vmin=global_min, vmax=global_max)

# =========== 绘图：5 个子图 (1 行) ===========

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))

# --- 子图 1：路由器 & 参考点分布示意 ---
ax0 = axes[0]
rx_coords = [r[0] for r in routers]
ry_coords = [r[1] for r in routers]

# 路由器
ax0.scatter(rx_coords, ry_coords, color='blue', label='APs', s=100)
# 参考点
ref_x = [p[0] for p in reference_points]
ref_y = [p[1] for p in reference_points]
ax0.scatter(ref_x, ref_y, color='red', label='RPs', s=6)

# 墙体
for (x1, y1), (x2, y2) in walls_initial:
    ax0.plot([x1, x2], [y1, y2], color='black', linewidth=2)
# 调整图例，使其为一行

ax0.set_xlim(0, space_size_x)
ax0.set_ylim(0, space_size_y)
# ax0.set_xlabel('X (m)')
# ax0.set_ylabel('Y (m)')
ax0.set_title('Layout \n(APs & RPs)')
ax0.legend()
ax0.grid(False)
ax0.set_aspect('equal', adjustable='box')
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_xlabel('')
ax0.set_ylabel('')

# --- 子图 2 ~ 5：4 个路由器的 Radio Map ---
ims = []
for i in range(len(routers)):
    ax = axes[i + 1]
    # 取出第 i 个路由器的 RSSI 网格
    rssi_grid = router_rssi_grids[i]

    # imshow：正方形像素，可视化
    im = ax.imshow(rssi_grid,
                   origin='lower',
                   extent=[x_min, x_max, y_min, y_max],
                   cmap='jet',
                   norm=norm,
                   interpolation='nearest',
                   aspect='equal')
    ims.append(im)

    # 叠加墙体
    for (x1, y1), (x2, y2) in walls_initial:
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=2)

    # 只在当前 Radio Map 上散点出对应的路由器位置
    rx, ry = routers[i]  # 当前 router 的坐标
    ax.scatter(rx, ry, color='blue',  s=100, marker='o')

    ax.set_xlim(0, space_size_x)
    ax.set_ylim(0, space_size_y)
    ax.set_aspect('equal', adjustable='box')
    #ax.set_xlabel('X (m)')
    # if i == 0:
    #     ax.set_ylabel('Y (m)')
    ax.set_title(f'Radio Map\nAP {i + 1}')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
# 调整图例，使其为一行
ax0.legend(loc='upper center', ncol=2, columnspacing=0.1,handletextpad=0.1)  # 调整列间距

# 只在最后一个 Radio Map 子图放颜色条
# 也可在这一行 4 个 subplot 右侧放一个统一 colorbar
cbar = fig.colorbar(ims[-1], ax=axes[4], orientation='vertical')
cbar.set_label('RSSI (dBm)')

plt.tight_layout()
plt.show()
