import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# 设置随机种子
np.random.seed(42)
# 定义空间大小和路由器位置
space_size_x = 10.35
space_size_y = 14.8
routers = [(1, 1), (1, space_size_y - 1), (space_size_x - 1, 1), (space_size_x - 1, space_size_y - 1)]
wall_loss = 3  # 墙的穿透损耗(dB)
person_loss = 6  # 人体损耗系数 (dB)

# 16组不同的墙壁布局
walls_layouts = [
    # 墙壁布局1全开
    [
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [# 2-----1关
        [(0, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#3-----2关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#4-----3关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#5----4关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 6.3)]  # 垂直墙
    ],
    [
        [(0, 6.3),  (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3),  (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],#layout6----12关
    [#layout7----13关
        [(0, 6.3),  (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#8----14关
        [(0, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0),  (7.1, 6.3)]  # 垂直墙
    ],
    [#9----23关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3),  (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#10----24关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3),  (3.6, 14.8)],  # 垂直墙
        [(7.1, 0),  (7.1, 6.3)]  # 垂直墙
    ],
    [#11----34关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 6.3)]  # 垂直墙
    ],
    [#12----123关
        [(0, 6.3),  (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3),  (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    [#13----124关
        [(0, 6.3),  (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 6.3)]  # 垂直墙
    ],
    [#14---134关
        [(0, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 6.3)]  # 垂直墙
    ],
    [#15---234关
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0),  (7.1, 6.3)]  # 垂直墙
    ],
    [#16----全关
        [(0, 6.3), (10.35, 6.3)],  # 水平墙
        [(7.1, 0), (7.1, 6.3)],  # 垂直墙
        [(3.6, 6.3), (3.6, 14.8)],  # 垂直墙
    ]
    # 其他墙壁布局在这里定义
]

# 定义人的影响范围（半径）
person_effect_radius = 0.3

# 判断两条线段是否相交
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)

    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)

# 判断信号路径是否与圆柱体相交
def line_circle_intersect(x1, y1, x2, y2, cx, cy, radius):
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy

    a = dx ** 2 + dy ** 2
    b = 2 * (fx * dx + fy * dy)
    c = (fx ** 2 + fy ** 2) - radius ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # 检查交点是否在线段上
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
    return False

# 计算信号穿过的墙数，并可选地考虑人的影响
def walls_crossed(x, y, router_x, router_y, walls, person_position=None, consider_person=False):
    crossed = 0
    for wall in walls:
        x1, y1 = wall[0]
        x2, y2 = wall[1]
        if lines_intersect(router_x, router_y, x, y, x1, y1, x2, y2):
            crossed += 1

    if consider_person and person_position is not None:
        person_x, person_y = person_position
        if line_circle_intersect(router_x, router_y, x, y, person_x, person_y, person_effect_radius):
            crossed += person_loss / wall_loss  # 增加类似墙的损耗
    return crossed

# 定义信号强度模型函数（考虑墙的影响和人的影响）
def signal_strength(x, y, router_x, router_y, walls, path_loss=2, noise_level=2, person_position=None,
                    consider_person=False, previous_rssi=None, stability_threshold=0.8):
    if previous_rssi is not None and np.random.rand() < stability_threshold:
        return previous_rssi

    distance = np.sqrt((x - router_x) ** 2 + (y - router_y) ** 2)
    distance = 0.1 if distance == 0 else distance
    crossed_walls = walls_crossed(x, y, router_x, router_y, walls, person_position, consider_person)
    noise = np.random.normal(0, noise_level)
    signal = -30 - 10 * path_loss * np.log10(distance) - wall_loss * crossed_walls + noise
    return round(signal)

# 自定义参考点
# reference_points = [
#     (1.5, 2), (3.5, 2), (5.5, 2), (7.5, 2.5), (9.5, 2.5),
#     (1.5, 4), (3.5, 4), (5.5, 4), (7.5, 4.5), (9.5, 4.5),
#     (1.5, 6), (3.5, 6), (5.5, 6),
#     (1.5, 8.5), (5.5, 8.5), (7.5, 8.5), (9.5, 8.5),
#     (1.5, 10.5), (5.5, 10.5), (7.5, 10.5), (9.5, 10.5),
#     (1.5, 12.5), (5.5, 12.5), (7.5, 12.5), (9.5, 12.5)
# ]

x_min, x_max = 1.5, 9.5
y_min, y_max = 2, 12.5
interval = 0.5

# Generate the grid points
x_coords = np.arange(x_min, x_max + interval, interval)
y_coords = np.arange(y_min, y_max + interval, interval)
reference_points = [(x, y) for x in x_coords for y in y_coords]

labels = {point: idx for idx, point in enumerate(reference_points)}
os.makedirs('generateRSSI', exist_ok=True)
# 遍历所有墙壁布局
for layout_idx, walls_initial in enumerate(walls_layouts):
    results = []
    location_list = []
    previous_rssi_values_router_to_router = [[None for _ in routers] for _ in routers]

    num_samples = 30
    for x, y in reference_points:
        previous_rssi_values = [None] * len(routers)
        for _ in range(num_samples):
            rssi_values = []
            for i, (rx, ry) in enumerate(routers):
                rssi = signal_strength(x, y, rx, ry, walls_initial, consider_person=False,
                                       previous_rssi=previous_rssi_values[i])
                rssi_values.append(rssi)
                previous_rssi_values[i] = rssi

            for i, (rx1, ry1) in enumerate(routers):
                for j, (rx2, ry2) in enumerate(routers):
                    if i != j:
                        rssi = signal_strength(rx1, ry1, rx2, ry2, walls_initial, person_position=(x, y),
                                               consider_person=True,
                                               previous_rssi=previous_rssi_values_router_to_router[i][j])
                        rssi_values.append(rssi)
                        previous_rssi_values_router_to_router[i][j] = rssi
            results.append([labels[(x, y)]] + rssi_values)
        location_list.append([labels[(x, y)], x, y])

    columns = ['Label'] + [f'Router {i + 1} RSSI' for i in range(len(routers))] + \
              [f'Router {i + 1} to Router {j + 1} RSSI' for i in range(len(routers)) for j in range(len(routers)) if i != j]
    df = pd.DataFrame(results, columns=columns)

    # 删除相同的行
    df.drop_duplicates(inplace=True)

    # 保存RSSI测量结果为Excel文件

    df.to_excel(f'generateRSSI/sim-person-close-stable-layout-{layout_idx + 1}.xlsx', index=False)

    # 保存位置列表为Excel文件
    location_columns = ['Label', 'X', 'Y']
    location_df = pd.DataFrame(location_list, columns=location_columns)
    location_df.to_excel(f'location-list.xlsx', index=False)

    # 绘制并保存图像
    plt.figure(figsize=(10, 10))
    plt.scatter(*zip(*routers), color='blue', label='Routers', s=100)
    plt.scatter(*zip(*reference_points), color='red', label='Reference Points', s=10)
    for (x1, y1), (x2, y2) in walls_initial:
        plt.plot([x1, x2], [y1, y2], color='black', linewidth=2)
    plt.xlim(0, space_size_x)
    plt.ylim(0, space_size_y)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Router and Reference Point Positions (Layout {layout_idx + 1})')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f'layout-{layout_idx + 1}.png')
    plt.close()
