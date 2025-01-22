import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    # 1 全开
    [
        [(0, 6.3), (1.3, 6.3)],  # 水平墙
        [(2.3, 6.3), (8.65, 6.3)],  # 水平墙
        [(9.6, 6.3), (10.35, 6.3)],  # 水平墙
        [(3.6, 6.3), (3.6, 6.75)],  # 垂直墙
        [(3.6, 7.85), (3.6, 14.8)],  # 垂直墙
        [(7.1, 0), (7.1, 4.9)],  # 垂直墙
        [(7.1, 5.7), (7.1, 6.3)]  # 垂直墙
    ],
    # 2-----1关
    [
        [(0, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 3-----2关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 4-----3关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 5----4关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 6.3)]
    ],
    # 6----12关
    [
        [(0, 6.3),  (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3),  (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 7----13关
    [
        [(0, 6.3),  (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 8----14关
    [
        [(0, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0),  (7.1, 6.3)]
    ],
    # 9----23关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (10.35, 6.3)],
        [(3.6, 6.3),  (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 10----24关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3),  (3.6, 14.8)],
        [(7.1, 0),  (7.1, 6.3)]
    ],
    # 11----34关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 6.3)]
    ],
    # 12----123关
    [
        [(0, 6.3),  (10.35, 6.3)],
        [(3.6, 6.3),  (3.6, 14.8)],
        [(7.1, 0), (7.1, 4.9)],
        [(7.1, 5.7), (7.1, 6.3)]
    ],
    # 13----124关
    [
        [(0, 6.3),  (8.65, 6.3)],
        [(9.6, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 14.8)],
        [(7.1, 0), (7.1, 6.3)]
    ],
    # 14---134关
    [
        [(0, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 6.75)],
        [(3.6, 7.85), (3.6, 14.8)],
        [(7.1, 0), (7.1, 6.3)]
    ],
    # 15---234关
    [
        [(0, 6.3), (1.3, 6.3)],
        [(2.3, 6.3), (10.35, 6.3)],
        [(3.6, 6.3), (3.6, 14.8)],
        [(7.1, 0),  (7.1, 6.3)]
    ],
    # 16----全关
    [
        [(0, 6.3), (10.35, 6.3)],
        [(7.1, 0), (7.1, 6.3)],
        [(3.6, 6.3), (3.6, 14.8)],
    ]
]

# 定义人的影响范围（半径）
person_effect_radius = 0.3

# 判断两条线段是否相交
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))

# 判断信号路径是否与圆柱体相交
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
def signal_strength(x, y, router_x, router_y, walls, path_loss=2, noise_level=2,
                    person_position=None, consider_person=False,
                    previous_rssi=None, stability_threshold=0.8):
    # 如果有 previous_rssi 且随机满足条件，则返回上一次的值（模拟RSSI的稳定性）
    if previous_rssi is not None and np.random.rand() < stability_threshold:
        return previous_rssi

    distance = np.sqrt((x - router_x)**2 + (y - router_y)**2)
    distance = 0.1 if distance == 0 else distance
    crossed_walls = walls_crossed(x, y, router_x, router_y, walls, person_position, consider_person)
    noise = np.random.normal(0, noise_level)
    signal = -30 - 10*path_loss*np.log10(distance) - wall_loss*crossed_walls + noise
    return round(signal)

# 构建参考点 (使用网格)
x_min, x_max = 1.5, 9.5
y_min, y_max = 2, 12.5
interval = 0.5

x_coords = np.arange(x_min, x_max + interval, interval)
y_coords = np.arange(y_min, y_max + interval, interval)
reference_points = [(x, y) for x in x_coords for y in y_coords]

# 给每个参考点定义一个标签
labels = {point: idx for idx, point in enumerate(reference_points)}

# 获取当前时间作为子文件夹名
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_folder = os.path.join('generateRSSI', current_time)
os.makedirs(output_folder, exist_ok=True)

# 遍历所有墙壁布局
import matplotlib.colors as mcolors

for layout_idx, walls_initial in enumerate(walls_layouts):
    results = []
    location_list = []
    # 用于存放 router->router 的 previous_rssi
    previous_rssi_values_router_to_router = [[None for _ in routers] for _ in routers]

    num_samples = 30
    for x, y in reference_points:
        # 存放 AP->测点 RSSI 的前一次值
        previous_rssi_values = [None]*len(routers)

        for _ in range(num_samples):
            rssi_values = []
            # 先计算 AP->测点 的 RSSI
            for i, (rx, ry) in enumerate(routers):
                rssi = signal_strength(x, y, rx, ry, walls_initial,
                                       consider_person=False,
                                       previous_rssi=previous_rssi_values[i])
                rssi_values.append(rssi)
                previous_rssi_values[i] = rssi

            # 再计算 router->router 的 RSSI
            for i, (rx1, ry1) in enumerate(routers):
                for j, (rx2, ry2) in enumerate(routers):
                    if i != j:
                        rssi = signal_strength(rx1, ry1, rx2, ry2, walls_initial,
                                               person_position=(x, y),
                                               consider_person=True,
                                               previous_rssi=previous_rssi_values_router_to_router[i][j])
                        rssi_values.append(rssi)
                        previous_rssi_values_router_to_router[i][j] = rssi

            # 最终结果： [label, Router1RSSI, Router2RSSI, Router3RSSI, Router4RSSI, R1->R2, R1->R3, ...]
            results.append([labels[(x, y)]] + rssi_values)

        location_list.append([labels[(x, y)], x, y])

    # 构建列名，去掉所有空格
    columns = (
        ['Label']
        + [f'Router{i+1}RSSI' for i in range(len(routers))]
        + [f'Router{i+1}toRouter{j+1}RSSI' for i in range(len(routers)) for j in range(len(routers)) if i != j]
    )

    # 建立 DataFrame
    df = pd.DataFrame(results, columns=columns)

    # 删除相同的行（完全相同才会被删）
    df.drop_duplicates(inplace=True)

    # 保存RSSI测量结果
    df.to_excel(os.path.join(output_folder, f'sim-person-close-stable-layout-{layout_idx+1}.xlsx'), index=False)

    # 位置列表
    location_columns = ['Label', 'X', 'Y']
    location_df = pd.DataFrame(location_list, columns=location_columns)
    location_df.to_excel(os.path.join(output_folder, 'location-list.xlsx'), index=False)

    # 对同一个 Label 的多次测量求均值
    grouped_df = df.groupby('Label').mean().reset_index()
    # 合并得到每个参考点最终的平均 RSSI
    merged_df = pd.merge(location_df, grouped_df, on='Label', how='left')

    # -------------------------
    # 生成“每个路由器”的 Radio Map (网格色块)
    # -------------------------
    for i in range(len(routers)):
        router_col = f'Router{i+1}RSSI'
        # 取出当前路由器的 RSSI 列，去掉 NaN
        plot_df = merged_df.dropna(subset=[router_col])

        # 从数据中提取所有唯一 X、Y 并排序
        x_unique = np.sort(plot_df['X'].unique())
        y_unique = np.sort(plot_df['Y'].unique())

        # 构建与 (y_unique, x_unique) 对应的 2D 数组 Z
        Z = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)

        # 把 (X, Y, RSSI) 填进 Z
        for row in plot_df.itertuples(index=False):
            x_val = row.X
            y_val = row.Y
            rssi_val = getattr(row, router_col)  # 或 row._asdict()[router_col]
            xi = np.where(x_unique == x_val)[0][0]
            yi = np.where(y_unique == y_val)[0][0]
            Z[yi, xi] = rssi_val

        # 计算网格边界，让 pcolormesh 的方块对准 (X, Y)
        dx = interval / 2.0
        dy = interval / 2.0

        x_edges = np.concatenate([
            [x_unique[0] - dx],
            0.5*(x_unique[:-1] + x_unique[1:]),
            [x_unique[-1] + dx]
        ])
        y_edges = np.concatenate([
            [y_unique[0] - dy],
            0.5*(y_unique[:-1] + y_unique[1:]),
            [y_unique[-1] + dy]
        ])

        plt.figure(figsize=(8, 6))
        mesh = plt.pcolormesh(x_edges, y_edges, Z, cmap='jet', shading='auto')

        # 绘制墙
        for (x1, y1), (x2, y2) in walls_initial:
            plt.plot([x1, x2], [y1, y2], color='black', linewidth=2)

        rx, ry = routers[i]  # 获取当前路由器的位置
        plt.scatter(rx, ry, marker='*', s=200,
                    edgecolors='k', facecolors='white',
                    linewidths=1.5,
                    label=f'Router{i + 1}')  # 仅标注当前路由器

        plt.xlim(0, space_size_x)
        plt.ylim(0, space_size_y)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f'Radio Map for Router {i+1} (Layout {layout_idx+1})')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, linestyle='--', alpha=0.5)
        cbar = plt.colorbar(mesh)
        cbar.set_label('RSSI (dBm)')
        # 如果想显示所有 Router 的图例，可以取消注释:
        # plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'radio-map-layout-{layout_idx+1}-router-{i+1}.png'))
        plt.close()

    # -------------------------
    # 绘制最终布局图 (路由器+测点)，可选
    # -------------------------
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
    plt.savefig(os.path.join(output_folder, f'layout-{layout_idx+1}.png'))
    plt.close()
