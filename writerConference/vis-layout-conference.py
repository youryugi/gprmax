import numpy as np
import matplotlib.pyplot as plt

# =========== 1. 基础参数 ===========
space_size_x = 16.5
space_size_y = 16.5
fontsizebig=20
routers = [
    (1, 1),
    (1, space_size_y - 1),
    (space_size_x - 1, 1),
    (space_size_x - 1, space_size_y - 1)
]

walls_base = [
    [(0, space_size_y / 2), (space_size_x / 2, space_size_y / 2)],#左
    [(space_size_x / 2, space_size_y / 2), (space_size_x / 2, space_size_y)],  # 上
    [(space_size_x / 2, space_size_y / 2), (space_size_x, space_size_y / 2)],#右
    [(space_size_x / 2, 0), (space_size_x / 2, space_size_y / 2)],  # 下
]

walls_layouts = []
layout_labels_4bits = []

for i in range(16):
    bits = [(i >> b) & 1 for b in range(4)]
    bits.reverse()
    layout_labels_4bits.append(bits)
    walls = [walls_base[j] for j in range(4) if bits[j] == 1]
    walls_layouts.append(walls)

layout_labels_4bits = np.array(layout_labels_4bits)

selected_layouts = [
    [0, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 1],
    [1, 1, 1, 1]
]

selected_indices = [i for i, bits in enumerate(layout_labels_4bits) if bits.tolist() in selected_layouts]

x_min, x_max = 1.5, 15
y_min, y_max = 1.5, 15
interval = 0.5

x_coords = np.arange(x_min, x_max + interval, interval)
y_coords = np.arange(y_min, y_max + interval, interval)

# =========== 2. 绘制四种布局 ===========
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

for i, idx in enumerate(selected_indices):
    walls = walls_layouts[idx]
    title = f"Layout {layout_labels_4bits[idx].tolist()}"
    ax = axes[i]

    # 画网格点
    X, Y = np.meshgrid(x_coords, y_coords)
    ax.scatter(X, Y, marker='.', color='gray', alpha=0.5, label="RPs")

    # 画路由器
    for j, (rx, ry) in enumerate(routers):
        ax.scatter(rx, ry, marker='s', color='red', s=100, label=f"APs" if j == 0 else None)

    # 画墙体
    for wall in walls:
        (x1, y1), (x2, y2) = wall
        ax.plot([x1, x2], [y1, y2], 'black', linewidth=2)

    ax.set_xlim(0, space_size_x)
    ax.set_ylim(0, space_size_y)
    ax.set_xlabel("X (m)", fontsize=fontsizebig)
    ax.tick_params(axis='both', labelsize=fontsizebig)
    if i == 0:
        ax.set_ylabel("Y (m)", fontsize=fontsizebig)
    ax.set_title(title, fontsize=fontsizebig)
    ax.legend(fontsize=fontsizebig,loc='upper center', bbox_to_anchor=(0.5, 1.04),ncol=2,frameon=False,columnspacing=0.1,handletextpad=0.1)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
