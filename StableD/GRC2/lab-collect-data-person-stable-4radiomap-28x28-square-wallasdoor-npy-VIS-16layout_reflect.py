import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 为保证可复现，固定随机种子
np.random.seed(42)

# =============================
# 1. 场景参数 & 全局设置
# =============================
# 房间尺寸 (米)
space_size_x = 16.5
space_size_y = 16.5
space_size_z = 3.0  # 高度 3m

# AP / 用户的高度（z=1m）
AP_height = 1.0
User_height = 1.0

# 四个路由器 3D 坐标（原先在 2D 角落，这里补上 z=1）
# 1(左下), 2(左上), 3(右下), 4(右上)
routers_3d = [
    (1.0, 1.0, AP_height),
    (1.0, space_size_y - 1, AP_height),
    (space_size_x - 1, 1.0, AP_height),
    (space_size_x - 1, space_size_y - 1, AP_height)
]

# 穿墙损耗和人体损耗（如果依然需要，可保留）
wall_loss_db = 15
person_loss_db = 6  # 若后面需要考虑“人体阻挡”也可写类似的逻辑

# 定义人的影响范围半径（如果需要考虑人体遮挡可以用，未使用则可注释）
person_effect_radius = 0.3

# 四个“门/隔墙”的 2D 坐标（原先的）
# 这里的每条线段都被视为在 z=0~z=3 的竖直平面
walls_base = [
    [(space_size_x / 2, 0), (space_size_x / 2, space_size_y / 2)],  # 门1（下半段）
    [(space_size_x / 2, space_size_y / 2), (space_size_x / 2, space_size_y)],  # 门2（上半段）
    [(0, space_size_y / 2), (space_size_x / 2, space_size_y / 2)],  # 门3（左半段）
    [(space_size_x / 2, space_size_y / 2), (space_size_x, space_size_y / 2)]  # 门4（右半段）
]

# 16 种门(墙)的开关组合
walls_layouts = []
layout_labels_4bits = []

for i in range(16):  # i=0..15
    # 解析成4位二进制 [bit0, bit1, bit2, bit3]
    bits = [(i >> b) & 1 for b in range(4)]
    bits.reverse()  # 让 bits[0] 对应最高位
    layout_labels_4bits.append(bits)

    # 若 bit=0 => 墙存在； bit=1 => 墙“打开”(无障碍)
    # 因此只把 bit=0 的那几条线段添加到 walls
    walls_2d = [walls_base[j] for j in range(4) if bits[j] == 1]
    walls_layouts.append(walls_2d)

layout_labels_4bits = np.array(layout_labels_4bits, dtype=np.int32)


# =============================
# 2. 一些旧版的 2D 函数保留 (线段相交等)
# =============================

# 线段相交判断(2D)，用于可视化或其他用途
def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)

    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))


# 线段-圆相交判断 (保留，但目前未使用)
def line_circle_intersect(x1, y1, x2, y2, cx, cy, radius):
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx ** 2 + dy ** 2
    b = 2 * (fx * dx + fy * dy)
    c = fx ** 2 + fy ** 2 - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant >= 0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
    return False


# =============================
# 3. 外墙镜像法 (3D 多次反射)
# =============================
# 反射系数(幅度衰减) => 若=0.7, 则一次反射功率系数=0.7^2=0.49
reflection_coef = 0.7


def mirror_x_min(x, y, z):
    # x=0 面
    return -x, y, z


def mirror_x_max(x, y, z):
    # x=space_size_x 面
    return 2 * space_size_x - x, y, z


def mirror_y_min(x, y, z):
    # y=0 面
    return x, -y, z


def mirror_y_max(x, y, z):
    # y=space_size_y 面
    return x, 2 * space_size_y - y, z


def mirror_z_min(x, y, z):
    # z=0 地面
    return x, y, -z


def mirror_z_max(x, y, z):
    # z=space_size_z 天花板
    return x, y, 2 * space_size_z - z


BOUNDARIES_3D = [
    mirror_x_min,
    mirror_x_max,
    mirror_y_min,
    mirror_y_max,
    mirror_z_min,
    mirror_z_max
]


# =============================
# 4. 内部墙穿透统计 (3D)
# =============================
def walls_crossed_3d(x, y, z, rx, ry, rz, walls_2d):
    """
    计算从 (rx, ry, rz) -> (x, y, z) 这条射线穿过了多少面“内部墙”。
    walls_2d 里每条线段[(x1,y1),(x2,y2)]视为 z=0~3 的竖直平面。
    简化：假设墙都是 x1==x2 或 y1==y2 (轴对齐)。
    """
    crossed_count = 0
    dx = x - rx
    dy = y - ry
    dz = z - rz

    for (x1, y1), (x2, y2) in walls_2d:
        # 判断竖直墙 or 水平墙
        if abs(x2 - x1) < 1e-9:
            # 竖直墙 => x=x1, y∈[y_min,y_max], z∈[0,3]
            wall_x = x1
            y_min_, y_max_ = sorted([y1, y2])
            if abs(dx) < 1e-9:
                continue
            t = (wall_x - rx) / dx
            if 0 <= t <= 1:
                y_int = ry + t * dy
                z_int = rz + t * dz
                if (y_min_ <= y_int <= y_max_) and (0 <= z_int <= space_size_z):
                    crossed_count += 1
        elif abs(y2 - y1) < 1e-9:
            # 水平墙 => y=y1, x∈[x_min,x_max], z∈[0,3]
            wall_y = y1
            x_min_, x_max_ = sorted([x1, x2])
            if abs(dy) < 1e-9:
                continue
            t = (wall_y - ry) / dy
            if 0 <= t <= 1:
                x_int = rx + t * dx
                z_int = rz + t * dz
                if (x_min_ <= x_int <= x_max_) and (0 <= z_int <= space_size_z):
                    crossed_count += 1
        else:
            # 若有斜墙，需要更一般的射线-矩形相交算法
            pass

    return crossed_count


# =============================
# 5. RSSI 计算 (3D 多次反射 + 内墙)
# =============================
def signal_strength_3d_with_reflections(x, y, z,
                                        rx, ry, rz,
                                        walls_2d,
                                        max_reflections=2,
                                        path_loss=2,
                                        noise_level=2):
    """
    计算 3D 场景下，从 AP(rx,ry,rz) 到 用户(x,y,z) 的 RSSI(dB)。
    考虑：
      - 直达路径 + 最多 max_reflections 次外墙(6面)镜像
      - 内部墙(竖直平面)的穿墙损耗
      - 非相干功率线性叠加
    """
    # 1) 直达路径
    dist_direct = np.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)
    if dist_direct == 0:
        dist_direct = 0.1
    crossed_direct = walls_crossed_3d(x, y, z, rx, ry, rz, walls_2d)
    pl_db_direct = 10 * path_loss * np.log10(dist_direct)
    direct_rssi_db = -30 - pl_db_direct - wall_loss_db * crossed_direct
    power_lin_direct = 10 ** (direct_rssi_db / 10.0)

    path_powers = [power_lin_direct]

    # 2) 一次反射
    if max_reflections >= 1:
        for mirror_func1 in BOUNDARIES_3D:
            rx1, ry1, rz1 = mirror_func1(rx, ry, rz)
            dist1 = np.sqrt((x - rx1) ** 2 + (y - ry1) ** 2 + (z - rz1) ** 2)
            if dist1 == 0:
                dist1 = 0.1
            crossed_1 = walls_crossed_3d(x, y, z, rx1, ry1, rz1, walls_2d)
            pl_db_1 = 10 * path_loss * np.log10(dist1)
            rssi_db_1 = -30 - pl_db_1 - wall_loss_db * crossed_1
            # 一次反射 => 幅度衰减 reflection_coef => 功率衰减 (reflection_coef^2)
            ref_factor_1 = reflection_coef ** 2
            power_lin_1 = 10 ** (rssi_db_1 / 10.0) * ref_factor_1
            path_powers.append(power_lin_1)

    # 3) 二次反射
    if max_reflections >= 2:
        for mirror_func1 in BOUNDARIES_3D:
            rx1, ry1, rz1 = mirror_func1(rx, ry, rz)
            for mirror_func2 in BOUNDARIES_3D:
                rx2, ry2, rz2 = mirror_func2(rx1, ry1, rz1)
                dist2 = np.sqrt((x - rx2) ** 2 + (y - ry2) ** 2 + (z - rz2) ** 2)
                if dist2 == 0:
                    dist2 = 0.1
                crossed_2 = walls_crossed_3d(x, y, z, rx2, ry2, rz2, walls_2d)
                pl_db_2 = 10 * path_loss * np.log10(dist2)
                rssi_db_2 = -30 - pl_db_2 - wall_loss_db * crossed_2
                # 二次反射 => 幅度衰减 (reflection_coef^2)^2 => reflection_coef^4
                ref_factor_2 = (reflection_coef ** 4)
                power_lin_2 = 10 ** (rssi_db_2 / 10.0) * ref_factor_2
                path_powers.append(power_lin_2)

    # 4) 线性功率求和 => dB => 加随机噪声
    total_power_lin = sum(path_powers)
    total_rssi_db = 10 * np.log10(total_power_lin)
    # 简单在 dB 上加一个高斯噪声(若要更精确可以先加噪声功率再转dB)
    noise = np.random.normal(0, noise_level)
    total_rssi_db += noise

    return round(total_rssi_db)


# =============================
# 6. 生成数据 & 可视化 & 保存
# =============================

# 用于可视化“布局示意图”（2D 顶视）
def visualize_layout(walls_2d, routers_2d, x_coords, y_coords, title, save_path):
    """
    仅在 2D 顶视图上展示：网格点(灰)、路由器(红)、内部墙(黑线)。
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 画出网格点
    X, Y = np.meshgrid(x_coords, y_coords)
    ax.scatter(X, Y, marker='.', color='gray', alpha=0.5, label="Reference Points")

    # 画出路由器位置(2D)
    # 注意：routers_2d 只取 (x,y) 做示意
    for i, (rx, ry) in enumerate(routers_2d):
        # 用方块表示AP
        ax.scatter(rx, ry, marker='s', color='red', s=100,
                   label=("Router" if i == 0 else None))

    # 画出墙壁
    for idx, wall_seg in enumerate(walls_2d):
        (x1, y1), (x2, y2) = wall_seg
        if idx == 0:
            ax.plot([x1, x2], [y1, y2], 'black', linewidth=2, label="Walls")
        else:
            ax.plot([x1, x2], [y1, y2], 'black', linewidth=2)

    ax.set_xlim(0, space_size_x)
    ax.set_ylim(0, space_size_y)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    # 网格范围 (仅 x,y 方向)
    x_min, x_max = 1.5, 15.0
    y_min, y_max = 1.5, 15.0
    interval = 0.5

    x_coords = np.arange(x_min, x_max + interval, interval)
    y_coords = np.arange(y_min, y_max + interval, interval)
    num_x = len(x_coords)
    num_y = len(y_coords)

    # 每种布局，采集 num_samples 组“RSSI图”
    num_samples = 500

    X_list = []
    Y_list = []

    # 输出文件夹
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f'./generateRSSI_3D_{current_time}'
    os.makedirs(output_folder, exist_ok=True)

    # 文件夹：保存 4 个路由器 RSSI 的热力图
    rssi_map_folder = os.path.join(output_folder, "RSSI_Maps")
    os.makedirs(rssi_map_folder, exist_ok=True)

    # ========== 生成并采集数据 ==========
    for layout_idx, walls_2d in enumerate(walls_layouts):
        layout_label_4d = layout_labels_4bits[layout_idx]

        # 为该布局单独创建子文件夹
        layout_rssi_folder = os.path.join(rssi_map_folder, f"layout_{layout_idx}")
        os.makedirs(layout_rssi_folder, exist_ok=True)

        for sample_idx in range(num_samples):
            # RSSI 图像 (num_y, num_x, 4) => 4个AP通道
            image_rssi = np.zeros((num_y, num_x, 4), dtype=np.float32)

            for iy, vy in enumerate(y_coords):
                for ix, vx in enumerate(x_coords):
                    # 用户 3D 坐标 (z=1)
                    ux, uy, uz = vx, vy, User_height

                    # 计算4个AP的RSSI
                    for ap_i, (arx, ary, arz) in enumerate(routers_3d):
                        rssi_val = signal_strength_3d_with_reflections(
                            ux, uy, uz,
                            arx, ary, arz,
                            walls_2d=walls_2d,
                            max_reflections=2,  # 最多2次反射
                            path_loss=2,
                            noise_level=2
                        )
                        image_rssi[iy, ix, ap_i] = rssi_val

            # 第一次 sample 才保存 4 通道热力图
            if sample_idx == 0:
                for ap_i in range(4):
                    plt.figure()
                    plt.imshow(
                        image_rssi[:, :, ap_i],
                        cmap='jet',
                        origin='lower',
                        extent=(x_min, x_max, y_min, y_max)
                    )
                    plt.colorbar(label='RSSI (dB)')
                    plt.title(f"Layout {layout_idx} - Router {ap_i + 1}")
                    save_map_path = os.path.join(
                        layout_rssi_folder, f"layout_{layout_idx}_router_{ap_i + 1}.png"
                    )
                    plt.savefig(save_map_path)
                    plt.close()

            # 记录 X / Y
            X_list.append(image_rssi)
            Y_list.append(layout_label_4d)

    # 转为 numpy 数组
    X_data = np.array(X_list, dtype=np.float32)
    Y_data = np.array(Y_list, dtype=np.int32)

    print("X_data shape:", X_data.shape)
    print("Y_data shape:", Y_data.shape)

    # 保存 npy
    x_save_path = os.path.join(output_folder, 'X_data.npy')
    y_save_path = os.path.join(output_folder, 'Y_data.npy')
    np.save(x_save_path, X_data)
    np.save(y_save_path, Y_data)

    print(f"数据已保存：\n  {x_save_path}\n  {y_save_path}")

    # ========== 可视化布局并保存 ==========

    # 为保存布局示意图创建文件夹
    layout_folder = os.path.join(output_folder, "Layouts")
    os.makedirs(layout_folder, exist_ok=True)

    # 因为示意图只需要 2D 顶视，所以 routers_2d 取其 (x,y)
    routers_2d = [(r[0], r[1]) for r in routers_3d]

    for idx, walls_2d in enumerate(walls_layouts):
        title = f"RSSI Layout - Config {idx}"
        save_path = os.path.join(layout_folder, f"layout_{idx}.png")
        visualize_layout(walls_2d, routers_2d, x_coords, y_coords, title, save_path)

    print(f"所有布局示意图已保存至 {layout_folder} 文件夹。")
    print(f"每种布局对应的 4 个路由器 RSSI 图已保存至 {rssi_map_folder} 文件夹。")


if __name__ == "__main__":
    main()
