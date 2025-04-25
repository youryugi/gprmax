import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd
fontsizebig=20

#zhiqi之前的数据的1111的标记是反过来的
# =========== 1. 预定义布局参数 ===========

space_size_x = 16.5
space_size_y = 16.5

routers = [
    (1, 1),
    (1, space_size_y - 1),
    (space_size_x - 1, 1),
    (space_size_x - 1, space_size_y - 1)
]

# 四个门（墙）的位置
walls_base = [
    [(space_size_x / 2, 0), (space_size_x / 2, space_size_y / 2)],  # 门1（左下）
    [(space_size_x / 2, space_size_y / 2), (space_size_x / 2, space_size_y)],  # 门2（左上）
    [(0, space_size_y / 2), (space_size_x / 2, space_size_y / 2)],  # 门3（右上）
    [(space_size_x / 2, space_size_y / 2), (space_size_x, space_size_y / 2)],  # 门4（右下）
]


# 生成16种 layout
walls_layouts = []
layout_labels_4bits = []

for i in range(16):  # 0000 ~ 1111 共 16 种
    bits = [(i >> b) & 1 for b in range(4)]  # 解析成二进制列表
    bits.reverse()  # 让最左边是最高位
    layout_labels_4bits.append(bits)

    # 生成该状态下的墙壁（如果 bit == 1代表墙存在）
    walls = [walls_base[j] for j in range(4) if bits[j] == 1]
    walls_layouts.append(walls)

layout_labels_4bits = np.array(layout_labels_4bits)


# =========== 2. 从文件名解析 Layout ===========

def extract_layout_from_filename(filename):
    """
    假设文件名里含 `_0_0_0_0` 这种格式，
    用正则提取四个数字，并返回 [0,0,0,0].
    若没有匹配则返回 None.
    """
    match = re.search(r'(\d)_(\d)_(\d)_(\d)', filename)
    print("match=",match)
    if match:
        return [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]

    return None


# =========== 3. 指定文件 & 解析布局 ===========

# 示例文件，可换成你自己的路径
file1 = r"C:\Users\79152\Desktop\3rdtopic\writerConference\data1111\0213_good_mean_0_0_0_1.npy"
file2 = r"C:\Users\79152\Desktop\3rdtopic\StableD\GRC2\generateRSSI_2025-04-24_12-49-23-redo\weights_16_experiments\excluded_1_0_1_0\untrained\context_1_0_1_0\samples.npy"

layout_from_file = extract_layout_from_filename(file1)
if layout_from_file is None:
    raise ValueError(f"无法从文件名解析 Layout: {file1}")

# 在16种布局中查找对应 index
selected_idx = None
for i, bits in enumerate(layout_labels_4bits):
    if bits.tolist() == layout_from_file:
        selected_idx = i
        break

if selected_idx is None:
    raise ValueError(f"在16种预定义布局中，未找到 {layout_from_file}！")

walls_layout_selected = walls_layouts[selected_idx]


# =========== 4. 加载 RSSI 数据 & 计算误差 ===========

data1 = np.load(file1)  # (1,28,28,4)
data2 = np.load(file2)  # (1,28,28,4)
# 对第二个文件做一些预处理，比如多个样本要取平均
data2 = np.mean(data2, axis=0, keepdims=True)  # => (1,28,28,4)

if data1.shape != (1, 28, 28, 4) or data2.shape != (1, 28, 28, 4):
    raise ValueError("数据形状不匹配，需 (1,28,28,4).")

channel_index = 0  # 选择第4个路由器通道 (0-based)
gt_rssi = data1[0, :, :, channel_index]  # (28,28)
print("data1",gt_rssi)
gen_rssi = data2[0, :, :, channel_index]
print("data2",gen_rssi)

# 创建 DataFrame
df_gt = pd.DataFrame(gt_rssi)
df_gen = pd.DataFrame(gen_rssi)

# 保存为 Excel 文件
with pd.ExcelWriter('rssi_output.xlsx', engine='openpyxl') as writer:
    df_gt.to_excel(writer, sheet_name='GT_RSSI', index=False, header=False)
    df_gen.to_excel(writer, sheet_name='Gen_RSSI', index=False, header=False)

print("保存成功：rssi_output.xlsx")
# 计算误差（百分比）
error_pct = np.abs(gt_rssi - gen_rssi) / (np.abs(gt_rssi) + 1e-6) * 100
accuracy_map = (error_pct <= 10).astype(float)  # True=1,False=0

# 准确率
accuracy_ratio = accuracy_map.mean()  # (True 个数) / (总数)


# =========== 5. 可视化 ===========

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6.2))

# RSSI 坐标范围
x_min, x_max = 1.5, 15
y_min, y_max = 1.5, 15
extent = [x_min, x_max, y_min, y_max]

titles = [
    f"Ground Truth (Router {channel_index+1})",
    f"Generated RSSI (Router {channel_index+1})",
    "Accuracy Map "
]
cmap_list = ["viridis", "viridis", "gray"]
data_list = [gt_rssi, gen_rssi, accuracy_map]

# 统一 RSSI 色阶（前两张图）不再需要了
#vmin, vmax = min(gt_rssi.min(), gen_rssi.min()), max(gt_rssi.max(), gen_rssi.max())

for i, ax in enumerate(axes):
    data_show = data_list[i]

    # 对 accuracy map 特殊处理
    import matplotlib.colors as colors  # 确保在开头已导入

    # 设置不等距的颜色区间
    boundaries = [-70,-69,-68,-67,-66,-65,-64,-63,-62,-60,-58,-56,-54,-52,-50,-48,-46, -44,-42,-40,-38,-36,-34,-32,-30]
    norm = colors.BoundaryNorm(boundaries=boundaries, ncolors=256)

    if i < 2:
        im = ax.imshow(data_show,
                       cmap=cmap_list[i],
                       interpolation="nearest",
                       extent=extent,
                       norm=norm)  # 注意这里使用 norm 替代 vmin/vmax

    else:
        # accuracy_map: 0=黑,1=白
        im = ax.imshow(data_show,
                       cmap=cmap_list[i],
                       interpolation="nearest",
                       extent=extent,
                       vmin=0, vmax=1)  # 限定 0~1

    # 叠加墙体
    for (x1, y1), (x2, y2) in walls_layout_selected:
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=2)

    # 叠加路由器
    # 叠加选定的路由器
    selected_router = routers[channel_index+1]  # 只获取当前选定的路由器
    ax.scatter(selected_router[0], selected_router[1],
               marker='s', color='red', s=150, label="Selected Router")

    ax.set_xlim(0, space_size_x)
    ax.set_ylim(0, space_size_y)
    ax.tick_params(axis='both', which='major', labelsize=fontsizebig)  # 放大 X 和 Y 轴刻度数字

    ax.set_xlabel("X (m)", fontsize=fontsizebig)
    ax.set_ylabel("Y (m)", fontsize=fontsizebig)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(titles[i], fontsize=fontsizebig)
    ax.grid(False)

    # 每个子图独立 colorbar，但 Accuracy Map 不需要
    if i < 3:  # 仅对前两个子图添加 colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)

        cbar.ax.tick_params(labelsize=16)
        cbar.set_label("RSSI (dBm)", fontsize=16)

# 整体标题（包含 layout 与准确率）
#plt.suptitle(f"Layout {layout_from_file} (Channel {channel_index})\nAccuracy Ratio: {accuracy_ratio:.2%}", fontsize=fontsizebig)
plt.tight_layout()
plt.show()
print(accuracy_ratio)
