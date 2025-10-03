# Copyright (C) 2015-2025: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#                 Mod: grayscale/mean-removal export by <your_name>
#
# Licensed under GPL v3 (see original headers).

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from .outputfiles_merge import get_output_data  # official helper to read merged B-scan
# Docs: https://docs.gprmax.com/en/latest/utils.html
# HDF5 output format: https://docs.gprmax.com/en/latest/output.html

def ensure_samples_traces(arr):
    """
    Ensure data is shaped as (samples, n_traces).
    Official plot_Bscan assumes arr.shape = (samples, n_traces),
    but some pipelines may produce (n_traces, samples).
    """
    if arr.shape[0] < arr.shape[1]:
        # likely (samples, n_traces) already
        return arr
    else:
        # likely (n_traces, samples) -> transpose
        return arr.T

def mean_removal_per_trace(section):
    """
    MALÅ风格的简单预处理：逐道去均值（去直流）
    section: (samples, n_traces)
    """
    mean_per_trace = section.mean(axis=0, keepdims=True)
    return section - mean_per_trace

def mpl_plot_gray(filename, section, dt, rxnumber, rxcomponent, cmap='gray'):
    """
    Matplotlib 绘制灰度 B-scan（与 MALÅ 例子一致的处理：仅去均值）
    section: (samples, n_traces) after mean removal
    dt: seconds per sample
    """
    (path, basename) = os.path.split(filename)
    samples, n_traces = section.shape
    time_window_ns = samples * dt * 1e9  # ns

    fig = plt.figure(num=f"{basename} - rx{rxnumber}", figsize=(10, 5), facecolor='w', edgecolor='w')

    extent = [0, n_traces, time_window_ns, 0]  # 上浅下深（时间向下增加）
    # 不做 dB/AGC，仅用灰度映射
    im = plt.imshow(section,
                    extent=extent,
                    interpolation='nearest',
                    aspect='auto',
                    cmap=cmap)

    plt.xlabel('Trace index')
    plt.ylabel('Time (ns)')

    cb = plt.colorbar(im)
    if 'E' in rxcomponent:
        cb.set_label('Field [V/m]')
    elif 'H' in rxcomponent:
        cb.set_label('Field [A/m]')
    elif 'I' in rxcomponent:
        cb.set_label('Current [A]')

    ax = fig.gca()
    ax.grid(which='both', axis='both', linestyle='-.')

    plt.tight_layout()
    return plt, time_window_ns

def export_png_csv(out_path, base_outfile, rx_idx, section, time_window_ns):
    """
    与 MALÅ 脚本对齐：导出 PNG 和 CSV
    CSV 第一列为 Time_ns，其后为 Trace_0...Trace_{N-1}
    section: (samples, n_traces)
    """
    import pandas as pd

    samples, n_traces = section.shape
    time_axis = np.linspace(0.0, time_window_ns, samples)  # ns

    # 生成文件名：bscan_<base>_rx<idx>_<timestamp>.png/csv
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(base_outfile))[0]
    png_name = f"bscan_{base}_rx{rx_idx}_{current_time}.png"
    csv_name = f"bscan_{base}_rx{rx_idx}_{current_time}.csv"
    png_path = os.path.join(out_path, png_name)
    csv_path = os.path.join(out_path, csv_name)

    # 保存 PNG（300 dpi）
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    # 保存 CSV
    # 构造 DataFrame：Time_ns + Trace_i
    trace_cols = [f"Trace_{i}" for i in range(n_traces)]
    # 注意：DataFrame 期望列为同长度向量，section 是 (samples, n_traces)
    # 先拼成二维数组后打上列名
    import pandas as pd
    df = pd.DataFrame(section, columns=trace_cols)
    df.insert(0, "Time_ns", time_axis)
    df.to_csv(csv_path, index=False)

    print(f"[Saved] PNG: {png_path}")
    print(f"[Saved] CSV: {csv_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Plots a B-scan image in grayscale and exports PNG/CSV '
                    '(MALÅ-like processing: per-trace mean removal, no dB/AGC).',
        usage='cd gprMax; python -m tools.plot_Bscan outputfile rx_component'
    )
    parser.add_argument('outputfile', help='name of MERGED output file including path (e.g., *_merged.out)')
    parser.add_argument('rx_component', help='output component to be plotted',
                        choices=['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 'Ix', 'Iy', 'Iz'])
    parser.add_argument('--cmap', default='gray', help="matplotlib colormap (default: gray, try gray_r for white=strong)")
    args = parser.parse_args()

    # 检查接收机数量
    with h5py.File(args.outputfile, 'r') as f:
        nrx = int(f.attrs.get('nrx', 0))
    if nrx == 0:
        raise CmdInputError(f'No receivers found in {args.outputfile}')

    # 输出目录（与 .out 同目录）
    out_dir = os.path.dirname(os.path.abspath(args.outputfile))

    last_plth = None
    last_time_ns = None

    for rx in range(1, nrx + 1):
        # 从官方工具获取 B-scan 数据（已合并的 out）
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        # 统一到 (samples, n_traces)
        section = ensure_samples_traces(outputdata)
        # 逐道去均值（与 MALÅ 示例一致）
        section = mean_removal_per_trace(section)

        # 绘图（灰度）
        plth, time_window_ns = mpl_plot_gray(args.outputfile, section, dt, rx, args.rx_component, cmap=args.cmap)
        # 保存 PNG 与 CSV
        export_png_csv(out_dir, args.outputfile, rx, section, time_window_ns)

        last_plth = plth
        last_time_ns = time_window_ns

    if last_plth is not None:
        last_plth.show()

if __name__ == "__main__":
    main()
