# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import matplotlib.pyplot as plt

from gprMax.exceptions import CmdInputError
from .outputfiles_merge import get_output_data

# 需要 scipy 做 Hilbert 包络
try:
    from scipy.signal import hilbert
except Exception as e:
    raise ImportError(
        "需要 scipy 才能进行 Hilbert 包络处理。请先安装：pip install scipy\n"
        f"原始错误：{e}"
    )

# -----------------------------
# 预处理：去背景 + 包络 + AGC
# -----------------------------
def preprocess_bscan(
    bscan: np.ndarray,
    dt: float,
    do_bgrem: bool = True,
    do_envelope: bool = True,
    do_agc: bool = True,
    agc_win_ns: float = 20.0,
    dewow_ns: float = 0.0,
) -> np.ndarray:
    """
    bscan: shape = (samples, traces)，时域在第0维，走时从上到下
    dt: 采样间隔（秒）
    do_bgrem: 背景去除（按样本减去跨道均值）
    do_envelope: Hilbert 包络
    do_agc: 滑窗RMS归一化
    agc_win_ns: AGC窗口（纳秒）
    dewow_ns: 若>0，则做去直流/去漂移（长窗滑动均值相减），单位纳秒
    """
    x = bscan.astype(np.float32, copy=False)

    # (可选) dewow：去直流/低频漂移（长窗滑动均值）
    if dewow_ns and dewow_ns > 0:
        win = max(3, int((dewow_ns * 1e-9) / dt))
        win = win if win % 2 == 1 else win + 1
        pad = win // 2
        # 沿时间维做滑动均值
        kernel = np.ones(win, dtype=np.float32) / win
        x_pad = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
        mean_smooth = np.apply_along_axis(
            lambda a: np.convolve(a, kernel, mode="valid"), 0, x_pad
        )
        x = x - mean_smooth

    # 1) 去背景（跨道均值）：消除水平条纹/系统响应
    if do_bgrem:
        mean_trace = np.mean(x, axis=1, keepdims=True)
        x = x - mean_trace

    # 2) 包络：Hilbert 解析信号幅值
    if do_envelope:
        x = np.abs(hilbert(x, axis=0)).astype(np.float32, copy=False)

    # 3) AGC：按时间维做滑窗RMS归一化，增强深部弱反射
    if do_agc:
        win = max(3, int((agc_win_ns * 1e-9) / dt))
        win = win if win % 2 == 1 else win + 1
        pad = win // 2
        x2 = np.pad(x**2, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(win, dtype=np.float32) / win
        # 滑动均值 -> RMS
        rms = np.sqrt(
            np.apply_along_axis(lambda a: np.convolve(a, kernel, mode="valid"), 0, x2)
        )
        x = x / (rms + 1e-9)

    return x


def mpl_plot(filename, outputdata, dt, rxnumber, rxcomponent):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    (path, filename_only) = os.path.split(filename)

    # —— 安全模式：只看原始 |abs| —— 
    data = np.nan_to_num(np.abs(outputdata).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # 自动对比度：若百分位不可用则兜底
    try:
        vmax = float(np.nanpercentile(data, 99.5))
    except Exception:
        vmax = float(np.nanmax(data))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(data) if np.nanmax(data) > 0 else 1.0)
    vmin = 0.0

    fig = plt.figure(num=filename_only + f" - rx{rxnumber}", figsize=(20, 10), facecolor="w")
    plt.imshow(data,
               extent=[0, data.shape[1], data.shape[0] * dt, 0],
               interpolation="nearest", aspect="auto",
               cmap="gray_r", vmin=vmin, vmax=vmax)
    plt.xlabel("Trace number")
    plt.ylabel("Time [s]")
    plt.colorbar(label="|Amplitude|")
    fig.gca().grid(which="both", axis="both", linestyle="-.")
    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plots a B-scan image.",
        usage="cd gprMax; python -m tools.plot_Bscan outputfile output",
    )
    parser.add_argument("outputfile", help="name of output file including path")
    parser.add_argument(
        "rx_component",
        help="name of output component to be plotted",
        choices=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "Ix", "Iy", "Iz"],
    )
    args = parser.parse_args()

    # Open output file and read number of outputs (receivers)
    f = h5py.File(args.outputfile, "r")
    nrx = f.attrs["nrx"]
    f.close()

    # Check there are any receivers
    if nrx == 0:
        raise CmdInputError("No receivers found in {}".format(args.outputfile))

    plthandle = None
    for rx in range(1, nrx + 1):
        outputdata, dt = get_output_data(args.outputfile, rx, args.rx_component)
        plthandle = mpl_plot(args.outputfile, outputdata, dt, rx, args.rx_component)

    # 显示
    plthandle.show()

    # 生成文件名并保存 PNG（与输出文件同目录）
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_name = os.path.splitext(os.path.basename(args.outputfile))[0]
    filename = f"bscan_{base_output_name}_{current_time}.png"
    out_dir = os.path.dirname(os.path.abspath(args.outputfile))
    savepath = os.path.join(out_dir, filename)

    plt.savefig(savepath, dpi=300)
    print(f"B-scan saved as: {savepath}")
