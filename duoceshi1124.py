#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小水槽实验：固定GPR天线观测水中加速运动的木头
生成时间序列B-scan用于深度+速度回归训练
"""
import os, sys, csv, argparse, subprocess, h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

BASE_HEADER_TEMPLATE = """\
#title: Moving wood block in water tank (accelerated flow)
#domain: {domain_x} {domain_y} {domain_z}
#dx_dy_dz: 0.002 0.002 0.002
#time_window: {time_window}

#material: 1      0       1 0 air
#material: 80     0.5     1 0 water
#material: 3      0.0002  1 0 wood_dry

#waveform: ricker 1 {freq} my_ricker
#hertzian_dipole: z {tx_x} {tx_y} {tx_z} my_ricker
#rx: {rx_x} {rx_y} {rx_z}

#box: 0 0 0  {domain_x} {domain_y} {water_depth} water
"""

def detect_gpu():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader"], text=True).strip().splitlines()
        ids = [int(x) for x in out if x.isdigit()]
        if ids: return ids[0]
    except Exception: pass
    return 0

def build_header(args):
    """构建GPRMax输入文件头部"""
    # 天线位置：在水面上方
    tx_x = args.domain_x / 2  # 水槽中央
    tx_y = args.domain_y / 2
    tx_z = args.water_depth + 0.05  # 水面上5cm
    
    # 接收天线：偏移配置或同位置
    rx_x = tx_x + args.antenna_offset
    rx_y = tx_y
    rx_z = tx_z
    
    return BASE_HEADER_TEMPLATE.format(
        domain_x=args.domain_x,
        domain_y=args.domain_y,
        domain_z=args.domain_z,
        time_window=f"{args.time_window:.9g}",
        freq=f"{args.wave_freq:.9g}",
        tx_x=f"{tx_x:.4f}",
        tx_y=f"{tx_y:.4f}",
        tx_z=f"{tx_z:.4f}",
        rx_x=f"{rx_x:.4f}",
        rx_y=f"{rx_y:.4f}",
        rx_z=f"{rx_z:.4f}",
        water_depth=f"{args.water_depth:.4f}"
    ).rstrip()

def write_infile(path, header, wood_params):
    """
    写入包含木头位置的输入文件
    wood_params: dict with keys (x_center, y_center, z_center, len_x, len_y, len_z)
    """
    x0 = wood_params['x_center'] - wood_params['len_x']/2
    x1 = wood_params['x_center'] + wood_params['len_x']/2
    y0 = wood_params['y_center'] - wood_params['len_y']/2
    y1 = wood_params['y_center'] + wood_params['len_y']/2
    z0 = wood_params['z_center'] - wood_params['len_z']/2
    z1 = wood_params['z_center'] + wood_params['len_z']/2
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(f"#box: {x0:.4f} {y0:.4f} {z0:.4f}   {x1:.4f} {y1:.4f} {z1:.4f} wood_dry\n")

def run_gprmax(infile, gpu, timeout=120):
    """运行单次GPRMax模拟（不使用-n参数）"""
    cmd = [sys.executable, "-m", "gprMax", infile, "-gpu", str(gpu)]
    print(">>", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Simulation timeout after {timeout}s")
        return False

def read_ascan(outfile, component='Ez'):
    """读取单次扫描的A-scan数据"""
    try:
        with h5py.File(outfile, "r") as f:
            trace = f["rxs"]["rx1"][component][()]
            dt = f.attrs.get("dt", None)
            if dt is None:
                tw = float(f.attrs["time_window"])
                it = int(f.attrs["iterations"])
                dt = tw / it if it > 0 else 1e-9
        return trace.flatten(), dt
    except Exception as e:
        print(f"Error reading {outfile}: {e}")
        return None, None

def plot_bscan_time_series(frame_data, dt, outpng, title=""):
    """
    生成时间序列B-scan（横轴=时间/木头位置，纵轴=双程走时）
    frame_data: list of (time_s, x_position, trace)
    """
    if not frame_data:
        print("No data to plot")
        return
    
    times = [t for t, x, tr in frame_data]
    positions = [x for t, x, tr in frame_data]
    traces = [tr for t, x, tr in frame_data]
    
    # 确保所有trace长度相同
    min_len = min(len(tr) for tr in traces)
    traces = [tr[:min_len] for tr in traces]
    
    data = np.array(traces).T  # shape: (time_samples, n_frames)
    
    # 去除直流分量
    data = data - np.mean(data, axis=0, keepdims=True)
    
    # 归一化显示
    vmax = np.percentile(np.abs(data), 98) or 1e-6
    
    Nt, Nframes = data.shape
    
    # 横轴可以选择实际时间或木头位置
    extent = [min(positions), max(positions), (Nt-1)*dt*1e9, 0]  # dt转换为ns
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # B-scan图像
    im = ax1.imshow(data, aspect='auto', cmap='seismic', 
                    vmin=-vmax, vmax=vmax, origin='upper', extent=extent)
    ax1.set_xlabel("Wood X position (m)")
    ax1.set_ylabel("Two-way travel time (ns)")
    ax1.set_title(f"B-scan: {title}")
    plt.colorbar(im, ax=ax1, label="Ez (V/m)")
    
    # 速度-位置曲线
    ax2.plot(times, positions, 'b-o', markersize=4)
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_ylabel("Wood position (m)")
    ax2.set_title("Wood trajectory")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()
    print(f"✓ B-scan saved: {outpng}")

def plot_single_ascan(trace, dt, outpng, frame_id=0):
    """绘制单条A-scan波形"""
    time_ns = np.arange(len(trace)) * dt * 1e9
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_ns, trace, 'b-', linewidth=0.8)
    plt.xlabel("Time (ns)")
    plt.ylabel("Ez (V/m)")
    plt.title(f"A-scan - Frame {frame_id}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpng, dpi=120)
    plt.close()

def main():
    ap = argparse.ArgumentParser(
        description="Small water tank GPR simulation: fixed antenna observing moving wood"
    )
    
    # 水槽几何参数
    ap.add_argument("--domain-x", type=float, default=2.0, help="Tank length (m)")
    ap.add_argument("--domain-y", type=float, default=0.5, help="Tank width (m)")
    ap.add_argument("--domain-z", type=float, default=0.4, help="Tank height (m)")
    ap.add_argument("--water-depth", type=float, default=0.25, help="Water depth (m)")
    
    # 木头参数
    ap.add_argument("--wood-len-x", type=float, default=0.05, help="Wood length X (m)")
    ap.add_argument("--wood-len-y", type=float, default=0.08, help="Wood length Y (m)")
    ap.add_argument("--wood-len-z", type=float, default=0.04, help="Wood length Z (m)")
    ap.add_argument("--wood-depth", type=float, default=0.10, 
                    help="Wood center depth below water surface (m)")
    
    # 运动参数（木头沿X轴运动）
    ap.add_argument("--x-start", type=float, default=0.3, 
                    help="Initial X position (m)")
    ap.add_argument("--v0", type=float, default=0.05, 
                    help="Initial velocity (m/s)")
    ap.add_argument("--acc", type=float, default=0.02, 
                    help="Acceleration (m/s²)")
    ap.add_argument("--frames", type=int, default=30, 
                    help="Number of time frames")
    ap.add_argument("--dt-frame", type=float, default=0.1, 
                    help="Real time interval between frames (s)")
    
    # GPR参数
    ap.add_argument("--time-window", type=float, default=20e-9,
                    help="GPR time window (s)")
    ap.add_argument("--wave-freq", type=float, default=1e9,
                    help="Center frequency (Hz)")
    ap.add_argument("--antenna-offset", type=float, default=0.05,
                    help="TX-RX offset distance (m), 0 for monostatic")
    
    # 处理参数
    ap.add_argument("--outdir", default="water_tank_sim")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--component", default="Ez", 
                    choices=['Ex','Ey','Ez'], help="Field component to plot")
    ap.add_argument("--skip-sim", action="store_true",
                    help="Skip simulation, only regenerate plots")
    ap.add_argument("--plot-ascans", action="store_true",
                    help="Plot individual A-scans")
    
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    gpu = detect_gpu() if args.gpu == "auto" else int(args.gpu)
    
    # 运动学函数
    def wood_x_at_time(t):
        return args.x_start + args.v0 * t + 0.5 * args.acc * t * t
    
    def wood_v_at_time(t):
        return args.v0 + args.acc * t
    
    # 木头Y坐标（水槽中央）和Z坐标（固定深度）
    y_center = args.domain_y / 2
    z_center = args.water_depth - args.wood_depth
    
    # 验证参数合理性
    if z_center - args.wood_len_z/2 < 0:
        print(f"⚠ Warning: Wood bottom at {z_center - args.wood_len_z/2:.3f}m < 0")
    if z_center + args.wood_len_z/2 > args.water_depth:
        print(f"⚠ Warning: Wood top above water surface")
    
    # CSV日志
    logcsv = os.path.join(args.outdir, "simulation_log.csv")
    need_header = not os.path.exists(logcsv) or args.skip_sim
    
    frame_data = []
    
    if not args.skip_sim:
        header = build_header(args)
        
        print("\n" + "="*60)
        print("SIMULATION PARAMETERS:")
        print(f"  Domain: {args.domain_x} × {args.domain_y} × {args.domain_z} m")
        print(f"  Water depth: {args.water_depth} m")
        print(f"  Wood size: {args.wood_len_x} × {args.wood_len_y} × {args.wood_len_z} m")
        print(f"  Wood depth: {args.wood_depth} m below surface (z={z_center:.3f}m)")
        print(f"  Motion: v0={args.v0} m/s, a={args.acc} m/s²")
        print(f"  Frames: {args.frames} × {args.dt_frame}s = {args.frames*args.dt_frame}s total")
        print("="*60 + "\n")
        
        with open(logcsv, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "frame", "time_s", "x_pos_m", "velocity_m/s", 
                "wood_depth_m", "outfile", "timestamp"
            ])
            
            for i in range(args.frames):
                t = i * args.dt_frame
                x_pos = wood_x_at_time(t)
                v_inst = wood_v_at_time(t)
                
                # 检查边界
                if x_pos < args.wood_len_x/2 or x_pos > args.domain_x - args.wood_len_x/2:
                    print(f"Frame {i}: Wood out of bounds (x={x_pos:.3f}m), stopping")
                    break
                
                wood_params = {
                    'x_center': x_pos,
                    'y_center': y_center,
                    'z_center': z_center,
                    'len_x': args.wood_len_x,
                    'len_y': args.wood_len_y,
                    'len_z': args.wood_len_z
                }
                
                inpath = os.path.join(args.outdir, f"frame_{i:04d}.in")
                write_infile(inpath, header, wood_params)
                
                if not run_gprmax(inpath, gpu):
                    print(f"❌ Frame {i} failed")
                    continue
                
                outfile = os.path.splitext(inpath)[0] + ".out"
                trace, dt = read_ascan(outfile, args.component)
                
                if trace is not None:
                    frame_data.append((t, x_pos, trace))
                    
                    if args.plot_ascans:
                        ascan_png = os.path.splitext(outfile)[0] + "_ascan.png"
                        plot_single_ascan(trace, dt, ascan_png, i)
                    
                    writer.writerow([
                        i, f"{t:.4f}", f"{x_pos:.4f}", f"{v_inst:.4f}",
                        f"{args.wood_depth:.4f}", os.path.basename(outfile),
                        datetime.now().isoformat(timespec='seconds')
                    ])
                    
                    print(f"✓ Frame {i:3d}/{args.frames}: t={t:5.2f}s, x={x_pos:5.3f}m, v={v_inst:5.3f}m/s")
                else:
                    print(f"❌ Frame {i}: Failed to read output")
    
    else:
        # 从已有文件读取
        print("Loading existing data...")
        for i in range(args.frames):
            outfile = os.path.join(args.outdir, f"frame_{i:04d}.out")
            if not os.path.exists(outfile):
                continue
            t = i * args.dt_frame
            x_pos = wood_x_at_time(t)
            trace, dt = read_ascan(outfile, args.component)
            if trace is not None:
                frame_data.append((t, x_pos, trace))
    
    # 生成B-scan
    if frame_data:
        bscan_png = os.path.join(args.outdir, f"bscan_{args.component}.png")
        plot_bscan_time_series(
            frame_data, dt, bscan_png, 
            title=f"Water tank - depth={args.wood_depth}m, v0={args.v0}m/s, a={args.acc}m/s²"
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Simulation complete: {len(frame_data)} frames")
        print(f"✓ B-scan image: {bscan_png}")
        print(f"✓ Log file: {logcsv}")
        print(f"{'='*60}\n")
    else:
        print("⚠ No valid data to plot")

if __name__ == "__main__":
    main()