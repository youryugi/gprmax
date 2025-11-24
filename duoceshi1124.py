#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, subprocess, sys, time, csv, math, glob
from datetime import datetime

BASE_HEADER = """\
#title: Moving wood block in water (accelerated flow)
#domain: 5.0 1.0 0.004
#dx_dy_dz: 0.003 0.003 0.003
#time_window: 20e-9

#material: 1      0       1 0 air
#material: 80     0.5     1 0 water
#material: 3      0.0002  1 0 wood_dry

#waveform: ricker 1 4e8 my_ricker
#hertzian_dipole: z 0.30 0.10 0 my_ricker
#rx: 0.30 0.10 0
#src_steps: 0 0 0
#rx_steps:  0 0 0

#box: 0 0.00 0.00  5.0 1.00 0.004 water
"""

def detect_all_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        ids = [int(x.strip()) for x in out if x.strip().isdigit()]
        if ids: return ids
    except Exception:
        pass
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return list(range(n))
    except Exception:
        pass
    return [0]

def write_infile(path, x0, y0, x1, y1):
    lines = [BASE_HEADER.rstrip()]
    # 去掉尾部 // 注释，保持严格 7 个参数：x0 y0 z0 x1 y1 z1 material
    lines.append(f"#box: {x0:.3f} {y0:.3f} 0.000   {x1:.3f} {y1:.3f} 0.004 wood_dry")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def run_gprmax(infile, runs, gpu):
    cmd = [sys.executable, "-m", "gprMax", infile, "-n", str(runs), "-gpu", str(gpu)]
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0

def merge_outputs(prefix, remove=True):
    cmd = [sys.executable, "-m", "tools.outputfiles_merge", prefix]
    if remove:
        cmd.append("--remove-files")
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return f"{prefix}_merged.out"

def plot_Bscan(outpath, comp, mute_ns):
    cmd = [sys.executable, "-m", "tools.plot_Bscan_nodirect_1002", outpath, comp, "--mute_ns", str(mute_ns)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    patt = os.path.splitext(outpath)[0] + "_*.png"
    files = glob.glob(patt)
    return os.path.basename(files[-1]) if files else ""

def main():
    ap = argparse.ArgumentParser(description="Moving single wood block with acceleration (frames).")
    ap.add_argument("--outdir", default="t1010/moving_block", help="output directory")
    ap.add_argument("--frames", type=int, default=200, help="number of frames")
    ap.add_argument("--dt", type=float, default=0.2, help="frame time step (s)")
    ap.add_argument("--v0", type=float, default=0.05, help="initial velocity (m/s)")
    ap.add_argument("--acc", type=float, default=0.02, help="acceleration (m/s^2)")
    ap.add_argument("--x-start", type=float, default=0.5, help="initial left x of block")
    ap.add_argument("--y-center", type=float, default=0.60, help="y center of block")
    ap.add_argument("--len-x", type=float, default=0.20, help="block length in x (m)")
    ap.add_argument("--len-y", type=float, default=0.12, help="block length in y (m)")
    ap.add_argument("--x-max", type=float, default=4.7, help="max allowed x before stop")
    ap.add_argument("--runs", type=int, default=1, help="-n traces per frame (keep 1 when radar fixed)")
    ap.add_argument("--gpu", default="auto", help="gpu id or 'auto'")
    ap.add_argument("--plot", action="store_true", help="plot B-scan")
    ap.add_argument("--comp", default="Ez", choices=["Ex","Ey","Ez","Hx","Hy","Hz"], help="component for plotting")
    ap.add_argument("--mute-ns", type=float, default=6.0)
    ap.add_argument("--keep-outs", action="store_true")
    ap.add_argument("--log-csv", default="moving_log.csv")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.gpu == "auto":
        gpu_ids = detect_all_gpus()
        gpu = gpu_ids[0]
    else:
        gpu = int(args.gpu)

    def displacement(t):
        return args.v0 * t + 0.5 * args.acc * t * t

    csv_path = os.path.join(args.outdir, args.log_csv)
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        if need_header:
            writer.writerow(["frame","t_s","x_left","x_right","y_bottom","y_top",
                             "v_inst","infile","merged_out","png","timestamp"])
        for i in range(args.frames):
            t = i * args.dt
            s = displacement(t)
            v_inst = args.v0 + args.acc * t
            x0 = args.x_start + s
            x1 = x0 + args.len_x
            if x1 > args.x_max:
                print(f"Frame {i} exceeds x-max, stop.")
                break
            y0 = args.y_center - args.len_y/2
            y1 = args.y_center + args.len_y/2

            inpath = os.path.join(args.outdir, f"frame_{i:04d}.in")
            write_infile(inpath, x0, y0, x1, y1)

            ok = run_gprmax(inpath, args.runs, gpu)
            if not ok:
                print(f"Frame {i} failed.")
                continue

            prefix = os.path.splitext(inpath)[0]
            merged = merge_outputs(prefix, remove=not args.keep_outs)

            png = ""
            if args.plot:
                png = plot_Bscan(merged, args.comp, args.mute_ns)

            writer.writerow([
                i, f"{t:.3f}", f"{x0:.3f}", f"{x1:.3f}",
                f"{y0:.3f}", f"{y1:.3f}", f"{v_inst:.3f}",
                os.path.basename(inpath),
                os.path.basename(merged),
                png,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
            print(f"Frame {i} done: x=({x0:.3f},{x1:.3f}) v={v_inst:.3f} m/s")

    print("All frames finished. CSV:", csv_path)

if __name__ == "__main__":
    main()