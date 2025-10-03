#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys, os, math, shutil, time

def sh(cmd, env=None):
    print(">>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        sys.exit(r.returncode)

def autodetect_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        ).decode().strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception:
        return []

def split_runs_by_weights(total_runs, weights):
    total_weight = sum(weights)
    sizes = [round(total_runs * w / total_weight) for w in weights]
    # 调整四舍五入误差
    diff = total_runs - sum(sizes)
    for i in range(abs(diff)):
        idx = i % len(sizes)
        sizes[idx] += 1 if diff > 0 else -1
    ranges = []
    start = 1
    for sz in sizes:
        end = min(total_runs, start + sz - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges

def modify_in_file(infile, out_infile, r1, r2, total_runs):
    # 复制 .in 文件并修改 #src_steps 和 #rx_steps
    shutil.copy(infile, out_infile)
    with open(out_infile, 'r') as f:
        lines = f.readlines()
    with open(out_infile, 'w') as f:
        for line in lines:
            if line.startswith('#src_steps'):
                # 修改步进：假设原步进为 dx dy dz，调整为子区间
                parts = line.split()
                # 简单：保持步进，但后续用 -n (r2-r1+1) 限制 runs
                f.write(line)
            elif line.startswith('#rx_steps'):
                f.write(line)
            else:
                f.write(line)
    # 注意：实际分区需根据步进计算起始位置，这里简化，只用 -n 限制 runs 数
    return out_infile

def main():
    ap = argparse.ArgumentParser(description="Run gprMax with GPUs, split runs by weights, then merge and plot B-scan.")
    ap.add_argument("--infile", required=True, help=".in file")
    ap.add_argument("--runs", type=int, required=True, help="Total runs for -n")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU ids, e.g. '0,1'. If empty, use all visible GPUs.")
    ap.add_argument("--weights", default="1,2", help="Comma-separated weights for GPUs, e.g. '1,2' (same length as --gpus). If empty, equal weights.")
    ap.add_argument("--comp", default="Ez", help="Field component for plot_Bscan (e.g. Ez)")
    ap.add_argument("--geometry-fixed", action="store_true", help="Pass --geometry-fixed to speed B-scan")
    args = ap.parse_args()

    # 校验 infile
    if not os.path.isfile(args.infile):
        print(f"Input file not found: {args.infile}")
        sys.exit(1)

    # 解析 GPU 列表
    if args.gpus.strip():
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    else:
        gpu_ids = autodetect_gpus()
        if not gpu_ids:
            print("No GPUs found. Use --gpus to specify GPU ids.")
            sys.exit(1)

    # 解析权重
    if args.weights.strip():
        weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
        if len(weights) != len(gpu_ids):
            print("Weights length must match GPUs length.")
            sys.exit(1)
    else:
        weights = [1.0] * len(gpu_ids)  # 默认等权重

    # 计算每个 GPU 的 runs 区间
    ranges = split_runs_by_weights(args.runs, weights)
    print(f"GPU assignments: {list(zip(gpu_ids, ranges))}")

    # 环境设置
    env = os.environ.copy()
    if not env.get("DISPLAY"):
        env["MPLBACKEND"] = "Agg"

    # 为每个 GPU 生成分区 .in 文件并运行
    procs = []
    gpu_times = {}
    for i, (gpu_id, (r1, r2)) in enumerate(zip(gpu_ids, ranges)):
        if r1 > r2:
            continue
        out_infile = f"{args.infile.rsplit('.', 1)[0]}_gpu{i}.in"
        modify_in_file(args.infile, out_infile, r1, r2, args.runs)
        cmd = [sys.executable, "-m", "gprMax", out_infile, "-n", str(r2 - r1 + 1), "-gpu", gpu_id]
        if args.geometry_fixed:
            cmd.append("--geometry-fixed")
        print(f"Launching GPU {gpu_id}: runs {r1}-{r2} with {out_infile}")
        start_time = time.time()
        p = subprocess.Popen(cmd, env=env)
        procs.append((p, gpu_id, start_time))

    # 等待所有并记录时间
    rc = 0
    for p, gpu_id, start_time in procs:
        p.wait()
        end_time = time.time()
        gpu_times[gpu_id] = end_time - start_time
        rc |= p.returncode or 0
    if rc != 0:
        print("Some GPU runs failed.")
        sys.exit(rc)

    # 打印时间统计
    print("\nGPU Time Statistics:")
    for gpu_id, t in gpu_times.items():
        print(f"GPU {gpu_id}: {t:.2f} seconds")

    # 重命名输出文件以匹配合并
    prefix = args.infile.rsplit(".", 1)[0]
    for i, (gpu_id, (r1, r2)) in enumerate(zip(gpu_ids, ranges)):
        gpu_prefix = f"{args.infile.rsplit('.', 1)[0]}_gpu{i}"
        for j in range(r1, r2 + 1):
            old_out = f"{gpu_prefix}{j}.out"
            new_out = f"{prefix}{j}.out"
            if os.path.exists(old_out):
                os.rename(old_out, new_out)

    # 合并与绘图
    sh([sys.executable, "-m", "tools.outputfiles_merge", prefix, "--remove-files"], env=env)
    merged = f"{prefix}_merged.out"
    print("Plotting B-scan using tools.plot_Bscan...")
    sh([sys.executable, "-m", "tools.plot_Bscan", merged, args.comp], env=env)

if __name__ == "__main__":
    main()
