#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random, math, subprocess, sys, time, glob, shutil, csv
from collections import deque
from datetime import datetime

BASE_HEADER = """\
#title: Random wood blocks in water
#domain: 5.0 1.0 0.004
#dx_dy_dz: 0.003 0.003 0.003
#time_window: 20e-9

#material: 1      0       1 0 air
#material: 2     0.5     1 0 water
#material: 30      0.0002  1 0 wood_dry

#waveform: ricker 1 4e8 my_ricker
#hertzian_dipole: z 0.30 0.10 0 my_ricker
#rx: 0.34 0.10 0
#src_steps: 0.01 0 0
#rx_steps:  0.01 0 0

#box: 0 0.00 0.00  5.0 1.00 0.004 water
"""

def detect_all_gpus():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        ids = [int(x.strip()) for x in out if x.strip().isdigit()]
        if ids:
            return ids
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

def non_overlap(a, b, margin=0.0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + margin <= bx1 or bx2 + margin <= ax1 or
                ay2 + margin <= by1 or by2 + margin <= ay1)

def gen_random_blocks(rng, k, x_rng, y_rng, w_rng, h_rng, max_trials=200, allow_overlap=False):
    boxes = []
    for _ in range(k):
        placed = False
        for _t in range(max_trials):
            w = rng.uniform(*w_rng)
            h = rng.uniform(*h_rng)
            x1 = rng.uniform(x_rng[0], x_rng[1] - w)
            y1 = rng.uniform(y_rng[0], y_rng[1] - h)
            cand = (x1, y1, x1 + w, y1 + h)
            if allow_overlap or all(not non_overlap(cand, b, margin=0.0) for b in boxes) is False:
                # 注意: non_overlap 返回“是否重叠的否定”。我们需要“不重叠”为True。
                pass
            # 正确的不重叠判断：
            ok = True
            if not allow_overlap:
                for b in boxes:
                    if non_overlap(cand, b, margin=0.0) is False:
                        ok = False
                        break
            if ok:
                boxes.append(cand)
                placed = True
                break
        if not placed and allow_overlap:
            # 强行放置
            w = rng.uniform(*w_rng); h = rng.uniform(*h_rng)
            x1 = rng.uniform(x_rng[0], x_rng[1] - w)
            y1 = rng.uniform(y_rng[0], y_rng[1] - h)
            boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes

def record_scene_params(csv_path, scene_data):
    """记录场景参数到CSV"""
    fieldnames = ['scene_id', 'infile', 'n_blocks', 'blocks_coords', 'frequency_hz', 
                  'tx_pos', 'rx_pos', 'n_traces', 'mute_ns', 'merged_outfile', 
                  'png_file', 'timestamp']
    
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(scene_data)

def build_in_text(blocks, material="wood_dry"):
    lines = [BASE_HEADER.rstrip()]
    for (x1, y1, x2, y2) in blocks:
        lines.append(f"#box: {x1:.3f} {y1:.3f} 0.000   {x2:.3f} {y2:.3f} 0.004 {material}")
    return "\n".join(lines) + "\n"

def sh(cmd, cwd=None, check=True):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r

def wait_any(active):
    """检查并返回已完成的进程PID列表"""
    finished = []
    for pid, item in list(active.items()):
        # 检查进程是否结束（非阻塞）
        try:
            # 使用 poll() 检查进程状态
            proc = subprocess.Popen(['ps', '-p', str(pid)], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            if proc.wait() != 0:  # 进程不存在
                finished.append(pid)
        except Exception:
            finished.append(pid)  # 异常时也认为进程结束
    return finished

def main():
    ap = argparse.ArgumentParser(description="Generate random wood blocks scenes and batch run on GPUs.")
    ap.add_argument("--outdir", default="t1010/sceneswater5", help="output directory for .in and results")
    ap.add_argument("--count", type=int, default=1, help="number of scenes to generate")
    ap.add_argument("--min-blocks", type=int, default=1, help="min wood blocks per scene")
    ap.add_argument("--max-blocks", type=int, default=5, help="max wood blocks per scene")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--runs", type=int, default=400, help="B-scan traces per scene (-n)")
    ap.add_argument("--gpus", default="auto", help="GPU list e.g. 0,1 or 'auto'")
    ap.add_argument("--keep-outs", action="store_true", help="do not delete individual out files after merge")
    ap.add_argument("--plot", action="store_true", help="plot B-scan PNG")
    ap.set_defaults(plot=True)
    ap.add_argument("--comp", default="Ez", choices=["Ex","Ey","Ez","Hx","Hy","Hz"], help="component for plotting")
    # wood placement ranges (meters)
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--x-max", type=float, default=4.5)
    ap.add_argument("--y-min", type=float, default=0.35)
    ap.add_argument("--y-max", type=float, default=0.95)
    ap.add_argument("--w-min", type=float, default=0.05)
    ap.add_argument("--w-max", type=float, default=0.25)
    ap.add_argument("--h-min", type=float, default=0.05)
    ap.add_argument("--h-max", type=float, default=0.20)
    ap.add_argument("--allow-overlap", action="store_true", help="allow blocks overlap")
    ap.add_argument("--mute-ns", type=float, default=6.0, help="Fixed mute time in ns (default 6.0 for 400MHz)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    
    # CSV记录文件
    csv_path = os.path.join(args.outdir, "scenes_log.csv")

    # Generate scenes
    inpaths = []
    scene_records = {}  # inpath -> scene_data for later recording
    
    for i in range(1, args.count + 1):
        k = rng.randint(args.min_blocks, args.max_blocks)
        blocks = gen_random_blocks(
            rng, k,
            (args.x_min, args.x_max),
            (args.y_min, args.y_max),
            (args.w_min, args.w_max),
            (args.h_min, args.h_max),
            allow_overlap=args.allow_overlap
        )
        text = build_in_text(blocks, material="wood_dry")
        inpath = os.path.join(args.outdir, f"scene_{i:04d}.in")
        with open(inpath, "w", encoding="utf-8") as f:
            f.write(text)
        inpaths.append(inpath)
        
        # 准备记录数据
        scene_records[inpath] = {
            'scene_id': f"scene_{i:04d}",
            'infile': os.path.basename(inpath),
            'n_blocks': k,
            'blocks_coords': '; '.join([f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})" for x1,y1,x2,y2 in blocks]),
            'frequency_hz': '4e8',  # 从 BASE_HEADER 中提取
            'tx_pos': '(0.30, 0.10, 0)',
            'rx_pos': '(0.34, 0.10, 0)', 
            'n_traces': args.runs,
            'mute_ns': args.mute_ns,
            'merged_outfile': '',  # 待填充
            'png_file': '',        # 待填充
            'timestamp': ''        # 待填充
        }

    # Detect GPUs
    if args.gpus.lower() == "auto":
        gpus = detect_all_gpus()
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if not gpus:
        print("No GPUs available.")
        sys.exit(1)
    print(f"Using GPUs: {gpus}")

    # Schedule per-scene runs across GPUs
    todo = deque(inpaths)
    active = {}  # pid -> dict(gpu, inpath, prefix, proc)
    
    while todo or active:
        busy = {it["gpu"] for it in active.values()}
        free = [g for g in gpus if g not in busy]
        
        # 启动新任务
        while todo and free:
            inpath = todo.popleft()
            gpu = free.pop(0)
            prefix = os.path.splitext(inpath)[0]
            cmd = [sys.executable, "-m", "gprMax", inpath, "-n", str(args.runs), "-gpu", str(gpu)]
            print(f"Launch {os.path.basename(inpath)} on GPU {gpu}")
            proc = subprocess.Popen(cmd)
            active[proc.pid] = dict(gpu=gpu, inpath=inpath, prefix=prefix, proc=proc)

        # 检查完成的任务
        finished = []
        for pid, item in list(active.items()):
            if item["proc"].poll() is not None:
                finished.append(pid)
        
        # 处理完成的任务
        for pid in finished:
            item = active.pop(pid)
            prefix = item["prefix"]
            inpath = item["inpath"]
            print(f"GPU {item['gpu']} finished: {os.path.basename(inpath)}")
            
            # 合并
            merge_cmd = [sys.executable, "-m", "tools.outputfiles_merge", prefix]
            if not args.keep_outs:
                merge_cmd.append("--remove-files")
            sh(merge_cmd)
            merged = f"{prefix}_merged.out"
            
            # 绘图（可选）
            png_file = ""
            if args.plot:
                sh([sys.executable, "-m", "tools.plot_Bscan_nodirect_1002", 
                    merged, args.comp, "--mute_ns", str(args.mute_ns)])
                # 推测PNG文件名（根据你的绘图脚本输出规律）
                png_pattern = f"{prefix}_*.png"
                png_files = glob.glob(png_pattern)
                if png_files:
                    png_file = os.path.basename(png_files[-1])  # 取最新的
            
            # 记录到CSV
            if inpath in scene_records:
                scene_records[inpath].update({
                    'merged_outfile': os.path.basename(merged),
                    'png_file': png_file,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                record_scene_params(csv_path, scene_records[inpath])
        
        if not finished and (todo or active):
            time.sleep(0.1)

    print(f"All scenes finished. Log saved to: {csv_path}")

if __name__ == "__main__":
    main()