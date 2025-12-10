#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random, subprocess, sys, time, glob, csv
from collections import deque
from datetime import datetime

import numpy as np
import h5py

BASE_HEADER = """\
#title: Random wood blocks in water
#domain: 5.0 1.0 0.004
#dx_dy_dz: 0.003 0.003 0.003
#time_window: 20e-9

#material: 1      0       1 0 air
#material: 1     0.5     1 0 water
#material: 20      0.0002  1 0 wood_dry

#waveform: ricker 1 4e8 my_ricker
#hertzian_dipole: z 1.30 0.10 0 my_ricker
#rx: 1.34 0.10 0
#src_steps: 0.1 0 0
#rx_steps:  0.1 0 0

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
                pass
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
            w = rng.uniform(*w_rng); h = rng.uniform(*h_rng)
            x1 = rng.uniform(x_rng[0], x_rng[1] - w)
            y1 = rng.uniform(y_rng[0], y_rng[1] - h)
            boxes.append((x1, y1, x1 + w, y1 + h))
    return boxes

def record_scene_params(csv_path, scene_data):
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

def merge_frames_to_big_out(outdir, scene_index, nframes, runs):
    base_scene = f"scene_{scene_index:04d}"
    outfiles = []
    for frame in range(nframes):
        prefix = os.path.join(outdir, f"{base_scene}_t{frame:04d}")
        datafile = prefix + ("_merged.out" if runs > 1 else ".out")
        if os.path.exists(datafile):
            outfiles.append(datafile)
        else:
            print(f"[WARN] Missing out for frame {frame}: {datafile}")
    if not outfiles:
        print(f"[WARN] No outfiles for {base_scene}")
        return None

    print(f"[INFO] Merging {len(outfiles)} frames for {base_scene}")
    first = outfiles[0]
    with h5py.File(first, "r") as f0:
        root_attrs = dict(f0.attrs)
        if "rxs" not in f0:
            print(f"[ERROR] No rxs in {first}")
            return None
        rx_names = list(f0["rxs"].keys())
        comp_names = list(f0["rxs"][rx_names[0]].keys())

    data_all = {rx: {comp: [] for comp in comp_names} for rx in rx_names}
    for path in outfiles:
        with h5py.File(path, "r") as f:
            for rx in rx_names:
                grp_rx = f["rxs"][rx]
                for comp in comp_names:
                    data_all[rx][comp].append(grp_rx[comp][()])

    min_len = {}
    for rx in rx_names:
        for comp in comp_names:
            arrs = data_all[rx][comp]
            min_len[(rx, comp)] = min(len(a) for a in arrs)

    big_out = os.path.join(outdir, f"{base_scene}_motion_merged.out")
    if os.path.exists(big_out):
        os.remove(big_out)

    with h5py.File(big_out, "w") as fout, h5py.File(first, "r") as f0:
        for k, v in root_attrs.items():
            fout.attrs[k] = v
        fout.attrs["nmodels"] = len(outfiles)
        if "nrx" in root_attrs:
            fout.attrs["nrx"] = root_attrs["nrx"]
        else:
            fout.attrs["nrx"] = len(rx_names)

        g_rxs_out = fout.create_group("rxs")
        g_rxs_in = f0["rxs"]

        for rx in rx_names:
            g_rx_out = g_rxs_out.create_group(rx)
            g_rx_in = g_rxs_in[rx]
            for comp in comp_names:
                arrs = data_all[rx][comp]
                T = min_len[(rx, comp)]
                N = len(arrs)
                mat = np.zeros((T, N), dtype=arrs[0].dtype)
                for j, a in enumerate(arrs):
                    mat[:, j] = a[:T]

                dset_in = g_rx_in[comp]
                dset_out = g_rx_out.create_dataset(comp, data=mat)
                for ak, av in dset_in.attrs.items():
                    dset_out.attrs[ak] = av

    print(f"[INFO] Big out written: {big_out}")
    return big_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="t1010/sceneswater3")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--min-blocks", type=int, default=1)
    ap.add_argument("--max-blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--gpus", default="auto")
    ap.add_argument("--keep-outs", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="plot from merged big out")
    ap.set_defaults(plot=True)
    ap.add_argument("--comp", default="Ez",
                    choices=["Ex","Ey","Ez","Hx","Hy","Hz"])
    ap.add_argument("--x-min", type=float, default=0.5)
    ap.add_argument("--x-max", type=float, default=4.5)
    ap.add_argument("--y-min", type=float, default=0.35)
    ap.add_argument("--y-max", type=float, default=0.95)
    ap.add_argument("--w-min", type=float, default=0.05)
    ap.add_argument("--w-max", type=float, default=0.25)
    ap.add_argument("--h-min", type=float, default=0.05)
    ap.add_argument("--h-max", type=float, default=0.20)
    ap.add_argument("--allow-overlap", action="store_true")
    ap.add_argument("--mute-ns", type=float, default=1.0)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--slow-dt", type=float, default=0.05)
    ap.add_argument("--vx-min", type=float, default=-0.5)
    ap.add_argument("--vx-max", type=float, default=0.5)
    ap.add_argument("--ax-min", type=float, default=-0.5)
    ap.add_argument("--ax-max", type=float, default=0.5)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "scenes_log.csv")

    inpaths = []
    scene_records = {}

    for i in range(1, args.count + 1):
        k = rng.randint(args.min_blocks, args.max_blocks)
        blocks0 = gen_random_blocks(
            rng, k,
            (args.x_min, args.x_max),
            (args.y_min, args.y_max),
            (args.w_min, args.w_max),
            (args.h_min, args.h_max),
            allow_overlap=args.allow_overlap
        )

        vxs = [rng.uniform(args.vx_min, args.vx_max) for _ in range(k)]
        axs = [rng.uniform(args.ax_min, args.ax_max) for _ in range(k)]

        for frame in range(args.frames):
            t = frame * args.slow_dt
            moved_blocks = []
            for idx, (x1, y1, x2, y2) in enumerate(blocks0):
                vx = vxs[idx]
                ax = axs[idx]
                dx = vx * t + 0.5 * ax * t * t
                moved_blocks.append((x1 + dx, y1, x2 + dx, y2))

            text = build_in_text(moved_blocks, material="wood_dry")
            scene_name = f"scene_{i:04d}_t{frame:04d}"
            inpath = os.path.join(args.outdir, f"{scene_name}.in")
            with open(inpath, "w", encoding="utf-8") as f:
                f.write(text)
            inpaths.append(inpath)

            scene_records[inpath] = {
                'scene_id': scene_name,
                'infile': os.path.basename(inpath),
                'n_blocks': k,
                'blocks_coords': '; '.join(
                    [f"({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})" for x1, y1, x2, y2 in moved_blocks]
                ),
                'frequency_hz': '4e8',
                'tx_pos': '(0.30, 0.10, 0)',
                'rx_pos': '(0.34, 0.10, 0)',
                'n_traces': args.runs,
                'mute_ns': args.mute_ns,
                'merged_outfile': '',
                'png_file': '',
                'timestamp': ''
            }

    # GPU
    if args.gpus.lower() == "auto":
        gpus = detect_all_gpus()
    else:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        print("No GPUs available.")
        sys.exit(1)
    print(f"Using GPUs: {gpus}")

    todo = deque(inpaths)
    active = {}

    while todo or active:
        busy = {it["gpu"] for it in active.values()}
        free = [g for g in gpus if g not in busy]

        while todo and free:
            inpath = todo.popleft()
            gpu = free.pop(0)
            prefix = os.path.splitext(inpath)[0]
            cmd = [sys.executable, "-m", "gprMax", inpath, "-n", str(args.runs), "-gpu", str(gpu)]
            print(f"Launch {os.path.basename(inpath)} on GPU {gpu}")
            proc = subprocess.Popen(cmd)
            active[proc.pid] = dict(gpu=gpu, inpath=inpath, prefix=prefix, proc=proc)

        finished = []
        for pid, item in list(active.items()):
            if item["proc"].poll() is not None:
                finished.append(pid)

        for pid in finished:
            item = active.pop(pid)
            prefix = item["prefix"]
            inpath = item["inpath"]
            print(f"GPU {item['gpu']} finished: {os.path.basename(inpath)}")

            # runs>1 时可以合并单个 .in 的多 run；现在一般 runs=1，就直接用 .out
            if args.runs > 1:
                merge_cmd = [sys.executable, "-m", "tools.outputfiles_merge", prefix]
                if not args.keep_outs:
                    merge_cmd.append("--remove-files")
                sh(merge_cmd)
                datafile = f"{prefix}_merged.out"
            else:
                datafile = f"{prefix}.out"

            merged_basename = os.path.basename(datafile)

            # 不在这里画图；只记录
            png_file = ""

            if inpath in scene_records:
                scene_records[inpath].update({
                    'merged_outfile': merged_basename,
                    'png_file': png_file,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                record_scene_params(csv_path, scene_records[inpath])

        if not finished and (todo or active):
            time.sleep(0.1)

    print(f"All scenes finished. Log saved to: {csv_path}")

    # 合并每个 scene 的所有 frame 成一个“大 out”，再调用 gprMax 的可视化脚本
    for i in range(1, args.count + 1):
        big_out = merge_frames_to_big_out(args.outdir, i, args.frames, args.runs)
        if big_out and args.plot:
            sh([sys.executable, "-m", "tools.plot_Bscan_nodirect_1002",
                big_out, args.comp, "--mute_ns", str(args.mute_ns)])

if __name__ == "__main__":
    main()
