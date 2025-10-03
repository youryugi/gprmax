#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys, os, time
from collections import deque

def sh(cmd):
    print(">>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

def run_tasks_on_gpu(pyexe, infile, total_runs, gpu_id, r1, r2):
    # 逐个 task 执行，避免使用不存在的 -r 选项
    for i in range(r1, r2 + 1):
        cmd = [pyexe, "-m", "gprMax", infile, "-n", str(total_runs), "-task", str(i), "-gpu", str(gpu_id)]
        print(f"GPU {gpu_id}: running task {i}/{total_runs}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            return r.returncode
    return 0

def launch_chunk_worker(pyexe, script_path, infile, total_runs, r1, r2, gpu_id):
    cmd = [pyexe, script_path, "--worker",
           "--infile", infile,
           "--runs", str(total_runs),
           "--gpu", str(gpu_id),
           "--range", str(r1), str(r2)]
    print(f"Launching GPU {gpu_id}: tasks {r1}-{r2}")
    return subprocess.Popen(cmd)

def make_chunks(N, chunk_size):
    chunks = []
    s = 1
    while s <= N:
        e = min(N, s + chunk_size - 1)
        chunks.append((s, e))
        s = e + 1
    return deque(chunks)

def main():
    ap = argparse.ArgumentParser(description="Split -n runs across multiple GPUs (dynamic chunk scheduling), then merge and plot.")
    ap.add_argument("--infile", required=True, help=".in file")
    ap.add_argument("--runs", type=int, required=True, help="Total runs for -n (e.g., 60)")
    ap.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids, e.g. '0,1'")
    ap.add_argument("--chunk-size", type=int, default=10, help="Runs per chunk")
    ap.add_argument("--comp", default="Ez", help="Field component, e.g. Ez")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", help=argparse.SUPPRESS)
    ap.add_argument("--range", nargs=2, type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    pyexe = sys.executable

    # 子进程模式：在指定 GPU 上跑一个区间的 tasks
    if args.worker:
        if not (args.gpu and args.range):
            print("Worker missing --gpu/--range")
            sys.exit(2)
        r1, r2 = args.range
        rc = run_tasks_on_gpu(pyexe, args.infile, args.runs, args.gpu, r1, r2)
        sys.exit(rc)

    # 主调度进程
    N = args.runs
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    if not gpus:
        print("No GPUs specified.")
        sys.exit(1)

    chunks = make_chunks(N, max(1, args.chunk_size))
    active = {}
    script_path = os.path.abspath(__file__)

    # 先给每块 GPU 派一个 chunk
    for gpu in gpus:
        if not chunks:
            break
        r1, r2 = chunks.popleft()
        active[gpu] = (launch_chunk_worker(pyexe, script_path, args.infile, N, r1, r2, gpu), (r1, r2))

    # 动态调度：哪个 GPU 先空闲就继续派下一个 chunk
    rc = 0
    while active:
        finished = []
        for gpu, (proc, rr) in list(active.items()):
            ret = proc.poll()
            if ret is not None:
                rc |= (ret or 0)
                finished.append(gpu)
        for gpu in finished:
            del active[gpu]
            if chunks:
                r1, r2 = chunks.popleft()
                active[gpu] = (launch_chunk_worker(pyexe, script_path, args.infile, N, r1, r2, gpu), (r1, r2))
        if active:
            time.sleep(0.2)

    if rc != 0:
        print("Some chunks failed.")
        sys.exit(rc)

    # 合并并画图（PNG 将保存到 .out/.in 同目录）
    prefix = args.infile.rsplit(".", 1)[0]
    sh([pyexe, "-m", "tools.outputfiles_merge", prefix, "--remove-files"])
    merged = f"{prefix}_merged.out"
    print("Plotting B-scan using official plot_Bscan.py...")
    sh([pyexe, "-m", "tools.plot_Bscan", merged, args.comp])

if __name__ == "__main__":
    main()
# 用法:
# python run_bscan22.py --infile waterwood/test.in --runs 60 --gpus 0,1 --chunk-size 8 --comp Ez