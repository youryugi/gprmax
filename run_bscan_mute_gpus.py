#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys
import os, re, time, glob, shutil
from collections import deque

def sh(cmd, cwd=None):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    return r

def get_mute_ns_from_in(infile:str):
    r = subprocess.run([sys.executable, "-m", "tools.mute", "--infile", infile],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("tools.mute failed:", r.stderr.strip())
        return None
    out = r.stdout
    m = re.search(r'--mute_ns\s+([0-9.]+)', out)
    if m:
        return float(m.group(1))
    m = re.search(r'MUTE window \[ns\]\s*:\s*\[[^,]+,\s*([0-9.]+)\]', out)
    if m:
        return float(m.group(1))
    return None

_float = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'

def parse_positions_and_steps(in_text:str):
    # hertzian_dipole
    m_tx = re.search(rf'^\s*#hertzian_dipole:\s*(\w)\s+({_float})\s+({_float})\s+({_float})\s+(\S+)\s*$',
                     in_text, re.M)
    if not m_tx:
        raise ValueError("Cannot find #hertzian_dipole")
    pol, tx0x, tx0y, tx0z, wf = m_tx.groups()
    tx0 = (float(tx0x), float(tx0y), float(tx0z))

    # rx
    m_rx = re.search(rf'^\s*#rx:\s*({_float})\s+({_float})\s+({_float})\s*$',
                     in_text, re.M)
    if not m_rx:
        raise ValueError("Cannot find #rx")
    rx0 = tuple(map(float, m_rx.groups()))

    # steps
    m_s = re.search(rf'^\s*#src_steps:\s*({_float})\s+({_float})\s+({_float})\s*$', in_text, re.M)
    m_r = re.search(rf'^\s*#rx_steps:\s*({_float})\s+({_float})\s+({_float})\s*$', in_text, re.M)
    if not (m_s and m_r):
        raise ValueError("Cannot find #src_steps/#rx_steps")
    src_step = tuple(map(float, m_s.groups()))
    rx_step  = tuple(map(float, m_r.groups()))

    return dict(pol=pol, wf=wf, tx0=tx0, rx0=rx0, src_step=src_step, rx_step=rx_step)

def format_vec(v):
    return f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}"

def make_in_for_run(in_text:str, meta:dict, run_idx:int):
    # 计算该步的绝对坐标
    k = run_idx - 1
    tx = tuple(meta['tx0'][i] + k * meta['src_step'][i] for i in range(3))
    rx = tuple(meta['rx0'][i] + k * meta['rx_step'][i] for i in range(3))

    # 替换 tx/rx 行
    new_text = re.sub(
        rf'^\s*#hertzian_dipole:.*$',
        f"#hertzian_dipole: {meta['pol']} {format_vec(tx)} {meta['wf']}",
        in_text, flags=re.M
    )
    new_text = re.sub(
        rf'^\s*#rx:.*$',
        f"#rx: {format_vec(rx)}",
        new_text, flags=re.M
    )
    # 移除步进（避免再被引擎步进）
    new_text = re.sub(r'^\s*#src_steps:.*$\n?', '', new_text, flags=re.M)
    new_text = re.sub(r'^\s*#rx_steps:.*$\n?',  '', new_text, flags=re.M)

    # 确保只运行一次
    # （我们用 -n 1 启动）
    return new_text

def run_one_step(temp_in:str, gpu_id:int):
    # gprMax -n 1 默认输出为 os.path.splitext(temp_in)[0] + ".out"
    cmd = [sys.executable, "-m", "gprMax", temp_in, "-n", "1", "-gpu", str(gpu_id)]
    print(f"Launching run on GPU {gpu_id}: {os.path.basename(temp_in)}")
    p = subprocess.Popen(cmd)
    return p

def main():
    ap = argparse.ArgumentParser(description="Multi-GPU scheduler: split steps, run concurrently, merge, plot with mute.")
    ap.add_argument("--infile", default="t1003/test7.in", help=".in file (contains #src_steps/#rx_steps)")
    ap.add_argument("--runs", default=300,type=int, help="N for -n (e.g., 60)")
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU IDs, e.g. 0,1,2,3")
    ap.add_argument("--comp", default="Ez", help="field component")
    ap.add_argument("--keep-outs", action="store_true", help="keep individual .out files (skip deletion)")
    ap.add_argument("--mute_ns", type=float, default=None, help="override mute window (ns); else auto")
    args = ap.parse_args()

    infile = args.infile
    base_prefix = os.path.splitext(infile)[0]  # e.g. path/prefix
    in_dir = os.path.dirname(infile) or "."
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    gpus = [int(g) for g in gpus]
    if not gpus:
        print("No GPUs specified.")
        sys.exit(1)

    # 读取并解析原 in
    with open(infile, "r", encoding="utf-8") as f:
        in_text = f.read()
    meta = parse_positions_and_steps(in_text)

    # 任务队列
    todo = deque(range(1, args.runs + 1))
    # 改为 pid -> item 的字典，便于 waitpid 回收
    active = {}  # pid -> dict(i, gpu, proc, temp_in, default_out, target_out)

    # 循环调度
    while todo or active:
        # 计算空闲GPU
        busy = {it["gpu"] for it in active.values()}
        free = [g for g in gpus if g not in busy]

        # 仅把任务派发到空闲GPU
        while todo and free:
            i = todo.popleft()
            gpu = free.pop(0)
            txt = make_in_for_run(in_text, meta, i)
            temp_in = os.path.join(in_dir, f"{os.path.basename(base_prefix)}__step{i}.in")
            with open(temp_in, "w", encoding="utf-8") as f:
                f.write(txt)

            target_prefix = f"{base_prefix}{i}"                 # e.g. t1003/test7 + 1
            default_out = f"{os.path.splitext(temp_in)[0]}.out"  # test7__step{i}.out
            p = run_one_step(temp_in, gpu)
            active[p.pid] = dict(i=i, gpu=gpu, proc=p, temp_in=temp_in,
                                  default_out=default_out, target_out=f"{target_prefix}.out")

        # 使用 waitpid 非阻塞回收所有已完成的子进程
        finished_any = False
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                # 没有子进程
                break
            if pid == 0:
                # 当前无已完成进程
                break
            finished_any = True
            item = active.pop(pid, None)
            if not item:
                continue

            temp_in = item["temp_in"]
            default_out = item["default_out"]
            target_out = item["target_out"]

            if os.path.exists(default_out):
                os.replace(default_out, target_out)
            elif not os.path.exists(target_out):
                print(f"Error: expected output not found: {target_out} (missing default {default_out})")
                sys.exit(2)

            os.remove(temp_in)
            print(f"[GPU {item['gpu']}] step {item['i']} done -> {os.path.basename(target_out)}")

        # 若本轮没有任务完成，短暂让出 CPU
        if not finished_any and (todo or active):
            time.sleep(0.02)

    # 合并
    merge_cmd = [sys.executable, "-m", "tools.outputfiles_merge", base_prefix]
    if not args.keep_outs:
        merge_cmd.append("--remove-files")
    sh(merge_cmd)
    merged = f"{base_prefix}_merged.out"

    # 计算/应用 mute
    mute_ns = args.mute_ns
    if mute_ns is None:
        mute_ns = get_mute_ns_from_in(infile)
        if mute_ns:
            print(f"Auto mute_ns (ns): {mute_ns:.3f}")
        else:
            print("Warn: cannot auto-compute mute_ns; plotting without mute.")

    # 绘图（优先用自定义，失败则回退官方）
    plot_mod = "tools.plot_Bscan_nodirect_1002"  # 需你实现 --mute_ns 支持
    plot_cmd = [sys.executable, "-m", plot_mod, merged, args.comp]
    if mute_ns:
        plot_cmd += ["--mute_ns", f"{mute_ns:.3f}"]
    try:
        sh(plot_cmd)
    except SystemExit:
        print("Fallback to official tools.plot_Bscan (no mute).")
        sh([sys.executable, "-m", "tools.plot_Bscan", merged, args.comp])

if __name__ == "__main__":
    main()
