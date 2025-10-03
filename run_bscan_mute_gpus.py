#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys
import os, re

def sh(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    return r

def get_mute_ns_from_in(infile:str):
    # 调 tools.mute 解析推荐的 --mute_ns 数值
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

def main():
    ap = argparse.ArgumentParser(description="Run gprMax (-n), merge, plot PNG with mute.")
    ap.add_argument("--infile", default="t1003/test7.in", help=".in file (contains #src_steps/#rx_steps)")
    ap.add_argument("--runs", default=60,type=int, help="N for -n (e.g., 60)")
    ap.add_argument("--gpu", default=0,type=int, help="GPU id")
    ap.add_argument("--comp", default="Ez", help="field component, default Ez")
    ap.add_argument("--mute_ns", type=float, default=None, help="override mute window (ns); if omitted, auto-compute")
    args = ap.parse_args()

    # 1) simulate
    cmd = [sys.executable, "-m", "gprMax", args.infile, "-n", str(args.runs)]
    if args.gpu is not None:
        cmd += ["-gpu", str(args.gpu)]
    sh(cmd)

    # 2) merge (and remove individuals)
    prefix = args.infile.rsplit(".",1)[0]
    sh([sys.executable, "-m", "tools.outputfiles_merge", prefix, "--remove-files"])
    merged = f"{prefix}_merged.out"

    # 3) compute/use mute_ns
    mute_ns = args.mute_ns
    if mute_ns is None:
        mute_ns = get_mute_ns_from_in(args.infile)
        if mute_ns is not None:
            print(f"Auto mute_ns (ns): {mute_ns:.3f}")
        else:
            print("Warning: failed to auto-compute mute_ns; plotting without mute.")

    # 4) plot with custom script that支持 --mute_ns，并将PNG保存到out同目录
    print("Plotting B-scan with mute...")
    plot_cmd = [sys.executable, "-m", "tools.plot_Bscan_nodirect_1002", merged, args.comp]
    if mute_ns is not None:
        plot_cmd += ["--mute_ns", f"{mute_ns:.3f}"]
    sh(plot_cmd)

if __name__ == "__main__":
    main()
