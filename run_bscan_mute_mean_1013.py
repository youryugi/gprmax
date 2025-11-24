#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys
import os

def sh(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    return r

def main():
    ap = argparse.ArgumentParser(description="Run gprMax (-n), merge, then plot (DC-mean only, optional envelope).")
    ap.add_argument("--infile", default="t1010/test8.in", help=".in file")
    ap.add_argument("--runs", type=int, default=200, help="N for -n")
    ap.add_argument("--gpu", type=int, default=0, help="GPU id")
    ap.add_argument("--comp", default="Ez", help="field component, default Ez")
    # 使用已实现“去直流/可选包络”的绘图脚本
    ap.add_argument("--plot_module", default="tools.plot_Bscan_nodirect_mean_1013",
                    help="Python module to plot B-scan")
    ap.add_argument("--envelope", action="store_true", help="Use Hilbert envelope in plot (if supported)")
    args = ap.parse_args()

    # 无显示环境时使用无头后端
    if not os.environ.get("DISPLAY"):
        os.environ["MPLBACKEND"] = "Agg"

    # 1) simulate
    sim = [sys.executable, "-m", "gprMax", args.infile, "-n", str(args.runs)]
    if args.gpu is not None:
        sim += ["-gpu", str(args.gpu)]
    sh(sim)

    # 2) merge
    prefix = args.infile.rsplit(".", 1)[0]
    sh([sys.executable, "-m", "tools.outputfiles_merge", prefix, "--remove-files"])
    merged = f"{prefix}_merged.out"

    # 3) plot（不再传 --mute_ns）
    plot_cmd = [sys.executable, "-m", args.plot_module, merged, args.comp]
    if args.envelope:
        plot_cmd.append("--envelope")
    sh(plot_cmd)

if __name__ == "__main__":
    main()
