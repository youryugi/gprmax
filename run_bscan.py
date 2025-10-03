#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, subprocess, sys
import os

def sh(cmd):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    ap = argparse.ArgumentParser(description="Run gprMax once (-n N), merge, plot PNG using official plot_Bscan.")
    ap.add_argument("--infile", required=True, help=".in file (contains #src_steps/#rx_steps)")
    ap.add_argument("--runs", type=int, required=True, help="N for -n (e.g., 60)")
    ap.add_argument("--gpu", type=int, help="GPU id")
    ap.add_argument("--comp", default="Ez", help="field component, default Ez")
    args = ap.parse_args()

    # 1) simulate once with -n
    cmd = [sys.executable, "-m", "gprMax", args.infile, "-n", str(args.runs)]
    if args.gpu is not None: 
        cmd += ["-gpu", str(args.gpu)]
    sh(cmd)

    # 2) merge (prefix = infile without extension)
    prefix = args.infile.rsplit(".",1)[0]
    sh([sys.executable, "-m", "tools.outputfiles_merge", prefix, "--remove-files"])
    merged = f"{prefix}_merged.out"

    # 3) plot using official plot_Bscan
    print(f"Plotting B-scan using official plot_Bscan.py...")
    sh([sys.executable, "-m", "tools.plot_Bscan", merged, args.comp])

if __name__ == "__main__":
    main()
