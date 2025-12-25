#!/usr/bin/env python3
import argparse
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tools.plot_Bscan import get_output_data

def mute_early_time(outputdata, dt, mute_ns=2.0):
    if dt <= 0:
        return outputdata
    samples_to_mute = int((mute_ns * 1e-9) / dt)
    if samples_to_mute > 0:
        outputdata[:samples_to_mute, :] = 0
    return outputdata

def plot_bscan(out_file, rx_number, rx_component, mute_ns, output_dir):
    outputdata, dt = get_output_data(out_file, rx_number, rx_component)
    outputdata = mute_early_time(outputdata, dt, mute_ns)
    plt.figure(figsize=(10, 6))
    plt.imshow(
        outputdata,
        aspect="auto",
        cmap="gray",
        origin="upper",
        extent=[0, outputdata.shape[1], outputdata.shape[0] * dt * 1e9, 0],
    )
    plt.title(f"B-scan ({os.path.basename(out_file)})\nMuted first {mute_ns} ns")
    plt.xlabel("Trace Number")
    plt.ylabel("Time (ns)")
    plt.colorbar(label="Field Strength")
    plt.tight_layout()
    outfile = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(out_file))[0]}_muted.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile

def main():
    parser = argparse.ArgumentParser(description="Visualize B-scan for all .out files in a folder with early-time muting.")
    parser.add_argument("--folder", type=str, default="/mnt/ssd2T/yang/gprmax/gengpr/genin1jiezhi", help="Folder containing merged .out files (e.g., *_merged.out)")
    parser.add_argument("--pattern", default="*_merged.out", help="Glob pattern for files (default: *_merged.out)")
    parser.add_argument("--rx", type=int, default=1, help="Receiver number (default: 1)")
    parser.add_argument("--component", type=str, default="Ez", help="Field component (default: Ez)")
    parser.add_argument("--mute-ns", type=float, default=2.5, help="Duration to mute in nanoseconds (default: 2.0)")
    parser.add_argument("--output-dir", type=str, default="/mnt/ssd2T/yang/gprmax/gengpr/gentongyijiezhi", help="Directory to save images (default: same as input folder)")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else folder
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(folder, args.pattern)))
    if not files:
        print(f"No files matched pattern '{args.pattern}' in {folder}")
        return

    print(f"Found {len(files)} files. Saving images to: {out_dir}")
    for fpath in files:
        try:
            out_png = plot_bscan(fpath, args.rx, args.component, args.mute_ns, out_dir)
            print(f"Saved: {out_png}")
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

if __name__ == "__main__":
    main()