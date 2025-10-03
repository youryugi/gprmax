#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute recommended early-time mute window from a gprMax .in file.

Usage:
  python -m tools.compute_mute_from_in path/to/model.in \
      [--k 0.8] [--epsilon 1.0] [--fc 1.5e9] [--dt 1e-11]

Outputs:
  - Tx/Rx distance, inferred epsilon_r and wave velocity
  - Direct-wave arrival time t_direct (ns)
  - Ricker main-lobe estimate (1/fc) (ns)
  - Recommended mute window [0, t_end] (ns) and (optional) samples if dt provided
"""
import argparse, math, os, re
from typing import Dict, Tuple, Optional

CMD = re.compile(r'^\s*#\s*([a-zA-Z_]+)\s*:\s*(.*)$')

def parse_infile(path:str):
    materials: Dict[str, float] = {}   # name -> epsr
    boxes = []  # list of (x1,y1,z1,x2,y2,z2,material_name)
    waveform = None  # ('ricker', amplitude, fc, name)
    tx = None  # (x,y,z)
    rx = None  # (x,y,z)

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            m = CMD.match(raw)
            if not m:
                continue
            key = m.group(1).strip().lower()
            args = m.group(2).strip()
            toks = args.split()

            if key == 'material' and len(toks) >= 5:
                # #material: epsr sigma mur kappa name
                try:
                    epsr = float(toks[0])
                    name = toks[5] if len(toks) >= 6 else toks[4]
                except Exception:
                    # Some files may omit kappa; fallback: last token is name
                    epsr = float(toks[0])
                    name = toks[-1]
                materials[name] = epsr

            elif key == 'box' and len(toks) >= 7:
                # #box: x1 y1 z1 x2 y2 z2 material
                x1,y1,z1,x2,y2,z2 = map(float, toks[:6])
                matname = toks[6]
                # normalize order
                xlo,xhi = sorted([x1,x2]); ylo,yhi = sorted([y1,y2]); zlo,zhi = sorted([z1,z2])
                boxes.append((xlo,ylo,zlo,xhi,yhi,zhi,matname))

            elif key == 'waveform' and len(toks) >= 4:
                # #waveform: ricker amp fc name
                wtype = toks[0].lower()
                if wtype == 'ricker':
                    try:
                        amp = float(toks[1]); fc = float(toks[2]); wname = toks[3]
                        waveform = ('ricker', amp, fc, wname)
                    except Exception:
                        pass

            elif key == 'hertzian_dipole' and len(toks) >= 5:
                # #hertzian_dipole: pol x y z name
                try:
                    x = float(toks[1]); y = float(toks[2]); z = float(toks[3])
                    tx = (x,y,z)
                except Exception:
                    pass

            elif key == 'rx' and len(toks) >= 3:
                try:
                    x = float(toks[0]); y = float(toks[1]); z = float(toks[2])
                    rx = (x,y,z)
                except Exception:
                    pass

    return materials, boxes, waveform, tx, rx

def point_in_box(p, box)->bool:
    x,y,z = p
    xlo,ylo,zlo,xhi,yhi,zhi,_ = box
    # include boundary
    return (xlo <= x <= xhi) and (ylo <= y <= yhi) and (zlo <= z <= zhi)

def infer_epsr_for_txrx(materials, boxes, tx, rx, default_epsr:float=1.0)->float:
    # If both points fall into the SAME box, use that material's epsr if available.
    box_tx = None; box_rx = None
    for b in boxes:
        if point_in_box(tx, b):
            box_tx = b
        if point_in_box(rx, b):
            box_rx = b
    if box_tx is not None and box_rx is not None and box_tx == box_rx:
        mat = box_tx[6]
        if mat in materials:
            return materials[mat]
        # If the material name is literally 'air' or similar, fallback to 1.0
        if mat.lower() in ('air','vacuum'):
            return 1.0
    # Otherwise, fallback to default (air)
    return default_epsr

def compute_mute(epsr: float, tx:Tuple[float,float,float], rx:Tuple[float,float,float],
                 fc: float, k: float, pulse_factor: float = 1.0):
    """
    epsr: relative permittivity
    fc: center frequency (Hz)
    k: multiplier for main-lobe width (mute_end = t_direct + k*(1/fc))
    pulse_factor: scale main-lobe width; for Ricker, ~1/fc is a decent proxy.
    """
    c = 299792458.0
    v = c / math.sqrt(max(epsr, 1e-9))
    dx = rx[0]-tx[0]; dy = rx[1]-tx[1]; dz = rx[2]-tx[2]
    d = math.sqrt(dx*dx + dy*dy + dz*dz)
    t_direct = d / v  # seconds
    main_lobe = pulse_factor / fc  # seconds, crude estimate
    t_end = t_direct + k * main_lobe
    return d, v, t_direct, main_lobe, t_end

def main():
    ap = argparse.ArgumentParser(description="Compute early-time mute window from a gprMax .in file.")
    ap.add_argument('--infile', help='.in file path')
    ap.add_argument('--epsilon', type=float, default=None, help='Override epsilon_r (default: inferred or 1.0)')
    ap.add_argument('--fc', type=float, default=None, help='Override center frequency in Hz (default: from waveform)')
    ap.add_argument('--k', type=float, default=0.8, help='Mute tail length multiplier for (1/fc). Default 0.8')
    ap.add_argument('--pulse_factor', type=float, default=1.0, help='Scale of main-lobe ~ pulse_factor/fc (Ricker≈1/fc)')
    ap.add_argument('--dt', type=float, default=None, help='Optional sampling interval (s) to output sample counts')
    args = ap.parse_args()

    materials, boxes, waveform, tx, rx = parse_infile(args.infile)

    if tx is None or rx is None:
        raise SystemExit("ERROR: Cannot find Tx (#hertzian_dipole) or Rx (#rx) in the .in file.")

    # center frequency
    if args.fc is not None:
        fc = args.fc
    else:
        if waveform is None:
            raise SystemExit("ERROR: No waveform found. Please pass --fc.")
        if waveform[0] != 'ricker':
            print("Warning: waveform is not Ricker. Using its frequency as fc anyway.")
        fc = waveform[2]

    # epsilon_r
    if args.epsilon is not None:
        epsr = args.epsilon
    else:
        epsr = infer_epsr_for_txrx(materials, boxes, tx, rx, default_epsr=1.0)

    d, v, t_direct, main_lobe, t_end = compute_mute(epsr, tx, rx, fc, k=args.k, pulse_factor=args.pulse_factor)

    def ns(x): return x * 1e9

    print("=== Early-time mute recommendation ===")
    print(f"Infile            : {os.path.abspath(args.infile)}")
    print(f"Tx (m)            : ({tx[0]:.6f}, {tx[1]:.6f}, {tx[2]:.6f})")
    print(f"Rx (m)            : ({rx[0]:.6f}, {rx[1]:.6f}, {rx[2]:.6f})")
    print(f"Distance d        : {d:.6f} m")
    print(f"Inferred epsilon_r: {epsr:.4f}")
    print(f"Wave velocity v   : {v:.3e} m/s")
    print(f"Center freq fc    : {fc:.3e} Hz")
    print(f"Direct time t0    : {ns(t_direct):.3f} ns")
    print(f"Main lobe ~1/fc   : {ns(main_lobe):.3f} ns  (pulse_factor={args.pulse_factor})")
    print(f"MUTE window [ns]  : [0.000, {ns(t_end):.3f}]  (k={args.k})")

    if args.dt is not None and args.dt > 0:
        n_end = int(round(t_end / args.dt))
        print(f"With dt={args.dt:.3e} s, mute samples: n=0..{n_end} (inclusive)")

    print("\nSuggested flag for your plot script:")
    print(f"  --mute_ns {ns(t_end):.3f}")

if __name__ == '__main__':
    main()
