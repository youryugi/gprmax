#!/usr/bin/env python3
"""
专门设计用于生成清晰双曲线反射的GPR测试
关键改进：
1. 固定木头位置，移动天线（模拟B-scan采集）
2. 更强的介电常数对比
3. 优化的频率和网格设置
"""

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import median_filter

def create_input_file(output_path, antenna_x, wood_x=1.0, wood_depth=0.25):
    """创建单次GPR输入文件 - 天线移动，木头固定"""
    
    content = f"""#title: Hyperbola Test - Moving Antenna over Fixed Wood
#domain: 2.5 0.8 1.2
#dx_dy_dz: 0.004 0.004 0.004
#time_window: 50e-9

#material: 1      0       1 0 air
#material: 81     0.5     1 0 water  
#material: 2      0.001   1 0 dry_wood
#material: 20     0.05    1 0 riverbed

#waveform: ricker 1 150e6 my_pulse
#hertzian_dipole: z {antenna_x:.4f} 0.4000 0.7500 my_pulse
#rx: {antenna_x:.4f} 0.4000 0.7500

#box: 0 0 0  2.5 0.8 0.6000 water
#box: 0 0 0  2.5 0.8 0.0500 riverbed
#box: {wood_x-0.1:.4f} 0.3000 {0.6-wood_depth-0.05:.4f}   {wood_x+0.1:.4f} 0.5000 {0.6-wood_depth+0.05:.4f} dry_wood
"""

    with open(output_path, 'w') as f:
        f.write(content)
    print(f"✓ Created input: {output_path}, antenna at x={antenna_x:.3f}m")

def run_gprmax(input_file, gpu=0):
    """运行GPRMax模拟"""
    cmd = [sys.executable, "-m", "gprMax", input_file, "-gpu", str(gpu)]
    print(">> " + " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        print(f"❌ gprMax failed: {result.stderr}")
        return False
    return True

def read_trace(output_file):
    """读取A-scan数据"""
    try:
        with h5py.File(output_file, 'r') as f:
            trace = f['rxs']['rx1']['Ez'][()]
            dt = f.attrs.get('dt', None)
            if dt is None:
                dt = float(f.attrs['time_window']) / int(f.attrs['iterations'])
        return trace.flatten(), dt
    except Exception as e:
        print(f"❌ Error reading {output_file}: {e}")
        return None, None

def create_hyperbola_bscan():
    """生成真正的双曲线B-scan"""
    
    # 天线位置序列 (从x=0.5m 到 x=1.5m, 木头在x=1.0m)
    antenna_positions = np.linspace(0.5, 1.5, 5)  # 5个位置快速测试
    wood_x = 1.0  # 木头固定在x=1.0m
    wood_depth = 0.25  # 水面下25cm
    
    output_dir = "hyperbola_test_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    traces_data = []
    
    print(f"\n🎯 生成双曲线B-scan: 木头@x={wood_x}m, 深度={wood_depth}m")
    print("="*60)
    
    for i, ant_x in enumerate(antenna_positions):
        input_file = f"{output_dir}/scan_{i:02d}.in"
        output_file = f"{output_dir}/scan_{i:02d}.out"
        
        # 创建输入文件
        create_input_file(input_file, ant_x, wood_x, wood_depth)
        
        # 运行模拟
        if not run_gprmax(input_file, gpu=0):
            print(f"❌ Failed at position {i+1}/{len(antenna_positions)}")
            continue
            
        # 读取数据
        trace, dt = read_trace(output_file)
        if trace is not None:
            traces_data.append({
                'antenna_x': ant_x,
                'trace': trace,
                'distance_to_wood': abs(ant_x - wood_x)
            })
            print(f"✓ Position {i+1:2d}/{len(antenna_positions)}: "
                  f"x={ant_x:.3f}m, dist_to_target={abs(ant_x-wood_x):.3f}m")
    
    if not traces_data:
        print("❌ No valid traces collected")
        return
    
    # 生成B-scan图像
    print("\n📊 Generating hyperbola B-scan...")
    create_enhanced_bscan(traces_data, dt, f"{output_dir}/hyperbola_bscan.png", 
                         wood_x, wood_depth)

def create_enhanced_bscan(traces_data, dt, output_png, wood_x, wood_depth):
    """创建增强的双曲线B-scan图像"""
    
    antenna_positions = [item['antenna_x'] for item in traces_data]
    traces = [item['trace'] for item in traces_data]
    
    # 数据预处理
    min_len = min(len(tr) for tr in traces)
    traces = [tr[:min_len] for tr in traces]
    data = np.array(traces).T  # shape: (time, position)
    
    # 信号增强处理
    data = data - np.mean(data, axis=0, keepdims=True)  # 去直流
    
    # 时间增益补偿
    time_gain = np.arange(data.shape[0]).reshape(-1, 1) * 0.15 + 1
    data = data * time_gain
    
    # 滤波
    data = median_filter(data, size=(3, 1))
    
    # 归一化
    data_abs = np.abs(data)
    vmax = np.percentile(data_abs, 99.8)
    data_norm = np.clip(data_abs / vmax, 0, 1)
    data_enhanced = np.sqrt(data_norm)  # 平方根增强
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # B-scan图像
    extent = [min(antenna_positions), max(antenna_positions), 
              (data.shape[0]-1) * dt * 1e9, 0]
    
    im = ax1.imshow(data_enhanced, aspect='auto', cmap='gray_r', 
                   vmin=0, vmax=1, origin='upper', extent=extent, 
                   interpolation='bilinear')
    
    ax1.axvline(x=wood_x, color='red', linestyle='--', alpha=0.7, 
                label=f'Wood position (x={wood_x}m)')
    ax1.set_xlabel('Antenna position (m)')
    ax1.set_ylabel('Two-way travel time (ns)')
    ax1.set_title(f'GPR B-scan: Hyperbola over Wood Target\\n'
                 f'Wood at x={wood_x}m, depth={wood_depth}m below water surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax1, label='Normalized amplitude')
    
    # 理论双曲线叠加
    water_velocity = 3e8 / np.sqrt(81)  # 水中电磁波速度
    antenna_x_theory = np.linspace(min(antenna_positions), max(antenna_positions), 100)
    
    # 双曲线方程: t = sqrt(t0^2 + (4*dx^2/v^2))
    t0_wood = 2 * wood_depth / water_velocity * 1e9  # 垂直往返时间(ns)
    dx = antenna_x_theory - wood_x  # 水平距离
    t_hyperbola = np.sqrt(t0_wood**2 + (2*dx/water_velocity*1e9)**2)
    
    ax1.plot(antenna_x_theory, t_hyperbola, 'cyan', linewidth=2, 
             label=f'Theoretical hyperbola (t0={t0_wood:.1f}ns)')
    ax1.legend()
    ax1.set_ylim([0, 40])  # 限制显示范围
    
    # 中心A-scan对比
    center_idx = len(antenna_positions) // 2
    center_trace = traces[center_idx]
    time_ns = np.arange(len(center_trace)) * dt * 1e9
    
    ax2.plot(time_ns, center_trace, 'b-', linewidth=1, 
             label=f'A-scan at x={antenna_positions[center_idx]:.2f}m')
    ax2.axvline(x=t0_wood, color='red', linestyle='--', 
                label=f'Expected wood reflection ({t0_wood:.1f}ns)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (V/m)')
    ax2.set_title('Central A-scan')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 40])
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Hyperbola B-scan saved: {output_png}")
    print(f"📐 Theoretical wood reflection time: {t0_wood:.2f} ns")
    print(f"🌊 Water wave velocity: {water_velocity:.0f} m/s")

if __name__ == "__main__":
    create_hyperbola_bscan()