#!/usr/bin/env python3
"""
河流GPR数据集生成器
用于训练深度学习模型：从B-scan图像预测木头深度和流速

场景设置：
- 固定GPR系统悬浮在河流上方
- 木头在水中以不同深度和速度流动
- 生成B-scan图像和对应的深度/速度标签
"""

import os
import sys
import argparse
import subprocess
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from datetime import datetime
import itertools
from pathlib import Path
from scipy.ndimage import median_filter

def detect_gpu():
    """检测可用GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0:
            gpus = [line for line in result.stdout.strip().split('\n') if 'GPU' in line]
            return 0 if gpus else None
        return None
    except:
        return None

def build_input_template(domain_x=3.0, domain_y=0.8, domain_z=1.0, 
                        water_depth=0.6, dx=0.005):
    """构建gprMax输入文件模板（河流场景）"""
    # GPR系统固定在河流上方
    max_gpr_height = domain_z - water_depth - 0.05  # 留出5cm边界
    gpr_height = min(0.2, max_gpr_height)  # 最多20cm，但不超出域
    ant_x = domain_x / 2  # 河流中央
    ant_y = domain_y / 2  
    ant_z = water_depth + gpr_height
    
    # 单静态配置（发射接收天线在同一位置）
    rx_x, rx_y, rx_z = ant_x, ant_y, ant_z
    
    template = f"""#title: River GPR - Wood Flow Dataset
#domain: {domain_x} {domain_y} {domain_z}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {{time_window}}

#material: 1      0       1 0 air
#material: 2     0.5     1 0 river_water
#material: 2.5    0.002   1 0 dry_wood
#material: 15     0.02    1 0 riverbed_soil

#waveform: ricker 1 {{frequency}} my_ricker
#hertzian_dipole: z {ant_x:.4f} {ant_y:.4f} {ant_z:.4f} my_ricker
#rx: {rx_x:.4f} {rx_y:.4f} {rx_z:.4f}

#box: 0 0 0  {domain_x} {domain_y} 0.05 riverbed_soil
#box: 0 0 0.05  {domain_x} {domain_y} {water_depth} river_water
{{wood_geometry}}

"""
    return template

def generate_wood_geometry(x_center, y_center, z_center, 
                          len_x=0.08, len_y=0.12, len_z=0.05):
    """生成木头几何定义"""
    x0 = x_center - len_x/2
    x1 = x_center + len_x/2
    y0 = y_center - len_y/2
    y1 = y_center + len_y/2
    z0 = z_center - len_z/2
    z1 = z_center + len_z/2
    
    return f"#box: {x0:.4f} {y0:.4f} {z0:.4f}   {x1:.4f} {y1:.4f} {z1:.4f} dry_wood"

def write_input_file(template, output_path, wood_x, wood_depth, 
                    frequency=800e6, time_window=25e-9, wood_params=None):
    """写入单个输入文件"""
    if wood_params is None:
        wood_params = {'y_center': 0.4, 'len_x': 0.08, 'len_y': 0.12, 'len_z': 0.05}
    
    wood_z = 0.05 + wood_depth  # 河床以上的深度
    wood_geom = generate_wood_geometry(
        wood_x, wood_params['y_center'], wood_z,
        wood_params['len_x'], wood_params['len_y'], wood_params['len_z']
    )
    
    content = template.format(
        time_window=time_window,
        frequency=frequency,
        wood_geometry=wood_geom
    )
    
    with open(output_path, 'w') as f:
        f.write(content)
    return True

def run_gprmax_sim(input_file, gpu, timeout=180):
    """运行gprMax模拟"""
    cmd = [sys.executable, "-m", "gprMax", str(input_file)]
    if gpu is not None:
        cmd.extend(["-gpu", str(gpu)])
    
    print(f">> Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True
        else:
            print(f"❌ gprMax failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ Simulation timeout ({timeout}s)")
        return False

def read_output_trace(output_file, component='Ez'):
    """读取GPR输出数据"""
    try:
        with h5py.File(output_file, 'r') as f:
            trace = f['rxs']['rx1'][component][()]
            dt = f.attrs.get('dt', None)
            if dt is None:
                tw = float(f.attrs['time_window'])
                iterations = int(f.attrs['iterations'])
                dt = tw / iterations if iterations > 0 else 1e-9
        return trace.flatten(), dt
    except Exception as e:
        print(f"❌ Error reading {output_file}: {e}")
        return None, None

def save_bscan_image(traces_data, dt, output_png, metadata):
    """保存B-scan图像和元数据"""
    if not traces_data:
        return False
    
    # 组装数据矩阵 (时间 × 位置)
    positions = [item['x_pos'] for item in traces_data]
    traces = [item['trace'] for item in traces_data]
    
    # 确保所有trace长度一致
    min_len = min(len(tr) for tr in traces)
    traces = [tr[:min_len] for tr in traces]
    
    data = np.array(traces).T  # shape: (time_samples, positions)
    
    # 信号处理增强
    # 1. 去直流偏移
    data = data - np.mean(data, axis=0, keepdims=True)
    
    # 2. 增益处理 (时间增益补偿)
    time_samples = np.arange(data.shape[0]).reshape(-1, 1)
    gain_factor = 1 + time_samples * 0.1  # 线性时间增益
    data = data * gain_factor
    
    # 3. 中值滤波去噪
    data = median_filter(data, size=(3, 1))
    
    # 4. 动态范围压缩
    data_abs = np.abs(data)
    vmax = np.percentile(data_abs, 99.5)  # 使用更高百分位数
    if vmax == 0:
        vmax = 1e-6
    
    # 5. 对数压缩增强弱信号
    data_normalized = np.clip(data_abs / vmax, 0, 1)
    data_enhanced = np.sqrt(data_normalized)  # 平方根压缩
    
    # 创建灰度图像
    fig, ax = plt.subplots(figsize=(12, 8))
    
    extent = [min(positions), max(positions), 
              (data.shape[0]-1) * dt * 1e9, 0]  # x=位置(m), y=时间(ns)
    
    # 使用灰度颜色映射，反转颜色（强信号为白色）
    im = ax.imshow(data_enhanced, aspect='auto', cmap='gray_r', 
                   vmin=0, vmax=1, origin='upper', extent=extent, interpolation='bilinear')
    
    ax.set_xlabel('River flow direction (m)', fontsize=12)
    ax.set_ylabel('Two-way travel time (ns)', fontsize=12)
    ax.set_title(f"River GPR B-scan (Grayscale)\\nDepth: {metadata['depth']:.3f}m, "
                f"Velocity: {metadata['velocity']:.3f}m/s", fontsize=14)
    
    # 添加网格线帮助观察
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.colorbar(im, ax=ax, label='Signal Amplitude (Normalized)', shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def generate_parameter_combinations(config):
    """生成参数组合用于数据集"""
    depths = np.linspace(config['depth_min'], config['depth_max'], config['depth_steps'])
    velocities = np.linspace(config['vel_min'], config['vel_max'], config['vel_steps'])
    
    combinations = []
    for depth, vel in itertools.product(depths, velocities):
        combinations.append({
            'depth': depth,
            'velocity': vel,
            'id': len(combinations)
        })
    
    return combinations

def simulate_wood_trajectory(velocity, acceleration, domain_x, n_positions=21):
    """模拟木头轨迹"""
    # 木头从上游进入，流经GPR下方区域
    x_start = 0.2  # 从域的20%位置开始
    x_end = domain_x - 0.2  # 到域的80%位置结束
    
    positions = np.linspace(x_start, x_end, n_positions)
    
    # 计算每个位置对应的时间（考虑加速度）
    if acceleration == 0:
        # 匀速运动
        times = [(x - x_start) / velocity for x in positions]
    else:
        # 加速运动: x = x0 + v0*t + 0.5*a*t^2
        # 求解二次方程得到时间
        times = []
        for x in positions:
            delta_x = x - x_start
            # 0.5*a*t^2 + v0*t - delta_x = 0
            if acceleration > 0:
                discriminant = velocity**2 + 2 * acceleration * delta_x
                t = (-velocity + np.sqrt(discriminant)) / acceleration
            else:
                t = delta_x / velocity
            times.append(t)
    
    return positions, times

def main():
    parser = argparse.ArgumentParser(
        description="Generate river GPR dataset for deep learning training"
    )
    
    # 数据集配置
    parser.add_argument('--output-dir', default='river_gpr_dataset', 
                       help='Output directory for dataset')
    parser.add_argument('--depth-range', nargs=2, type=float, default=[0.1, 0.5],
                       help='Wood depth range (m) [min, max]')
    parser.add_argument('--depth-steps', type=int, default=5,
                       help='Number of depth variations')
    parser.add_argument('--vel-range', nargs=2, type=float, default=[0.5, 2.0],
                       help='Flow velocity range (m/s) [min, max]')
    parser.add_argument('--vel-steps', type=int, default=4,
                       help='Number of velocity variations')
    parser.add_argument('--positions', type=int, default=15,
                       help='Number of wood positions per trajectory')
    
    # 物理参数
    parser.add_argument('--domain-size', nargs=3, type=float, default=[3.0, 0.8, 1.0],
                       help='Simulation domain [x, y, z] (m)')
    parser.add_argument('--water-depth', type=float, default=0.6,
                       help='River water depth (m)')
    parser.add_argument('--grid-size', type=float, default=0.005,
                       help='Grid spacing (m)')
    parser.add_argument('--frequency', type=float, default=800e6,
                       help='GPR frequency (Hz)')
    parser.add_argument('--time-window', type=float, default=25e-9,
                       help='Time window (s)')
    
    # 运行控制
    parser.add_argument('--gpu', default='auto', help='GPU ID or "auto"')
    parser.add_argument('--component', default='Ez', choices=['Ex','Ey','Ez'])
    parser.add_argument('--acceleration', type=float, default=0.1,
                       help='Flow acceleration (m/s²)')
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Maximum samples to generate (0=all)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing progress')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU检测
    gpu = detect_gpu() if args.gpu == 'auto' else int(args.gpu)
    if gpu is not None:
        print(f"✓ Using GPU {gpu}")
    else:
        print("⚠ Using CPU mode")
    
    # 生成参数组合
    config = {
        'depth_min': args.depth_range[0],
        'depth_max': args.depth_range[1], 
        'depth_steps': args.depth_steps,
        'vel_min': args.vel_range[0],
        'vel_max': args.vel_range[1],
        'vel_steps': args.vel_steps
    }
    
    param_combinations = generate_parameter_combinations(config)
    if args.max_samples > 0:
        param_combinations = param_combinations[:args.max_samples]
    
    # 构建输入文件模板
    template = build_input_template(
        domain_x=args.domain_size[0],
        domain_y=args.domain_size[1], 
        domain_z=args.domain_size[2],
        water_depth=args.water_depth,
        dx=args.grid_size
    )
    
    # 数据集记录
    dataset_csv = output_dir / 'dataset_labels.csv'
    metadata_json = output_dir / 'dataset_metadata.json'
    
    # 写入元数据
    metadata = {
        'creation_time': datetime.now().isoformat(),
        'total_samples': len(param_combinations),
        'domain_size': args.domain_size,
        'water_depth': args.water_depth,
        'grid_size': args.grid_size,
        'frequency': args.frequency,
        'time_window': args.time_window,
        'depth_range': args.depth_range,
        'velocity_range': args.vel_range,
        'positions_per_trajectory': args.positions,
        'acceleration': args.acceleration,
        'component': args.component
    }
    
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 初始化CSV文件
    csv_headers = ['sample_id', 'depth_m', 'velocity_m_s', 'bscan_image', 
                   'positions', 'times', 'simulation_time', 'success']
    
    progress_file = output_dir / 'progress.txt'
    start_idx = 0
    
    if args.resume and progress_file.exists():
        start_idx = int(progress_file.read_text().strip())
        print(f"📄 Resuming from sample {start_idx}")
    
    with open(dataset_csv, 'w' if not args.resume else 'a', 
              newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        if not args.resume or start_idx == 0:
            writer.writerow(csv_headers)
        
        print(f"\\n🚀 Generating {len(param_combinations)} GPR dataset samples...")
        print(f"Parameters: depth {args.depth_range[0]}-{args.depth_range[1]}m, "
              f"velocity {args.vel_range[0]}-{args.vel_range[1]}m/s")
        print("=" * 70)
        
        for i, params in enumerate(param_combinations[start_idx:], start_idx):
            depth = params['depth']
            velocity = params['velocity']
            
            print(f"\\n[{i+1:3d}/{len(param_combinations)}] "
                  f"Depth: {depth:.3f}m, Velocity: {velocity:.3f}m/s")
            
            # 计算木头轨迹
            positions, times = simulate_wood_trajectory(
                velocity, args.acceleration, args.domain_size[0], args.positions
            )
            
            # 存储轨迹数据
            traces_data = []
            sim_success = True
            
            for j, (x_pos, time) in enumerate(zip(positions, times)):
                # 生成输入文件
                input_file = output_dir / f'sample_{i:04d}_pos_{j:02d}.in'
                
                success = write_input_file(
                    template, input_file, x_pos, depth,
                    args.frequency, args.time_window
                )
                
                if not success:
                    print(f"❌ Failed to write {input_file}")
                    sim_success = False
                    break
                
                # 运行模拟
                if not run_gprmax_sim(input_file, gpu):
                    print(f"❌ Simulation failed for position {j}")
                    sim_success = False
                    break
                
                # 读取结果
                output_file = input_file.with_suffix('.out')
                trace, dt = read_output_trace(output_file, args.component)
                
                if trace is None:
                    print(f"❌ Failed to read output for position {j}")
                    sim_success = False
                    break
                
                traces_data.append({
                    'x_pos': x_pos,
                    'time': time,
                    'trace': trace
                })
                
                print(f"   ✓ Position {j+1:2d}/{len(positions)}: "
                      f"x={x_pos:.3f}m, t={time:.3f}s")
            
            if sim_success and traces_data:
                # 生成B-scan图像
                bscan_png = output_dir / f'bscan_sample_{i:04d}.png'
                
                bscan_metadata = {
                    'depth': depth,
                    'velocity': velocity,
                    'component': args.component,
                    'sample_id': i
                }
                
                if save_bscan_image(traces_data, dt, bscan_png, bscan_metadata):
                    print(f"   ✓ B-scan saved: {bscan_png.name}")
                else:
                    print("   ❌ Failed to generate B-scan")
                    sim_success = False
            
            # 记录到CSV
            writer.writerow([
                i, f"{depth:.4f}", f"{velocity:.4f}", 
                f"bscan_sample_{i:04d}.png" if sim_success else "",
                len(positions), f"{times[-1]:.4f}",
                datetime.now().isoformat(timespec='seconds'),
                sim_success
            ])
            
            # 更新进度
            progress_file.write_text(str(i + 1))
            
            # 清理中间文件（可选）
            if sim_success:
                for j in range(len(positions)):
                    temp_in = output_dir / f'sample_{i:04d}_pos_{j:02d}.in'
                    temp_out = output_dir / f'sample_{i:04d}_pos_{j:02d}.out'
                    if temp_in.exists():
                        temp_in.unlink()
                    if temp_out.exists():
                        temp_out.unlink()
            
            status = "✅ SUCCESS" if sim_success else "❌ FAILED"
            print(f"   {status}")
    
    print(f"\\n🎉 Dataset generation complete!")
    print(f"📊 Total samples: {len(param_combinations)}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Labels file: {dataset_csv}")
    print(f"⚙️  Metadata file: {metadata_json}")

if __name__ == '__main__':
    main()