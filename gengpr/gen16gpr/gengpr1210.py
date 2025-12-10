import os
import sys
import subprocess
from gprMax.gprMax import api
from tools.plot_Bscan import get_output_data, mpl_plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 场景参数（统一等高、等间距）
y = 0.60       # 同一深度
r = 0.05       # 同一半径
x0 = 0.40      # 起始 x
dx = 0.40      # 间距
centers = [x0 + i * dx for i in range(4)]  # 左到右 4 个位置

def write_infile(input_file, present_bits):
    with open(input_file, 'w') as f:
        # 基础域与材料
        f.write("#domain: 2.0 1.0 0.002\n")
        f.write("#dx_dy_dz: 0.002 0.002 0.002\n")
        f.write("#time_window: 12e-9\n")
        f.write("#material: 3 0.001 1 0 soil\n")
        f.write("#material: 1 1e8 1 0 metal\n")
        f.write("#box: 0 0 0 2.0 1.0 0.002 soil\n")

        # 4 球：根据位掩码决定是否写入
        # 位序：bit3 bit2 bit1 bit0 对应从左到右 sphere0..sphere3
        for i, cx in enumerate(centers):
            present = (present_bits >> (3 - i)) & 1  # 左到右
            if present:
                f.write(f"#cylinder: {cx:.3f} {y:.3f} 0  {cx:.3f} {y:.3f} 0.002  {r:.3f} metal\n")

        # 源/接收与扫描
        f.write("#waveform: ricker 1 1.5e9 my_ricker\n")
        f.write("#hertzian_dipole: z 0.10 0.90 0 my_ricker\n")
        f.write("#rx: 0.14 0.90 0\n")
        f.write("#src_steps: 0.02 0 0\n")
        f.write("#rx_steps: 0.02 0 0\n")

def run_case(bitstr, n_traces=85, gpu_id=0):
    filename = f"simulation_4spheres_{bitstr}"
    input_file = f"{filename}.in"
    # 写入 .in
    write_infile(input_file, int(bitstr, 2))
    print(f"Input file '{input_file}' created for case {bitstr}")

    # 运行仿真
    print("Running simulation...")
    api(input_file, n=n_traces, geometry_only=False, gpu=[gpu_id])

    # 合并输出
    print("Merging outputs...")
    subprocess.run([sys.executable, "-m", "tools.outputfiles_merge", filename, "--remove-files"], check=True)
    merged_out = f"{filename}_merged.out"

    # 绘图保存
    print(f"Processing output: {merged_out}")
    try:
        rx_number = 1
        rx_component = 'Ez'
        outputdata, dt = get_output_data(merged_out, rx_number, rx_component)
        plt.figure(figsize=(10, 6))
        plt.imshow(outputdata, aspect='auto', cmap='gray',
                   origin='upper',
                   extent=[0, outputdata.shape[1], outputdata.shape[0] * dt * 1e9, 0])
        plt.title(f'B-scan ({bitstr})')
        plt.xlabel('Trace Number (Scan position)')
        plt.ylabel('Time (ns)')
        plt.colorbar(label='Field Strength')
        image_filename = f"bscan_{bitstr}.png"
        plt.savefig(image_filename, dpi=150)
        plt.close()
        print(f"B-scan image saved to {image_filename}")
    except Exception as e:
        print(f"Error processing output ({bitstr}): {e}")

if __name__ == "__main__":
    # 生成 16 种 4-bit 组合：0000 到 1111
    cases = [format(i, "04b") for i in range(16)]
    # 可按需改变 GPU 或 n
    gpu_id = 0
    n_traces = 85
    for bitstr in cases:
        run_case(bitstr, n_traces=n_traces, gpu_id=gpu_id)