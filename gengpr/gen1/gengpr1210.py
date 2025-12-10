import os
import sys
from gprMax.gprMax import api
from tools.plot_Bscan import get_output_data, mpl_plot

# 1. Define the simulation parameters and write the input file (.in)
filename = 'simulation_4spheres'
input_file = f'{filename}.in'

with open(input_file, 'w') as f:
    # Domain size: X=2.0m, Y=1.0m, Z=0.002m (2D simulation)
    f.write("#domain: 2.0 1.0 0.002\n")
    # Grid step: 2mm
    f.write("#dx_dy_dz: 0.002 0.002 0.002\n")
    # Time window: 12ns
    f.write("#time_window: 12e-9\n")

    # Material: Soil (dry sand properties approx: relative permittivity=3, conductivity=0.001)
    f.write("#material: 3 0.001 1 0 soil\n")
    # Material: Metal (PEC - Perfect Electric Conductor)
    f.write("#material: 1 1e8 1 0 metal\n")

    # Geometry: Background is soil
    f.write("#box: 0 0 0 2.0 1.0 0.002 soil\n")

    # 4 equal-height, equal-spacing metal "spheres" (cylinders in 2D)
    y = 0.60          # same depth for all
    r = 0.05          # same radius for all
    x0 = 0.40         # starting x
    dx = 0.40         # equal spacing
    for i in range(4):
        x = x0 + i * dx
        f.write(f"#cylinder: {x:.3f} {y:.3f} 0  {x:.3f} {y:.3f} 0.002  {r:.3f} metal\n")

    # Source and Receiver (Ricker wavelet, 1.5 GHz center frequency)
    f.write("#waveform: ricker 1 1.5e9 my_ricker\n")
    # Hertzian dipole source
    f.write("#hertzian_dipole: z 0.10 0.90 0 my_ricker\n")
    # Receiver point
    f.write("#rx: 0.14 0.90 0\n")

    # Scan: move source/receiver 2 cm per step
    f.write("#src_steps: 0.02 0 0\n")
    f.write("#rx_steps: 0.02 0 0\n")

print(f"Input file '{input_file}' created.")

# 2. Run the simulation (n=85 A-scans)
# This might take a few minutes depending on your CPU
print("Running simulation...")
# 使用 GPU 0 运行 85 个 A-scan
api(input_file, n=85, geometry_only=False, gpu=[0])

# 2.5 合并多输出为一个 HDF5
print("Merging outputs...")
import subprocess
subprocess.run([sys.executable, "-m", "tools.outputfiles_merge", filename, "--remove-files"], check=True)
merged_out = f"{filename}_merged.out"

# 3. Plot and save the B-scan
output_file = merged_out
print(f"Processing output: {output_file}")

try:
    import matplotlib
    matplotlib.use("Agg")  # 强制非交互后端，避免显示
    import matplotlib.pyplot as plt
    rx_number = 1
    rx_component = 'Ez'
    outputdata, dt = get_output_data(output_file, rx_number, rx_component)

    # outputdata 形状通常为 (time_samples, n_traces)
    plt.figure(figsize=(10, 6))
    plt.imshow(outputdata, aspect='auto', cmap='gray',
               origin='upper',
               extent=[0, outputdata.shape[1], outputdata.shape[0] * dt * 1e9, 0])
    plt.title('GPR B-scan Simulation (4 Metal Spheres)')
    plt.xlabel('Trace Number (Scan position)')
    plt.ylabel('Time (ns)')
    plt.colorbar(label='Field Strength')

    image_filename = 'bscan_result.png'
    plt.savefig(image_filename, dpi=150)
    plt.close()
    print(f"B-scan image saved to {image_filename}")
except Exception as e:
    print(f"Error processing output: {e}")