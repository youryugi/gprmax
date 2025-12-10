import os
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

    # Geometry: 4 Metal Spheres (cylinders in 2D effectively act as cross-sections of spheres)
    # Format: #cylinder: x1 y1 z1 x2 y2 z2 radius material
    # Sphere 1
    f.write("#cylinder: 0.4 0.7 0 0.4 0.7 0.002 0.05 metal\n")
    # Sphere 2 (deeper)
    f.write("#cylinder: 0.8 0.5 0 0.8 0.5 0.002 0.05 metal\n")
    # Sphere 3
    f.write("#cylinder: 1.2 0.6 0 1.2 0.6 0.002 0.04 metal\n")
    # Sphere 4
    f.write("#cylinder: 1.6 0.4 0 1.6 0.4 0.002 0.06 metal\n")

    # Source and Receiver (Ricker wavelet, 1.5 GHz center frequency)
    f.write("#waveform: ricker 1 1.5e9 my_ricker\n")
    
    # Hertzian dipole source
    f.write("#hertzian_dipole: z 0.1 0.9 0 my_ricker\n")
    
    # Receiver point
    f.write("#rx: 0.14 0.9 0\n")

    # Scan parameters: Move source and receiver 0.02m (2cm) every step
    # Total scan length approx 1.8m -> ~90 steps
    f.write("#src_steps: 0.02 0 0\n")
    f.write("#rx_steps: 0.02 0 0\n")

print(f"Input file '{input_file}' created.")

# 2. Run the simulation (n=85 A-scans)
# This might take a few minutes depending on your CPU
print("Running simulation...")
# 添加 gpu=[0] 参数以使用第一块显卡
api(input_file, n=85, geometry_only=False, gpu=[0]) 

# 3. Plot and save the B-scan
output_file = f'{filename}.out'
print(f"Processing output: {output_file}")

try:
    # Load the simulation output data
    # The output file usually contains Ex, Ey, Ez fields. We typically look at Ez for this setup.
    rx_number = 1
    rx_component = 'Ez'
    
    # Get data and properties
    outputdata, dt = get_output_data(output_file, rx_number, rx_component)
    
    # Plotting
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.imshow(outputdata, aspect='auto', cmap='gray', extent=[0, 85, outputdata.shape[0] * dt * 1e9, 0])
    plt.title('GPR B-scan Simulation (4 Metal Spheres)')
    plt.xlabel('Trace Number (Scan position)')
    plt.ylabel('Time (ns)')
    plt.colorbar(label='Field Strength')
    
    # Save the figure
    image_filename = 'bscan_result.png'
    plt.savefig(image_filename)
    print(f"B-scan image saved to {image_filename}")
    plt.close()

except Exception as e:
    print(f"Error processing output: {e}")