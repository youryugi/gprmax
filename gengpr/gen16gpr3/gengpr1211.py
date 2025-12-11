import os
import sys
import subprocess
import random
import multiprocessing
from queue import Empty
from gprMax.gprMax import api
from tools.plot_Bscan import get_output_data, mpl_plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# 场景参数（统一等高、等间距）
y = 0.60       # 同一深度
r = 0.05       # 同一半径
x0 = 0.40      # 起始 x
dx = 0.40      # 间距
centers = [x0 + i * dx for i in range(4)]  # 左到右 4 个位置

def write_infile(input_file, present_bits, soil_er, soil_cond):
    """
    写入 .in 文件，支持自定义土壤参数
    soil_er: 土壤相对介电常数
    soil_cond: 土壤电导率
    """
    with open(input_file, 'w') as f:
        # 基础域与材料
        f.write("#domain: 2.0 1.0 0.002\n")
        f.write("#dx_dy_dz: 0.002 0.002 0.002\n")
        f.write("#time_window: 12e-9\n")
        
        # 使用随机生成的土壤参数
        f.write(f"#material: {soil_er:.2f} {soil_cond:.4f} 1 0 soil\n")
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

def run_case_task(task_data, gpu_id):
    """
    执行单个仿真任务
    task_data: (bitstr, variation_idx, soil_er, soil_cond, n_traces)
    """
    bitstr, var_idx, soil_er, soil_cond, n_traces = task_data
    
    # 文件名增加变体后缀，例如 simulation_4spheres_1010_v1
    filename = f"simulation_4spheres_{bitstr}_v{var_idx}"
    input_file = f"{filename}.in"
    
    # 1. 写入 .in
    write_infile(input_file, int(bitstr, 2), soil_er, soil_cond)
    print(f"[GPU {gpu_id}] Created {input_file} (Er={soil_er:.2f}, Cond={soil_cond:.4f})")

    # 2. 运行仿真
    # 注意：这里捕获可能的 gprMax 内部错误，防止一个任务挂掉整个进程
    try:
        api(input_file, n=n_traces, geometry_only=False, gpu=[gpu_id])
    except Exception as e:
        print(f"[GPU {gpu_id}] Error running simulation for {filename}: {e}")
        return

    # 3. 合并输出
    merged_out = f"{filename}_merged.out"
    try:
        subprocess.run([sys.executable, "-m", "tools.outputfiles_merge", filename, "--remove-files"], 
                       check=True, stdout=subprocess.DEVNULL) # 抑制合并时的输出
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] Error merging {filename}: {e}")
        return

    # 4. 绘图保存
    try:
        rx_number = 1
        rx_component = 'Ez'
        outputdata, dt = get_output_data(merged_out, rx_number, rx_component)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(outputdata, aspect='auto', cmap='gray',
                   origin='upper',
                   extent=[0, outputdata.shape[1], outputdata.shape[0] * dt * 1e9, 0])
        
        plt.title(f'B-scan ({bitstr}) Var:{var_idx}\nEr={soil_er:.2f}, Cond={soil_cond:.4f}')
        plt.xlabel('Trace Number')
        plt.ylabel('Time (ns)')
        plt.colorbar(label='Field Strength')
        
        image_filename = f"bscan_{bitstr}_v{var_idx}.png"
        plt.savefig(image_filename, dpi=150)
        plt.close()
        print(f"[GPU {gpu_id}] Saved {image_filename}")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error plotting {filename}: {e}")

def worker(gpu_id, task_queue):
    """
    工作进程：绑定一个 GPU，不断从队列取任务执行
    """
    print(f"Worker started on GPU {gpu_id}")
    while True:
        try:
            # 非阻塞获取，或者设置超时
            task = task_queue.get(timeout=3) 
        except Empty:
            # 队列空了，退出
            break
        
        # 增加一层宽泛的 try-except，确保无论发生什么，Worker 都能活着去取下一个任务
        try:
            run_case_task(task, gpu_id)
        except Exception as e:
            print(f"CRITICAL ERROR in Worker GPU {gpu_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Worker on GPU {gpu_id} finished.")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 定义可用的 GPU ID 列表。
    # 如果你有 2 张卡，写 [0, 1]。如果有 4 张，写 [0, 1, 2, 3]。
    # 请根据实际情况修改这里！
    AVAILABLE_GPUS = [0, 1] 
    
    N_TRACES = 85
    VARIATIONS_PER_CASE = 3  # 每种情况生成 3 个变体
    # ===========================================

    # 1. 生成所有任务参数
    # 16种情况 * 3种变体
    cases = [format(i, "04b") for i in range(16)]
    all_tasks = []

    # 用于保存到 CSV 的数据列表
    summary_data = []

    for bitstr in cases:
        for v in range(VARIATIONS_PER_CASE):
            # 随机生成土壤参数
            # 介电常数 3.0 ~ 9.0
            er = random.uniform(3.0, 9.0)
            # 电导率 0.001 ~ 0.02 S/m
            cond = random.uniform(0.001, 0.02)
            
            # 任务元组
            task = (bitstr, v, er, cond, N_TRACES)
            all_tasks.append(task)

            # 收集摘要数据
            summary_data.append({
                "Filename": f"simulation_4spheres_{bitstr}_v{v}",
                "Case_Bits": bitstr,
                "Variation_ID": v,
                "Soil_Er": f"{er:.4f}",
                "Soil_Cond": f"{cond:.6f}",
                "N_Traces": N_TRACES
            })

    print(f"Total tasks generated: {len(all_tasks)}")

    # 保存参数表到 CSV
    csv_filename = "simulation_summary.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ["Filename", "Case_Bits", "Variation_ID", "Soil_Er", "Soil_Cond", "N_Traces"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_data:
            writer.writerow(row)
    print(f"Parameter summary saved to {csv_filename}")

    # 2. 创建任务队列
    task_queue = multiprocessing.Queue()
    for task in all_tasks:
        task_queue.put(task)

    # 3. 启动工作进程
    processes = []
    # 为每个 GPU 启动一个进程
    # 如果任务很多而 GPU 很少，每个 GPU 会自动处理多个任务（串行处理队列）
    for gpu_id in AVAILABLE_GPUS:
        p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    # 4. 等待所有进程结束
    for p in processes:
        p.join()

    print("All simulations completed.")