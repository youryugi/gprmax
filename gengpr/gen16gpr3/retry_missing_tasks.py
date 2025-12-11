import os
import sys
import subprocess
import multiprocessing
import csv
from queue import Empty
from gprMax.gprMax import api
from tools.plot_Bscan import get_output_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 引入原脚本中的函数，确保逻辑一致
# 假设原脚本名为 gengpr1211.py，如果不是请修改 import
from gengpr1211 import write_infile, run_case_task, worker

def check_if_completed(filename):
    """
    检查任务是否已完成。
    判断标准：对应的 B-scan 图片是否存在。
    """
    # 这里假设生成的图片格式是 bscan_{bitstr}_v{var_idx}.png
    # 我们需要从 filename (simulation_4spheres_1010_v1) 解析出 bitstr 和 var_idx
    try:
        parts = filename.split('_')
        # parts: ['simulation', '4spheres', '1010', 'v1']
        bitstr = parts[2]
        var_idx = parts[3].replace('v', '')
        
        image_file = f"bscan_{bitstr}_v{var_idx}.png"
        return os.path.exists(image_file)
    except Exception:
        return False

if __name__ == "__main__":
    # ================= 配置区域 =================
    AVAILABLE_GPUS = [0, 1]  # 请根据实际情况修改
    CSV_FILE = "simulation_summary.csv"
    # ===========================================

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Cannot retry missing tasks.")
        sys.exit(1)

    missing_tasks = []

    print("Checking for missing tasks...")
    
    with open(CSV_FILE, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = row["Filename"]
            
            if not check_if_completed(filename):
                print(f"Missing: {filename}")
                
                # 重构任务元组
                # task_data: (bitstr, variation_idx, soil_er, soil_cond, n_traces)
                task = (
                    row["Case_Bits"],
                    int(row["Variation_ID"]),
                    float(row["Soil_Er"]),
                    float(row["Soil_Cond"]),
                    int(row["N_Traces"])
                )
                missing_tasks.append(task)

    if not missing_tasks:
        print("All tasks seem to be completed! No retry needed.")
        sys.exit(0)

    print(f"Found {len(missing_tasks)} missing tasks. Starting retry...")

    # 创建任务队列
    task_queue = multiprocessing.Queue()
    for task in missing_tasks:
        task_queue.put(task)

    # 启动工作进程
    processes = []
    for gpu_id in AVAILABLE_GPUS:
        # 复用原脚本的 worker 函数
        p = multiprocessing.Process(target=worker, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    # 等待结束
    for p in processes:
        p.join()

    print("Retry process completed.")