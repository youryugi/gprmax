import paramiko
import threading
import subprocess

# 全局参数值
param_value = "chairinonethreetwofour0"

# SSH连接信息和命令
hosts = [
    {
        "host": "192.168.1.20",
        "username": "scope",
        "password": "scope",
        "command": f"conda activate rss && python C:/Users/scope/Desktop/rss-yang/scanWithBat.py {param_value}"
    },
    {
        "host": "192.168.1.30",
        "username": "cs29",
        "password": "cs29",
        "command": f"python C:/Users/79152/Desktop/rss-yang/scanWithBat.py {param_value}"
    },
    {
        "host": "192.168.1.40",
        "username": "cs29yang",
        "password": "cs29",
        "command": f"conda activate rss && python C:/Users/cs29yang/Desktop/rss-yang/scanWithBat.py {param_value}"
    },
    {
        "host": "192.168.1.10",
        "username": "yly",
        "password": "2525",
        "command": f"conda activate yolo5 && python C:/Users/yly/Desktop/rss-yang/scanWithBat.py {param_value}"
    },
]

def run_command(host_info):
    host = host_info["host"]
    username = host_info["username"]
    password = host_info["password"]
    command = host_info["command"]

    try:
        # 创建SSH客户端并连接
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=username, password=password)

        # 执行命令
        stdin, stdout, stderr = ssh.exec_command(command)

        # 获取输出
        print(f"Output from {host}:")
        print(stdout.read().decode())
        print(stderr.read().decode())

        # 关闭连接
        ssh.close()
    except Exception as e:
        print(f"Failed to execute command on {host}: {e}")

def run_local_script(script_path, param):
    try:
        # 运行本地Python脚本，并传递参数值
        result = subprocess.run(["python", script_path, param], capture_output=True, text=True)
        print(f"Output from local script {script_path}:")
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(f"Failed to run local script {script_path}: {e}")

# 创建远程执行命令的线程
threads = []
for host_info in hosts:
    thread = threading.Thread(target=run_command, args=(host_info,))
    threads.append(thread)
    thread.start()

# 运行本地脚本
local_script_path = r"C:\Users\79152\Desktop\3rdtopic\collect\scanWithBat.py"  # 替换为你的本地脚本路径
run_local_thread = threading.Thread(target=run_local_script, args=(local_script_path, param_value))
run_local_thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()
run_local_thread.join()
