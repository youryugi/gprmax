import paramiko

# SSH连接信息和命令
hosts = [
    {
        "host": "192.168.1.20",
        "username": "scope",
        "password": "scope",
        "command": "conda activate rss && python C:/Users/scope/Desktop/rss-yang/scanWithBat.py"
    },
    {
        "host": "192.168.1.30",
        "username": "cs29",
        "password": "cs29",
        "command": "python C:/Users/79152/Desktop/rss-yang/scanWithBat.py"
    },
    {
        "host": "192.168.1.40",
        "username": "yang",
        "password": "2525",
        "command": "python C:/Users/yang/Desktop/rss-yang/scanWithBat.py"
    },

]

# 遍历每个主机并执行命令
for host_info in hosts:
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
