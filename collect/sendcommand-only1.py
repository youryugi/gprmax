import paramiko

# SSH连接信息
host = "192.168.1.40"
username = "cs29yang"
password = "cs29"  # 实际的密码

# 创建SSH客户端并连接
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=username, password=password)

try:
    # 执行命令
    command = 'conda activate rss && python "C:/Users/cs29yang/Desktop/rss-yang/scanWithBat.py" ttt32'
    stdin, stdout, stderr = ssh.exec_command(command)

    # 获取输出
    output = stdout.read().decode(errors='ignore')
    errors = stderr.read().decode(errors='ignore')

    # 打印输出和错误信息
    print(f"Output from {host}:")
    print(output)
    if errors:
        print(f"Errors from {host}:")
        print(errors)

    # 检查错误信息
    if errors:
        print(f"Error executing command on {host}: {errors}")
    else:
        print(f"Command executed successfully on {host}.")
finally:
    # 确保关闭SSH连接
    print('23')
