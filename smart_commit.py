import subprocess
import sys
import os

def run_command(cmd_args):
    """运行命令"""
    try:
        subprocess.run(cmd_args, check=True)
    except subprocess.CalledProcessError:
        print(f"Error running command: {' '.join(cmd_args)}")
        sys.exit(1)

def get_output(cmd_str):
    """运行 shell 命令并获取输出"""
    try:
        return subprocess.check_output(cmd_str, shell=True, text=True).strip()
    except:
        return ""

def main():
    print("=== 智能自动提交与推送工具 ===")

    # 1. 确保所有文件处于非暂存状态 (Mixed Reset)
    # 这样我们才能重新按批次 add
    print("-> 正在整理工作区状态...")
    subprocess.run(["git", "reset"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. 获取所有待处理文件 (包括未追踪的和修改过的)
    print("-> 扫描变动文件...")
    untracked = get_output("git ls-files --others --exclude-standard").splitlines()
    modified = get_output("git diff --name-only").splitlines()
    
    # 去重合并
    all_files = list(set(untracked + modified))
    
    if not all_files:
        print("没有发现需要提交的文件。")
        sys.exit(0)

    print(f"-> 发现 {len(all_files)} 个文件需要提交。")

    # 3. 检查单个文件是否超过 100MB
    print("-> 检查大文件限制 (>100MB)...")
    oversized_files = []
    for f in all_files:
        if os.path.exists(f) and os.path.getsize(f) > 100 * 1024 * 1024:
            oversized_files.append(f)
    
    if oversized_files:
        print("\n!!! 错误: 以下文件超过 100MB，GitHub 无法接收 !!!")
        for f in oversized_files:
            print(f"- {f} ({os.path.getsize(f)/1024/1024:.2f} MB)")
        print("请删除这些文件或配置 Git LFS 后再重试。")
        sys.exit(1)

    # 4. 智能分批 Commit
    MAX_BATCH_BYTES = 1024 * 1024 * 1024  # 1GB 限制
    
    current_batch = []
    current_batch_size = 0
    batch_count = 1
    
    # 获取当前分支名
    branch = get_output("git rev-parse --abbrev-ref HEAD")
    if not branch: branch = "main"

    print(f"-> 开始分批提交 (目标分支: {branch})...")

    for file_path in all_files:
        if not os.path.exists(file_path): continue
            
        file_size = os.path.getsize(file_path)
        
        # 如果当前批次满了，先提交
        if (current_batch_size + file_size > MAX_BATCH_BYTES) and current_batch:
            print(f"   提交第 {batch_count} 批 (大小: {current_batch_size/1024/1024:.2f} MB, 文件数: {len(current_batch)})...")
            run_command(["git", "add"] + current_batch)
            run_command(["git", "commit", "-m", f"Auto-commit batch {batch_count} (size controlled)"])
            
            current_batch = []
            current_batch_size = 0
            batch_count += 1
        
        current_batch.append(file_path)
        current_batch_size += file_size

    # 提交最后一批
    if current_batch:
        print(f"   提交第 {batch_count} 批 (大小: {current_batch_size/1024/1024:.2f} MB, 文件数: {len(current_batch)})...")
        run_command(["git", "add"] + current_batch)
        run_command(["git", "commit", "-m", f"Auto-commit batch {batch_count} (final)"])

    print("\n=== 所有文件已安全提交 ===")

    # 5. 自动调用 push.py
    # 我们需要找到刚才提交之前的那个 commit 作为起点，或者直接让 push.py 自动检测
    # 这里我们简单地直接运行 push.py，让它去处理推送
    print("-> 开始自动推送...")
    
    # 获取远程仓库最新的 commit hash，作为推送起点，避免重复检查旧的
    try:
        remote_head = get_output(f"git ls-remote origin refs/heads/{branch}").split()[0]
        print(f"-> 远程起点: {remote_head}")
        push_cmd = ["python3", "push.py", "origin", branch, remote_head, "1"]
    except:
        print("-> 无法检测远程状态，尝试默认推送...")
        push_cmd = ["python3", "push.py", "origin", branch, "-", "1"]

    subprocess.run(push_cmd)

if __name__ == "__main__":
    main()