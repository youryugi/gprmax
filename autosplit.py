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
    return subprocess.check_output(cmd_str, shell=True, text=True).strip()

def main():
    if len(sys.argv) < 2:
        print("使用方法: python3 autosplit.py <失败的commit_hash>")
        sys.exit(1)

    bad_commit = sys.argv[1]
    print(f"=== 开始智能拆分 Commit: {bad_commit} ===")

    # 1. 获取父 commit
    try:
        parent_commit = get_output(f"git rev-parse {bad_commit}^")
        print(f"-> 父 Commit 是: {parent_commit}")
    except:
        print("错误: 找不到父 commit。")
        sys.exit(1)

    # 2. 回退
    print(f"-> 正在回退到 {parent_commit}...")
    run_command(["git", "reset", "--soft", parent_commit])
    run_command(["git", "reset"]) 

    # 3. 检查单个文件是否超过 100MB (GitHub 硬限制)
    # 我们把限制放宽到 100MB，因为你有 96MB 的文件
    print("-> 检查是否有单个文件超过 100MB...")
    try:
        large_files = get_output("find . -type f -size +100M -not -path '*/.*'")
        if large_files:
            print("\n!!! 警告: 发现单个文件超过 100MB !!!")
            print(large_files)
            sys.exit(1)
    except:
        pass

    # 4. 获取文件列表
    files_output = get_output("git diff --name-only")
    if not files_output:
        print("没有文件需要提交。")
        sys.exit(0)
    
    all_files = files_output.splitlines()
    print(f"-> 共有 {len(all_files)} 个文件待处理。")

    # 5. 智能分批 (按大小限制)
    MAX_BATCH_BYTES = 1024 * 1024 * 1024  # 限制每个 Commit 最大 1GB (非常安全)
    
    current_batch = []
    current_batch_size = 0
    batch_count = 1

    for file_path in all_files:
        if not os.path.exists(file_path):
            continue # 文件可能被删除了，跳过
            
        file_size = os.path.getsize(file_path)
        
        # 如果当前批次加上这个文件会超过 1GB，且当前批次不为空，则先提交当前批次
        if (current_batch_size + file_size > MAX_BATCH_BYTES) and current_batch:
            print(f"-> 提交第 {batch_count} 批 (大小: {current_batch_size/1024/1024:.2f} MB, 文件数: {len(current_batch)})...")
            run_command(["git", "add"] + current_batch)
            run_command(["git", "commit", "-m", f"Split commit {batch_count} (size controlled)"])
            
            # 重置计数器
            current_batch = []
            current_batch_size = 0
            batch_count += 1
        
        # 加入当前批次
        current_batch.append(file_path)
        current_batch_size += file_size

    # 提交剩余的文件
    if current_batch:
        print(f"-> 提交第 {batch_count} 批 (大小: {current_batch_size/1024/1024:.2f} MB, 文件数: {len(current_batch)})...")
        run_command(["git", "add"] + current_batch)
        run_command(["git", "commit", "-m", f"Split commit {batch_count} (final batch)"])

    print("\n=== 拆分完成! ===")
    print("每个 Commit 都已确保小于 1GB。")
    print(f"请运行: python push.py origin main {parent_commit} 1")

if __name__ == "__main__":
    main()