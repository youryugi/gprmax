#!/usr/bin/env python3

import sys
import subprocess
import time
import math
import argparse

def run_command(command, capture_output=True):
    """运行 shell 命令并返回结果"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=False, 
            text=True, 
            stdout=subprocess.PIPE if capture_output else None, 
            stderr=subprocess.PIPE if capture_output else None
        )
        return result.returncode, result.stdout.strip() if result.stdout else "", result.stderr.strip() if result.stderr else ""
    except Exception as e:
        return 1, "", str(e)

def main():
    parser = argparse.ArgumentParser(description="Git 分批推送脚本 (解决 pack exceeds maximum allowed size 问题)")
    
    parser.add_argument("remote", nargs="?", default="origin", help="远程仓库名 (默认: origin)")
    parser.add_argument("branch", nargs="?", default="main", help="分支名 (默认: main)")
    parser.add_argument("start_commit", nargs="?", default="-", help="起始 commit hash，输入 '-' 表示自动检测远程位置或从第一次提交开始 (默认: -)")
    parser.add_argument("batch_size", nargs="?", type=int, default=1, help="每批推送的 commit 数量 (默认: 1)")

    args = parser.parse_args()

    remote = args.remote
    branch = args.branch
    start_commit = args.start_commit
    batch_size = args.batch_size

    print(f"配置: Remote='{remote}', Branch='{branch}', Start='{start_commit}', Batch={batch_size}")

    # 验证 remote 是否存在
    code, _, _ = run_command(f"git remote get-url {remote}")
    if code != 0:
        print(f"Error: Remote '{remote}' does not exist")
        sys.exit(1)

    # 如果 START_COMMIT 为空或为 "-"，尝试自动检测
    if not start_commit or start_commit == "-":
        # 直接查询远程仓库状态（最可靠的方法）
        print("Querying remote branch status...")
        code, remote_output, _ = run_command(f"git ls-remote {remote} refs/heads/{branch}")
        
        if code == 0 and remote_output:
            remote_commit = remote_output.split()[0]
            start_commit = remote_commit
            print(f"Remote branch exists, continuing from: {start_commit}")
        else:
            # 如果远程分支不存在，使用最早的提交
            code, out, _ = run_command("git rev-list --max-parents=0 HEAD")
            if code != 0:
                print("Error: Could not find initial commit.")
                sys.exit(1)
            start_commit = out
            print(f"Remote branch does not exist, using root commit: {start_commit}")

    # 验证起始 commit 是否有效
    code, _, _ = run_command(f"git rev-parse --verify {start_commit}^{{commit}}")
    if code != 0:
        print(f"Error: Invalid start commit: {start_commit}")
        sys.exit(1)

    # 获取从起始 commit 到 HEAD 的所有提交
    print("Fetching commit list...")
    code, out, _ = run_command(f"git rev-list --reverse {start_commit}..HEAD")
    
    if not out:
        print(f"No commits to push after {start_commit}")
        sys.exit(0)
        
    commits = out.splitlines()
    total_commits = len(commits)

    print(f"Found {total_commits} commits to push")
    print(f"Will push in batches of {batch_size} commits")

    # 计算批次
    num_batches = math.ceil(total_commits / batch_size)

    # 循环推送
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_commits)
        
        # 获取当前批次的最后一个 commit
        current_batch_commits = commits[start_idx:end_idx]
        end_commit = current_batch_commits[-1]
        
        # 获取用于显示的短 hash
        _, start_short, _ = run_command(f"git rev-parse --short {current_batch_commits[0]}")
        _, end_short, _ = run_command(f"git rev-parse --short {end_commit}")

        print(f"Pushing batch {i + 1}/{num_batches}: {start_short}..{end_short}")

        # 推送这一批次
        # 修改这里：添加 --force 参数
        push_cmd = f"git push {remote} {end_commit}:refs/heads/{branch} --force"
        
        push_process = subprocess.run(push_cmd, shell=True)

        if push_process.returncode == 0:
            print(f"Successfully pushed batch {i + 1}")
        else:
            print(f"Error pushing batch {i + 1}")
            print(f"Failed at commit range: {start_short}..{end_short}")
            print(f"Resume with: python push.py {remote} {branch} {end_commit} {batch_size}")
            sys.exit(1)

        # 添加延迟
        if i < num_batches - 1:
            time.sleep(2)

    print("All commits have been pushed successfully!")

if __name__ == "__main__":
    main()