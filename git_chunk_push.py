import os
import subprocess
import sys
import pathlib

# =================配置区域=================
# 每个 Commit 的最大大小 (MB)
# GitHub 建议推送到远程的包大小最好不要超过 1GB
MAX_BATCH_SIZE_MB = 500 

# GitHub 单个文件硬限制 (MB)
GITHUB_FILE_LIMIT_MB = 100

# 远程仓库名称
REMOTE_NAME = "origin"
# =========================================

def run_command(command, ignore_errors=False, stream=False):
    """运行 Shell 命令并返回输出；stream=True 时直通输出"""
    try:
        if stream:
            result = subprocess.run(command, shell=True, check=True)
            return ""  # push 类命令主要看终端输出
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if not ignore_errors:
            if not stream:
                print(f"❌ Error running command: {command}")
                print(f"Error details: {e.stderr}")
            sys.exit(1)
        return None

def get_file_size_mb(path):
    """获取路径大小 (MB)，文件用 getsize，目录递归累加；不存在返回0"""
    p = pathlib.Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        try:
            return p.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0
    # 目录：递归累加内部文件大小
    total = 0
    try:
        for fp in p.rglob("*"):
            if fp.is_file():
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return total / (1024 * 1024)

def get_current_branch():
    return run_command("git branch --show-current")

def reset_unpushed_commits(branch_name):
    """
    检查是否有未推送的 commit，如果有，则软重置回远程分支状态，
    将所有更改变为 '待提交' 状态，以便重新分包。
    """
    print("Checking for unpushed commits...")
    try:
        # 确保远程引用最新
        run_command(f"git fetch {REMOTE_NAME} {branch_name}", ignore_errors=True)

        remote_branch = f"{REMOTE_NAME}/{branch_name}"
        subprocess.run(
            f"git rev-parse --verify {remote_branch}",
            shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        count = run_command(f"git rev-list --count {remote_branch}..HEAD") or "0"
        ahead = int(count)
        if ahead > 0:
            print(f"⚠️  Found {ahead} unpushed commits.")
            print(f"   Resetting branch to {remote_branch} to re-chunk files...")
            run_command(f"git reset --soft {remote_branch}")
            run_command("git reset")
            print("✅ Commits reset. All changes are now in working directory ready for chunking.")
            return True
        else:
            print("ℹ️ No unpushed commits. Skip reset.")
    except subprocess.CalledProcessError:
        print(f"ℹ️  Remote branch {REMOTE_NAME}/{branch_name} not found or error accessing it. Skipping reset.")
    return False

def get_changed_files():
    """获取所有未提交的文件列表 (包括未追踪的)，展开未追踪目录"""
    output = run_command("git status --porcelain")
    if not output:
        return []
    
    file_list = []
    for line in output.split('\n'):
        if not line:
            continue
        # 第 0-1 位是状态，第 3 位起是路径
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ")[-1]
        path = pathlib.Path(raw_path.strip('"'))
        if path.is_dir():
            # 展开目录中的所有文件
            for fp in path.rglob("*"):
                if fp.is_file():
                    file_list.append(str(fp))
        else:
            file_list.append(str(path))
    return file_list

def commit_and_push(files, batch_num, branch_name, batch_size_mb):
    """提交并推送一批文件"""
    if not files:
        return

    print(f"\n🚀 Processing Batch #{batch_num} ({len(files)} files, {batch_size_mb:.2f} MB)...")
    
    for f in files:
        if not os.path.exists(f):
            print(f"   ⚠️  Skip missing path: {f}")
            continue
        run_command(f'git add "{f}"')
    
    commit_msg = f"chore: auto-commit batch {batch_num} (chunk upload)"
    print(f"   Committing...")
    run_command(f'git commit -m "{commit_msg}"', stream=True)
    
    print(f"   Pushing to {REMOTE_NAME}/{branch_name}...")
    run_command(f"git push {REMOTE_NAME} {branch_name}", stream=True)
    print(f"✅ Batch #{batch_num} done.")

def ensure_gitignore(entries):
    """将给定路径写入 .gitignore（如不存在则创建），避免重复写入"""
    gi = pathlib.Path(".gitignore")
    existing = set()
    if gi.exists():
        existing = {line.strip() for line in gi.read_text(encoding="utf-8").splitlines() if line.strip()}
    new_items = []
    for e in entries:
        e = e.replace("\\", "/")
        if e not in existing:
            new_items.append(e)
    if new_items:
        with gi.open("a", encoding="utf-8") as f:
            for e in new_items:
                f.write(e + "\n")
        print("   📄 Added to .gitignore:")
        for e in new_items:
            print(f"      - {e}")

def main():
    print("=== Git Chunk Splitter & Pusher (Auto-Reset Version) ===")
    
    if not os.path.exists(".git"):
        print("❌ Error: Current directory is not a git repository.")
        return

    branch_name = get_current_branch()
    print(f"📍 Current Branch: {branch_name}")
    
    # 1. 自动拆解现有的 Commit
    reset_unpushed_commits(branch_name)

    # 2. 获取文件列表
    files = get_changed_files()
    if not files:
        print("✨ No changes to commit.")
        return

    print(f"🔍 Found {len(files)} changed/untracked files.")

    # 3. 分组逻辑
    current_batch = []
    current_batch_size = 0
    batch_counter = 1

    for file_path in files:
        f_size = get_file_size_mb(file_path)
        
        # 严重警告：超过 GitHub 单文件限制
        if f_size >= GITHUB_FILE_LIMIT_MB:
            ensure_gitignore([file_path])
            print(f"❌ [SKIP] File '{file_path}' is {f_size:.2f} MB.")
            print(f"   Reason: Exceeds GitHub 100MB limit. Please use Git LFS or remove it.")
            continue

        # 批次限制逻辑
        if current_batch_size + f_size > MAX_BATCH_SIZE_MB:
            commit_and_push(current_batch, batch_counter, branch_name, current_batch_size)
            batch_counter += 1
            current_batch = []
            current_batch_size = 0

        current_batch.append(file_path)
        current_batch_size += f_size

    # 4. 提交剩余的文件
    if current_batch:
        commit_and_push(current_batch, batch_counter, branch_name, current_batch_size)

    print("\n🎉 All chunks processed successfully!")
    print("Note: Files larger than 100MB were skipped.")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n🛑 Script interrupted by user.")
        sys.exit(1)