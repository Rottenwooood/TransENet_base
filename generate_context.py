#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目上下文文档生成器 (Project Context Generator)

此脚本用于自动生成项目的完整上下文文档，以便 AI 更好地理解项目结构和代码内容。
生成的文档采用 Markdown 格式，包含项目文件树和所有相关代码文件的完整内容。

AI 助手调用指南 (For AI Agents):
------------------------------
如果您是 Antigravity 或其他 AI Agent，在需要查阅特定模块、文件夹或文件的代码上下文时，
**请直接调用此脚本并传入路径参数，而不是请求用户手动操作。**

命令格式：
```bash
python3 generate_context.py [path1] [path2] ...
```

示例场景：
1. "请分析 reader 组件和 store 状态..."
   -> `python3 generate_context.py components/reader/ store/`

2. "检查 prisma schema 和 agent service..."
   -> `python3 generate_context.py prisma/schema.prisma lib/services/agent-service.ts`

3. "获取 i18n 配置相关文件..."
   -> `python3 generate_context.py i18n/ messages/ middleware.ts`

输出：
执行成功后，读取 `_PROJECT_CONTEXT.md` 文件即可获得针对性的上下文。

常规使用方法：
----------
1. 确保您位于项目根目录（Iterate/）下
2. 运行脚本（交互模式）：
   ```bash
   python3 generate_context.py
   ```

输出文件：
---------
- 文件名：`_PROJECT_CONTEXT.md`
- 位置：项目根目录
- 格式：Markdown
- 内容：
  - 项目文件树概览
  - 选定文件的完整内容

作者：Auto-generated
创建时间：2025
"""

import os
import subprocess
import sys
import argparse

# ==================== 配置区域 ====================

# 项目根目录（脚本所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 输出文件名
OUTPUT_FILE = "_PROJECT_CONTEXT.md"

# 1. 全局忽略的目录名 (任何层级匹配即忽略)
# 这些通常是构建产物、缓存或元数据目录，名称具有唯一性且通常不需要
IGNORE_DIR_NAMES = {
    # Version Control & IDE
    ".git", ".husky", ".agent", ".cursor", ".idea", ".vscode", ".hypothesis",
    
    # Build & Cache
    "__pycache__", ".venv", "venv", "env", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", "egg-info", "node_modules", ".pnpm-store",
    "dist", "build", "out", "target", "coverage", ".nyc_output", "test-results",
    
    # Logs & Temp
    "logs", "tmp", "temp", ".gemini", "wandb", "runs",
    
    # Data & Assets (Avoid large/binary/redundant data)
    "data", "dataset", "datasets", "results", "checkpoints", "weights", "models",
    "assets", "images", "img", "figures", "media", "video", "videos", "public",
    
    # Documentation (Strictly Excluded)
    "docs", "doc", "documentation", "references", "paper", "papers", 
    "manuscript", "manuscripts", "report", "reports",
    
    # Experimental / Examples
    "experiments", "examples", "demos", "test-data", "fixtures", "__fixtures__",
    
    # Tool Specific
    ".quarto", "_site", "_freeze", "playwright-report", "blob-report",
}

# 2. 指定路径忽略 (相对于项目根目录的路径)
# 仅当目录的相对路径完全匹配集合中的项时忽略。
# 这解决了 "bin", "src/bin" 同名但不同含义的问题。
IGNORE_PATHS = {
    "bin",           # 仅忽略根目录下的 bin
    "resources/bin", # 忽略 resources 下的 bin
}

# 要忽略的特定文件（精确匹配文件名）
IGNORE_FILES = {
    ".gitignore",
    "poetry.lock",   # 依赖锁定文件
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "AUTHORS",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    os.path.basename(__file__),  # 忽略脚本自身
    ".gitattributes",
    
    # System / OS
    ".DS_Store",     # macOS 系统文件
    "Thumbs.db",     # Windows 系统文件
    "ehthumbs.db",
    ".AppleDouble",
    ".LSOverride",
    
    # Environment
    ".env",          # 安全：绝对不要包含 env
    ".env.local",
    ".env.development.local",
    ".env.test.local",
    ".env.production.local",
    
    "test_output.txt",
    "junit.xml",
}

# 要忽略的文件扩展名（任何以这些扩展名结尾的文件都会被忽略）
IGNORE_EXTENSIONS = {
    # Documentation & Text
    ".md", ".markdown", ".rst", ".txt", ".pdf", ".docx", ".doc", ".odt",
    ".csv", ".tsv", ".xlsx", ".xls", ".xml", # Data
    
    # Images & Media
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".webm", ".mp3", ".wav", ".mov", ".avi",
    
    # Binary / Compiled / System
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin", 
    ".lock", ".log", ".map", ".swp", ".swo", ".DS_Store",
    
    # Model / Weights
    ".h5", ".pt", ".pth", ".onnx", ".pkl", ".gguf", ".safetensors",
    
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    
    # Notebooks (Often redundant to scripts or output code)
    ".ipynb",
    
    # Data & Config (Allow-list preferred for config)
    ".json",
    
    # Build
    ".tsbuildinfo", ".node",
}

# 强制保留的白名单文件 (即使扩展名在忽略列表中)
ALLOW_FILES = {
    "package.json",
    "tsconfig.json",
    "jsconfig.json",
    "vercel.json",
    "next.config.json",
    "deno.json",
    ".eslintrc.json",
    ".prettierrc.json"
}

# 语言映射：文件扩展名 -> Markdown 代码块语言标识符
# 用于在生成的文档中正确标识代码语言，提高 AI 理解准确性
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".css": "css",
    ".html": "html",
    ".md": "markdown",
    ".sql": "sql",
    ".sh": "bash",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".xml": "xml",
}

# ==================== 函数定义 ====================


def should_ignore_dir(dirpath, dirname, root_dir):
    """
    判断目录是否应该被忽略
    """
    # 1. 检查全局黑名单 (名字匹配)
    if dirname in IGNORE_DIR_NAMES:
        return True
        
    # 2. 检查路径黑名单 (相对路径匹配)
    # 计算相对于项目根目录的路径
    rel_path = os.path.relpath(dirpath, root_dir).replace('\\', '/')
    
    # 如果路径本身在黑名单中，忽略
    if rel_path in IGNORE_PATHS:
        return True
        
    return False


def should_ignore_file(filepath, filename, root_dir):
    """
    判断文件是否应该被忽略
    """
    # 1. 优先检查白名单 (必须包含的文件)
    if filename in ALLOW_FILES:
        return False

    # 计算相对路径
    relative_filepath = os.path.relpath(filepath, root_dir).replace('\\', '/')
    
    # 如果文件在 0_har_analysis 目录下，且扩展名是 .har 或 .json，则忽略
    if relative_filepath.startswith("0_har_analysis/"):
        _, ext = os.path.splitext(filename)
        if ext.lower() in (".har", ".json"):
            return True
    
    # 检查是否在 IGNORE_FILES 列表中
    if filename in IGNORE_FILES:
        return True
    
    # 检查扩展名是否在忽略列表中 (不区分大小写)
    _, ext = os.path.splitext(filename)
    if ext.lower() in IGNORE_EXTENSIONS:
        return True
    
    return False


def get_language_identifier(filepath):
    _, ext = os.path.splitext(filepath)
    return LANGUAGE_MAP.get(ext, "text")


def generate_file_tree(root_dir):
    """
    生成项目文件树（文本格式）
    """
    tree_lines = ["Iterate/"]
    dir_structure = {}
    
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 使用新的目录过滤逻辑
        valid_dirs = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not should_ignore_dir(dir_path, d, root_dir):
                valid_dirs.append(d)
        dirs[:] = valid_dirs
        
        relative_root = os.path.relpath(root, root_dir)
        if relative_root == ".":
            relative_root = ""
        else:
            relative_root = relative_root.replace('\\', '/')
        
        if relative_root not in dir_structure:
            dir_structure[relative_root] = {"dirs": [], "files": []}
        
        for d in sorted(dirs):
            dir_structure[relative_root]["dirs"].append(d)
        
        for f in sorted(files):
            filepath = os.path.join(root, f)
            if should_ignore_file(filepath, f, root_dir):
                continue
            dir_structure[relative_root]["files"].append(f)
    
    def print_tree(current_path, prefix="", is_last=True):
        if current_path not in dir_structure:
            return
        
        items = dir_structure[current_path]
        all_items = []
        
        for d in items["dirs"]:
            all_items.append(("dir", d))
        for f in items["files"]:
            all_items.append(("file", f))
        
        for idx, (item_type, item_name) in enumerate(all_items):
            is_last_item = (idx == len(all_items) - 1)
            
            if is_last_item:
                connector = "└── "
                next_prefix = prefix + "    "
            else:
                connector = "├── "
                next_prefix = prefix + "│   "
            
            tree_lines.append(f"{prefix}{connector}{item_name}" + ("/" if item_type == "dir" else ""))
            
            if item_type == "dir":
                sub_path = os.path.join(current_path, item_name) if current_path else item_name
                sub_path = sub_path.replace('\\', '/')
                print_tree(sub_path, next_prefix, is_last_item)
    
    print_tree("", "", True)
    return "\n".join(tree_lines)


def collect_valid_files(root_dir):
    """
    收集所有需要处理的有效文件路径（全量模式）
    """
    valid_files = []
    
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 使用新的目录过滤逻辑
        valid_dirs = []
        for d in dirs:
            dir_path = os.path.join(root, d)
            if not should_ignore_dir(dir_path, d, root_dir):
                valid_dirs.append(d)
        dirs[:] = valid_dirs
        
        for filename in sorted(files):
            filepath = os.path.join(root, filename)
            if should_ignore_file(filepath, filename, root_dir):
                continue
            valid_files.append(filepath)
            
    return valid_files


def expand_targets(targets, root_dir):
    """
    展开目标路径列表为有效文件列表
    支持：文件路径、目录路径 (递归)
    """
    expanded_files = set()
    
    print(f"模式：定向生成 (Targeted)")
    print(f"目标：{targets}")

    for target in targets:
        # 处理可能的相对路径
        if os.path.isabs(target):
            path = target
        else:
            path = os.path.join(root_dir, target)
        
        if not os.path.exists(path):
            print(f"❌ 警告: 目标不存在，已跳过 -> {target}")
            continue
            
        if os.path.isfile(path):
            filename = os.path.basename(path)
            if not should_ignore_file(path, filename, root_dir):
                expanded_files.add(path)
            else:
                print(f"⚠️ 跳过忽略文件: {target}")
                
        elif os.path.isdir(path):
            print(f"📂 扫描目录: {target} ...")
            # 递归扫描目录
            for root, dirs, files in os.walk(path, topdown=True):
                # 使用新的目录过滤逻辑
                valid_dirs = []
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    if not should_ignore_dir(dir_path, d, root_dir):
                        valid_dirs.append(d)
                dirs[:] = valid_dirs
                
                for f in files:
                    f_path = os.path.join(root, f)
                    if not should_ignore_file(f_path, f, root_dir):
                        expanded_files.add(f_path)
    
    return sorted(list(expanded_files))


def generate_full_content(files, root_dir):
    """
    生成所有文件的完整内容
    """
    all_contents = []
    
    for filepath in files:
        relative_filepath = os.path.relpath(filepath, root_dir).replace('\\', '/')
        filename = os.path.basename(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lang = get_language_identifier(filename)
            
            file_block = [
                f"--- START OF FILE: {relative_filepath} ---",
                f"```{lang}",
                content.strip(),
                "```",
                f"--- END OF FILE: {relative_filepath} ---"
            ]
            all_contents.append("\n\n".join(file_block))
            
        except UnicodeDecodeError:
            print(f"警告: 无法解码文件 {filepath}，已跳过。")
        except Exception as e:
            print(f"错误: 读取文件 {filepath} 时出错: {e}")
            
    return "\n\n\n".join(all_contents)

def is_file_tracked(filepath, root_dir):
    try:
        relative_filepath = os.path.relpath(filepath, root_dir)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative_filepath],
            cwd=root_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_deleted_files(root_dir):
    """
    获取相对于 HEAD 被删除的文件列表 (返回相对路径)
    """
    try:
        # git diff --name-status HEAD
        # output format:
        # D   path/to/file
        # M   path/to/file
        result = subprocess.run(
            ["git", "diff", "--name-status", "HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        deleted_files = []
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                status, filepath = parts
                if status.startswith('D'):
                    deleted_files.append(filepath.strip())
        return deleted_files
    except Exception:
        return []


def generate_diff_content(files, root_dir, target_paths=None):
    """
    生成 Smart Diff 内容，包含状态摘要
    """
    content_blocks = []
    
    # Summary Lists
    list_modified = []
    list_untracked = []
    list_deleted = []
    
    # 检查 git 是否可用
    try:
        subprocess.run(["git", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: 未检测到 Git 环境，无法生成 Diff。")
        sys.exit(1)
        
    print("正在计算 Git Diff (包括新文件和删除文件)...")
    
    # --- 1. 处理变更和新文件 (Modified & Untracked) ---
    for filepath in files:
        relative_filepath = os.path.relpath(filepath, root_dir).replace('\\', '/')
        filename = os.path.basename(filepath)
        
        # 1.1 检查文件是否 Tracked
        if not is_file_tracked(filepath, root_dir):
            try:
                # Untracked File
                list_untracked.append(relative_filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                lang = get_language_identifier(filename)
                file_block = [
                    f"--- NEW FILE (Untracked): {relative_filepath} ---",
                    f"```{lang}",
                    content.strip(),
                    "```",
                    f"--- END OF NEW FILE: {relative_filepath} ---"
                ]
                content_blocks.append("\n\n".join(file_block))
            except Exception as e:
                print(f"警告: 读取新文件失败 {filepath}: {e}")
            continue

        # 1.2 Tracked File -> 计算 Diff
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD", "--", relative_filepath],
                cwd=root_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            diff_output = result.stdout.strip()
            
            if diff_output:
                # Modified File
                list_modified.append(relative_filepath)
                
                file_block = [
                    f"--- GIT DIFF: {relative_filepath} ---",
                    "```diff",
                    diff_output,
                    "```",
                    f"--- END OF DIFF: {relative_filepath} ---"
                ]
                content_blocks.append("\n\n".join(file_block))
                
        except Exception as e:
            print(f"警告: 获取文件 Diff 失败 {filepath}: {e}")

    # --- 2. 处理删除的文件 (Deleted Files) ---
    deleted_files = get_deleted_files(root_dir)
    has_target_filter = target_paths is not None and len(target_paths) > 0

    for del_path in deleted_files:
        # 如果是定向模式，需要过滤
        if has_target_filter:
            abs_del = os.path.join(root_dir, del_path)
            matched = False
            for t in target_paths:
                if os.path.isabs(t):
                    abs_t = t
                else:
                    abs_t = os.path.join(root_dir, t)
                
                if abs_del == abs_t or abs_del.startswith(os.path.join(abs_t, "")):
                    matched = True
                    break
            if not matched:
                continue

        # Deleted File
        list_deleted.append(del_path)
        
        # 用户要求：删除的文件指出文件名即可
        file_block = [
            f"--- DELETED FILE: {del_path} ---",
            "(File has been deleted)",
            f"--- END OF DELETED FILE: {del_path} ---"
        ]
        content_blocks.append("\n\n".join(file_block))

    # --- 3. 生成摘要 (Summary) ---
    summary_lines = ["### Git Status Summary"]
    
    if not (list_modified or list_untracked or list_deleted):
        summary_lines.append("(No changes detected)")
    else:
        if list_modified:
            summary_lines.append(f"\n**Modified Files ({len(list_modified)})**:")
            for f in sorted(list_modified):
                summary_lines.append(f"- {f}")
                
        if list_untracked:
            summary_lines.append(f"\n**New / Untracked Files ({len(list_untracked)})**:")
            for f in sorted(list_untracked):
                summary_lines.append(f"- {f}")
                
        if list_deleted:
            summary_lines.append(f"\n**Deleted Files ({len(list_deleted)})**:")
            for f in sorted(list_deleted):
                summary_lines.append(f"- {f}")

    summary_block = "\n".join(summary_lines)
    
    # 组合最终内容
    final_output = [summary_block]
    if content_blocks:
        final_output.append("---")
        final_output.extend(content_blocks)
        
    if not content_blocks and not list_deleted: # Double check logic
         if not (list_modified or list_untracked):
            return "### Git Status Summary\n(Clean: No changes detected relative to HEAD)"

    return "\n\n\n".join(final_output)


def main():
    """
    主函数：生成最终的项目上下文文档
    """
    # 0. 命令行参数解析
    parser = argparse.ArgumentParser(description="Iterate Context Generator")
    parser.add_argument("targets", nargs="*", help="Specific files or directories to include (Targeted Mode)")
    parser.add_argument("--diff", "-d", action="store_true", help="Force Git Diff output (default is Full Content for targeted mode)")
    args = parser.parse_args()

    print("=" * 60)
    print("项目上下文文档生成器")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"输出文件: {OUTPUT_FILE}")
    print()
    
    mode = "full"
    target_paths = []

    # 1. 确定运行模式
    if args.targets:
        # 命令行参数模式
        mode = "targeted"
        target_paths = args.targets
        print(f">> 检测到命令行参数，使用：定向生成模式")
    else:
        # 交互模式
        print("请选择生成模式 / Select Mode:")
        print("1. 生成完整上下文 (Full Context) - 包含所有文件完整代码")
        print("2. 生成智能增量 (Smart Diff)    - Git Diff 变更 + 新增文件(Untracked)全量")
        print("3. 定向生成 (Targeted)          - 手动输入文件/文件夹路径")
        
        while True:
            try:
                choice = input("请输入序号 (1/2/3) [默认 1]: ").strip()
            except EOFError:
                choice = "1"
                print("\n无法读取输入，使用默认模式 1")
                
            if choice in ("", "1"):
                mode = "full"
                print(">> 已选择: 完整上下文模式")
                break
            elif choice == "2":
                mode = "diff"
                print(">> 已选择: 智能增量模式 (Diff + New Files)")
                break
            elif choice == "3":
                mode = "targeted"
                print(">> 已选择: 定向生成模式")
                try:
                    raw_input = input("请输入路径 (空格分隔): ").strip()
                    if raw_input:
                        target_paths = raw_input.split()
                        break
                    else:
                        print("未输入路径，返回主菜单")
                        continue
                except EOFError:
                    print("\n无法读取输入，退出")
                    return 1
            else:
                print("输入无效，请输入 1, 2 或 3")
    
    print()
    
    # 2. 生成文件树
    # 注意：为了保持上下文完整性，我们总是生成完整文件树，
    # 这样 AI 即使只看部分代码，也能知道它们在项目中的位置。
    print("正在生成文件树...")
    file_tree = generate_file_tree(PROJECT_ROOT)
    print(f"✓ 文件树生成完成（共 {len(file_tree.splitlines())} 行）")
    print()
    
    # 3. 收集/筛选文件列表
    print("正在扫描文件列表...")
    if mode == "targeted":
        valid_files = expand_targets(target_paths, PROJECT_ROOT)
    else:
        valid_files = collect_valid_files(PROJECT_ROOT)
        
    print(f"✓ 最终包含 {len(valid_files)} 个文件")
    if len(valid_files) == 0:
        print("⚠️ 警告: 没有找到任何符合条件的文件！")
    print()
    
    # 4. 根据模式生成内容
    if mode == "diff":
        # Smart Diff 模式
        print("正在生成智能增量内容 (Diff + New Files)...")
        content_body = generate_diff_content(valid_files, PROJECT_ROOT)
        content_title = "## 2. Changes & New Files"
        content_intro = "以下是项目中相对于 git HEAD 的变更内容。\n- Modified Files: 显示 git diff\n- New/Untracked Files: 显示完整文件内容"
        mode_desc = "Smart Diff (变更 + 新文件)"
    elif mode == "targeted":
        # Targeted 模式 (通常包含完整代码)
        print("正在读取选定文件的完整内容...")
        # 如果命令行强制加了 --diff，则用 diff 模式，否则默认 full
        if args.diff:
             content_body = generate_diff_content(valid_files, PROJECT_ROOT)
             content_title = "## 2. Targeted Changes (Diff)"
             mode_desc = "Targeted Diff (定向变更)"
        else:
            content_body = generate_full_content(valid_files, PROJECT_ROOT)
            content_title = "## 2. Targeted File Contents"
            mode_desc = f"Targeted Context (包含 {len(valid_files)} 个文件)"
        
        content_intro = f"以下是您指定的 {len(valid_files)} 个文件的内容。"
    else:
        # Full 模式
        print("正在读取所有文件完整内容...")
        content_body = generate_full_content(valid_files, PROJECT_ROOT)
        content_title = "## 2. File Contents"
        content_intro = "以下是项目中所有相关代码文件的完整内容。"
        mode_desc = "Full Context (完整代码)"
    
    # 5. 写入最终文件
    print(f"正在写入 {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # 写入文档头部
            f.write("# Project Context: Iterate\n\n")
            
            # 写入模式说明
            f.write(f"> **模式**: {mode_desc}\n")
            if mode == "targeted":
                f.write(f"> **目标**: {', '.join(target_paths)}\n")
            f.write(f"> **包含**: {len(valid_files)} 个文件\n\n")
            
            f.write("## 1. Project Structure Overview\n\n")
            f.write("这是一个机器可读的项目文件树，用于快速了解项目整体结构。\n\n")
            f.write("```text\n")
            f.write(file_tree)
            f.write("\n```\n\n")
            
            # 写入内容部分
            f.write(f"{content_title}\n\n")
            f.write(f"{content_intro}\n\n")
            f.write(content_body)
        
        # 获取文件大小
        file_size = os.path.getsize(OUTPUT_FILE)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✓ 文档生成成功！")
        print(f"  文件大小: {file_size_mb:.2f} MB ({file_size:,} 字节)")
        print()
        print("=" * 60)
        print("生成完成！您现在可以将此文件用作 AI 的上下文。")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 写入文件时出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
