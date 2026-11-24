import os
from pathlib import Path

# Cấu hình
OUTPUT_FILE = "all_code_full.txt"
INCLUDE_DIRS = ["src", "scripts", "configs", "tests", "notebooks"]
INCLUDE_ROOT_FILES = [
    "README.md", "requirements.txt", "pytest.ini", ".gitignore", 
    "setup.py", "pyproject.toml", "Makefile"
]
INCLUDE_EXTS = {
    ".py", ".yaml", ".yml", ".md", ".txt", ".ini", ".json", ".ipynb", ".cfg", ".toml"
}

# Các thư mục cần bỏ qua (dữ liệu nặng, môi trường, kết quả)
EXCLUDE_DIRS = {
    ".git", ".pytest_cache", ".vscode", "checkpoints", 
    "data", "results", "venv", "__pycache__", ".ipynb_checkpoints", "node_modules"
}

def generate_tree(root_path):
    tree_str = "## PROJECT TREE\n"
    for root, dirs, files in os.walk(root_path):
        # Lọc bỏ các thư mục rác
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        level = root.replace(str(root_path), '').count(os.sep)
        indent = '│   ' * level
        tree_str += f'{indent}├── {os.path.basename(root)}/\n'
        
        subindent = '│   ' * (level + 1)
        for f in files:
            if f.endswith(tuple(INCLUDE_EXTS)) or f in INCLUDE_ROOT_FILES:
                tree_str += f'{subindent}├── {f}\n'
    return tree_str

def main():
    root = Path(__file__).parent.resolve()
    output_path = root / OUTPUT_FILE
    
    lines = []
    lines.append(generate_tree(root))
    lines.append("\n\n## FILE CONTENTS\n")
    
    total_files = 0
    seen_files = set()
    
    # 1. Quét các thư mục được chỉ định
    for dir_name in INCLUDE_DIRS:
        dir_path = root / dir_name
        if not dir_path.exists():
            continue
            
        for p in sorted(dir_path.rglob("*")):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.suffix not in INCLUDE_EXTS:
                continue
            if p in seen_files:
                continue
                
            seen_files.add(p)
            rel_path = p.relative_to(root)
            lines.append(f"\n{'='*80}\n")
            lines.append(f"FILE: {rel_path}\n")
            lines.append(f"{'='*80}\n")
            try:
                content = p.read_text(encoding="utf-8")
                lines.append(content)
                total_files += 1
            except UnicodeDecodeError:
                lines.append("[BINARY FILE - SKIPPED]\n")
                
    # 2. Quét các file lẻ ở thư mục gốc
    for file_name in INCLUDE_ROOT_FILES:
        p = root / file_name
        if p.exists() and p.is_file() and p not in seen_files:
            lines.append(f"\n{'='*80}\n")
            lines.append(f"FILE: {file_name}\n")
            lines.append(f"{'='*80}\n")
            try:
                content = p.read_text(encoding="utf-8")
                lines.append(content)
                total_files += 1
                seen_files.add(p)
            except UnicodeDecodeError:
                lines.append("[BINARY FILE - SKIPPED]\n")

    # Ghi ra file
    output_path.write_text("\n".join(lines), encoding="utf-8")
    
    size_kb = output_path.stat().st_size / 1024
    print(f"✅ Đã gom thành công {total_files} files.")
    print(f"✅ File kết quả: {OUTPUT_FILE} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    main()