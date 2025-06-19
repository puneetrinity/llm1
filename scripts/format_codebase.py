import os
import subprocess

# Directories to format
TARGET_DIRS = [
    ".",
    "./services",
    "./models",
    "./middleware",
    "./utils",
    "./test",
    "./security",
]

# Exclude virtual environments and node_modules
EXCLUDES = ["venv", "env", "node_modules", "__pycache__"]


def should_exclude(path):
    return any(ex in path for ex in EXCLUDES)


def format_with_black():
    for target in TARGET_DIRS:
        for root, dirs, files in os.walk(target):
            if should_exclude(root):
                continue
            py_files = [os.path.join(root, f) for f in files if f.endswith(".py")]
            if py_files:
                print(f"Formatting {root}...")
                subprocess.run(["black", "--line-length", "88"] + py_files)


def main():
    print("Running Black to format all Python files...")
    format_with_black()
    print("Formatting complete.")


if __name__ == "__main__":
    main()
