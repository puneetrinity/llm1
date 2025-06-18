# Save this as check_encoding.py and run: python check_encoding.py
import os

root = r"c:\Users\EverWanderingSoul\llm1"  # Adjust as needed

for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
        if fname.endswith(".py"):
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    f.read()
            except UnicodeDecodeError:
                print(f"Encoding issue: {fpath}")
