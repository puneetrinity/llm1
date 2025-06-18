# fix_encoding.py
import os


def fix_file_encoding(filepath):
    """Fix file encoding issues"""
    try:
        # Try to read with different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        content = None

        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"✅ Successfully read {filepath} with {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"❌ Could not read {filepath} with any encoding")
            return False

        # Write back with UTF-8 encoding
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed encoding for {filepath}")
        return True

    except Exception as e:
        print(f"❌ Error fixing {filepath}: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Fixing encoding issues...")

    files_to_fix = [
        'services/enhanced_imports.py',
        'services/enhanced_ollama_client.py'
    ]

    for file in files_to_fix:
        if os.path.exists(file):
            fix_file_encoding(file)
        else:
            print(f"⚠️  File not found: {file}")

    print("✅ Encoding fixes completed!")
    input("Press Enter to close...")
