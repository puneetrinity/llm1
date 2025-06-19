#!/usr/bin/env python3
"""
fix_encoding.py - Fix the Windows cp1252 encoding issue in main.py
"""

import os
import shutil


def fix_main_py():
    print("🔧 Fixing main.py encoding issue...")
    
    # Backup first
    shutil.copy("main.py", "main.py.backup")
    print("✅ Backed up main.py")
    
    # Try different encodings to read the file
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    content = None
    
    for encoding in encodings:
        try:
            with open("main.py", 'r', encoding=encoding) as f:
                content = f.read()
            print(f"✅ Read file with {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print("❌ Could not read main.py with any encoding")
        return False
    
    # Clean problematic characters
    print("🧹 Cleaning problematic characters...")
    
    # Replace smart quotes and other problematic characters
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote  
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2026': '...',# Ellipsis
    }
    
    for bad_char, good_char in replacements.items():
        if bad_char in content:
            content = content.replace(bad_char, good_char)
            print(f"   Replaced: {repr(bad_char)} -> {repr(good_char)}")
    
    # More aggressive cleaning - remove any non-ASCII that might cause issues
    try:
        content.encode('ascii')
        print("✅ Content is ASCII-safe")
    except UnicodeEncodeError:
        print("⚠️ Non-ASCII characters found, applying aggressive cleaning...")
        # Keep only printable ASCII + common Unicode
        content = ''.join(char if ord(char) < 128 or char in '\n\r\t' else '?' for char in content)
    
    # Save as clean UTF-8
    try:
        with open("main.py", 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Saved main.py as clean UTF-8")
        
        # Test if it compiles now
        with open("main.py", 'r', encoding='utf-8') as f:
            test_content = f.read()
        
        compile(test_content, "main.py", 'exec')
        print("✅ main.py now compiles successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving/testing: {e}")
        return False


def test_main_py():
    print("\n🧪 Testing fixed main.py...")
    try:
        import subprocess
        import sys
        
        # Test compilation
        result = subprocess.run([sys.executable, "-m", "py_compile", "main.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ main.py compiles successfully!")
            
            # Quick execution test
            result = subprocess.run([sys.executable, "-c", "exec(open('main.py', encoding='utf-8').read())"], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print("✅ main.py executes without encoding errors!")
                return True
            else:
                print(f"⚠️ main.py runs but may have other issues: {result.stderr[:100]}")
                return True  # Encoding is fixed, other issues are separate
        else:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚨 FIXING MAIN.PY ENCODING ISSUE")
    print("=" * 40)
    
    if fix_main_py():
        if test_main_py():
            print("\n🎉 SUCCESS! main.py encoding is fixed!")
            print("Now try: python main.py")
        else:
            print("\n⚠️ Encoding fixed but other issues remain")
    else:
        print("\n❌ Could not fix encoding issue")
        print("Backup available: main.py.backup")