#!/usr/bin/env python3
"""
final_fix.py - Last attempt to get Ollama working in RunPod
If this doesn't work, we containerize properly
"""

import os
import sys
import subprocess
import time
import requests


def try_fix():
    print("🚨 FINAL OLLAMA FIX ATTEMPT")
    print("=" * 40)
    
    # Method 1: Direct install and run
    print("Method 1: Direct approach...")
    try:
        # Install
        subprocess.run("curl -fsSL https://ollama.ai/install.sh | sh", shell=True, check=True)
        
        # Set environment
        os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
        os.environ['OLLAMA_ORIGINS'] = '*'
        
        # Start in background
        subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait and test
        time.sleep(15)
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ SUCCESS - Ollama is running!")
            return True
            
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Alternative binary location
    print("Method 2: Alternative path...")
    try:
        subprocess.run("wget https://github.com/ollama/ollama/releases/download/v0.1.32/ollama-linux-amd64 -O /usr/local/bin/ollama", shell=True, check=True)
        subprocess.run("chmod +x /usr/local/bin/ollama", shell=True, check=True)
        
        subprocess.Popen(['/usr/local/bin/ollama', 'serve'], env=dict(os.environ, OLLAMA_HOST='0.0.0.0:11434'))
        
        time.sleep(15)
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ SUCCESS - Ollama is running!")
            return True
            
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Docker approach
    print("Method 3: Docker fallback...")
    try:
        subprocess.run("docker run -d -p 11434:11434 --name ollama ollama/ollama", shell=True, check=True)
        
        time.sleep(20)
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✅ SUCCESS - Ollama Docker is running!")
            return True
            
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return False


def test_main_py():
    """Test if main.py works now"""
    print("\n🧪 Testing main.py...")
    try:
        # Quick test - just import and see if it crashes
        result = subprocess.run([sys.executable, '-c', 'exec(open("main.py").read())'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ main.py runs without crashing!")
            return True
        else:
            print(f"❌ main.py still fails: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ main.py test failed: {e}")
        return False


if __name__ == "__main__":
    if try_fix():
        if test_main_py():
            print("\n🎉 EVERYTHING WORKS!")
            print("Run: python main.py")
            sys.exit(0)
        else:
            print("\n⚠️ Ollama works but main.py still has issues")
            sys.exit(1)
    else:
        print("\n❌ ALL METHODS FAILED")
        print("TIME TO CONTAINERIZE PROPERLY")
        sys.exit(1)