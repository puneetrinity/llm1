#!/usr/bin/env python3
"""
download_4_models.py - Download the correct 4 models for the LLM proxy
"""

import subprocess
import sys
import time
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_ollama():
    """Check if Ollama is running"""
    success, _, _ = run_command("ollama list")
    return success

def start_ollama():
    """Start Ollama service"""
    print("🤖 Starting Ollama...")
    if os.name == 'nt':  # Windows
        subprocess.Popen("ollama serve", shell=True)
    else:  # Unix/Linux
        subprocess.Popen("ollama serve &", shell=True)
    
    # Wait for Ollama to start
    for i in range(30):
        if check_ollama():
            print("✅ Ollama is ready!")
            return True
        print(f"   Waiting for Ollama... ({i+1}/30)")
        time.sleep(2)
    
    return False

def download_model(model_name, description):
    """Download a single model"""
    print(f"📥 Downloading {description}...")
    success, stdout, stderr = run_command(f"ollama pull {model_name}")
    
    if success:
        print(f"✅ {description} ready!")
        return True
    else:
        print(f"❌ Failed to download {model_name}: {stderr}")
        return False

def main():
    """Main download function"""
    print("📦 Downloading 4 Models for LLM Proxy")
    print("=====================================")
    print("")
    print("🎯 Target Models:")
    print("├── 🧠 Phi-3.5 (Reasoning)")
    print("├── ⚡ Mistral 7B (General)")  
    print("├── ⚙️  Gemma 7B (Technical)")
    print("└── 🎨 Llama3 8B (Creative)")
    print("")
    
    # Check/start Ollama
    if not check_ollama():
        if not start_ollama():
            print("❌ Failed to start Ollama. Please start it manually:")
            print("   ollama serve")
            sys.exit(1)
    else:
        print("✅ Ollama is already running!")
    
    # Models to download
    models = [
        ("phi:3.5", "🧠 Phi-3.5 (Reasoning)"),
        ("mistral:7b-instruct-q4_0", "⚡ Mistral 7B (General)"),
        ("gemma:7b-instruct", "⚙️  Gemma 7B (Technical)"),
        ("llama3:8b-instruct-q4_0", "🎨 Llama3 8B (Creative)")
    ]
    
    print("📥 Downloading models in priority order...")
    
    failed_models = []
    for model_name, description in models:
        if not download_model(model_name, description):
            failed_models.append(model_name)
    
    print("")
    if failed_models:
        print(f"⚠️  Some models failed to download: {', '.join(failed_models)}")
        print("   You can try downloading them manually:")
        for model in failed_models:
            print(f"   ollama pull {model}")
    else:
        print("🎉 All 4 models downloaded successfully!")
    
    print("")
    print("📊 Verify with:")
    print("   ollama list")
    
    # Show current models
    success, stdout, stderr = run_command("ollama list")
    if success:
        print("")
        print("📋 Currently available models:")
        print(stdout)
    
    print("")
    print("🚀 Ready to start your 4-model LLM proxy:")
    print("   python main_master.py")

if __name__ == "__main__":
    main()
