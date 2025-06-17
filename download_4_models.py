#!/usr/bin/env python3
"""
download_4_models.py - Download the 4 models for the updated LLM Proxy system
"""

import subprocess
import sys
import time

def print_status(message, status="info"):
    """Print status messages with colors"""
    colors = {
        "info": "\033[94m",
        "success": "\033[92m",
        "warning": "\033[93m",
        "error": "\033[91m",
        "reset": "\033[0m"
    }
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    print(f"{colors.get(status, '')}{icons.get(status, '')} {message}{colors['reset']}")

def run_ollama_command(command):
    """Run ollama command and return result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=1800)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_ollama_running():
    """Check if Ollama service is running"""
    success, _, _ = run_ollama_command("ollama list")
    return success

def start_ollama():
    """Start Ollama service"""
    print_status("Starting Ollama service...", "info")
    subprocess.Popen("ollama serve", shell=True)
    time.sleep(10)

def download_model(model_name, description):
    """Download a single model"""
    print_status(f"Downloading {description}...", "info")
    success, stdout, stderr = run_ollama_command(f"ollama pull {model_name}")
    
    if success:
        print_status(f"{description} downloaded successfully", "success")
        return True
    else:
        print_status(f"Failed to download {description}: {stderr}", "error")
        return False

def main():
    print("📦 Downloading 4 Models for Updated LLM Proxy")
    print("==============================================")
    print()
    
    # Check if Ollama is running
    if not check_ollama_running():
        print_status("Ollama not running, attempting to start...", "warning")
        start_ollama()
        
        time.sleep(5)
        if not check_ollama_running():
            print_status("Failed to start Ollama. Please start it manually with 'ollama serve'", "error")
            return False
    
    print_status("Ollama is running", "success")
    print()
    
    # Models to download
    models = [
        ("phi:3.5", "🧠 Phi-3.5 (Reasoning model)"),
        ("mistral:7b-instruct-q4_0", "⚡ Mistral 7B (General model)"),
        ("gemma:7b-instruct", "⚙️ Gemma 7B (Technical model)"),
        ("llama3:8b-instruct-q4_0", "🎨 Llama3 8B (Creative model)")
    ]
    
    print_status("Starting downloads (this may take a while)...", "info")
    print()
    
    success_count = 0
    for model_name, description in models:
        if download_model(model_name, description):
            success_count += 1
        print()
    
    print("=" * 50)
    if success_count == len(models):
        print_status("🎉 All 4 models downloaded successfully!", "success")
    else:
        print_status(f"Downloaded {success_count}/{len(models)} models", "warning")
    
    print()
    print_status("Verifying downloaded models:", "info")
    success, stdout, stderr = run_ollama_command("ollama list")
    if success:
        print(stdout)
    else:
        print_status("Failed to list models", "error")
    
    print()
    print("🎯 Your 4 models will now route as:")
    print("├── Math/Logic queries    → Phi-3.5")
    print("├── Coding/Technical      → Gemma 7B")
    print("├── Creative/Writing      → Llama3 8B")
    print("└── General/Quick facts   → Mistral 7B")
    
    return success_count == len(models)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_status("Download interrupted by user", "warning")
        sys.exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {e}", "error")
        sys.exit(1)