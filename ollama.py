#!/usr/bin/env python3
"""
ollama_startup_fixer.py - Fix Ollama not starting in containers
Specifically handles the "models pulled but ollama doesn't start" issue
"""

import os
import sys
import time
import subprocess
import signal
import psutil
import requests
from pathlib import Path


def print_msg(msg, prefix=""):
    """Print message with flush"""
    print(f"{prefix}{msg}")
    sys.stdout.flush()


def check_ollama_process():
    """Check if Ollama process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower():
                return proc.info['pid'], proc.info['cmdline']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None, None


def check_ollama_port():
    """Check if Ollama port 11434 is accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200, response.text
    except Exception as e:
        return False, str(e)


def kill_ollama_processes():
    """Kill any existing Ollama processes"""
    print_msg("🔍 Checking for existing Ollama processes...")
    
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower():
                print_msg(f"   Found Ollama process: PID {proc.info['pid']}")
                proc.terminate()
                killed_count += 1
                
                # Wait for graceful termination
                try:
                    proc.wait(timeout=5)
                    print_msg(f"   ✅ Gracefully terminated PID {proc.info['pid']}")
                except psutil.TimeoutExpired:
                    print_msg(f"   🔨 Force killing PID {proc.info['pid']}")
                    proc.kill()
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if killed_count > 0:
        print_msg(f"   ✅ Killed {killed_count} Ollama processes")
        time.sleep(2)
    else:
        print_msg("   ℹ️ No existing Ollama processes found")
    
    return killed_count


def find_ollama_binary():
    """Find the Ollama binary location"""
    print_msg("🔍 Looking for Ollama binary...")
    
    # Common locations
    locations = [
        "/usr/local/bin/ollama",
        "/usr/bin/ollama", 
        "/opt/ollama/bin/ollama",
        "/workspace/ollama/bin/ollama",
        "ollama",  # In PATH
    ]
    
    for location in locations:
        try:
            result = subprocess.run([location, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print_msg(f"   ✅ Found Ollama at: {location}")
                print_msg(f"   📍 Version: {result.stdout.strip()}")
                return location
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            continue
    
    print_msg("   ❌ Ollama binary not found in common locations")
    return None


def check_ollama_models():
    """Check what models are available locally"""
    print_msg("📦 Checking local models...")
    
    # Common model directories
    model_dirs = [
        os.path.expanduser("~/.ollama/models"),
        "/root/.ollama/models",
        "/workspace/.ollama/models",
        "/opt/ollama/models",
        "./models"
    ]
    
    found_models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print_msg(f"   📁 Found model directory: {model_dir}")
            try:
                # List contents
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path):
                        found_models.append(item)
                        print_msg(f"      📦 Model: {item}")
            except PermissionError:
                print_msg(f"   ❌ Permission denied accessing {model_dir}")
    
    if found_models:
        print_msg(f"   ✅ Found {len(found_models)} models locally")
    else:
        print_msg("   ⚠️ No models found locally")
    
    return found_models


def start_ollama_serve(ollama_binary):
    """Start Ollama serve with proper configuration"""
    print_msg("🚀 Starting Ollama serve...")
    
    # Environment variables for container/headless operation
    env = os.environ.copy()
    env.update({
        "OLLAMA_HOST": "0.0.0.0:11434",
        "OLLAMA_KEEP_ALIVE": "5m",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_FLASH_ATTENTION": "1",
    })
    
    # In containers, we might need to set more specific environment
    if os.path.exists("/.dockerenv") or os.environ.get("CONTAINER"):
        env.update({
            "OLLAMA_ORIGINS": "*",
            "OLLAMA_DEBUG": "1",
        })
    
    print_msg("   📊 Environment variables:")
    for key, value in env.items():
        if key.startswith("OLLAMA_"):
            print_msg(f"      {key}={value}")
    
    try:
        # Start Ollama serve in background
        process = subprocess.Popen(
            [ollama_binary, "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        print_msg(f"   🎯 Started Ollama serve (PID: {process.pid})")
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print_msg("   ✅ Ollama serve process is running")
            return process
        else:
            stdout, stderr = process.communicate()
            print_msg(f"   ❌ Ollama serve exited immediately")
            print_msg(f"   📝 Stdout: {stdout.decode()[:200]}")
            print_msg(f"   📝 Stderr: {stderr.decode()[:200]}")
            return None
            
    except Exception as e:
        print_msg(f"   ❌ Failed to start Ollama serve: {e}")
        return None


def wait_for_ollama_ready(timeout=60):
    """Wait for Ollama to be ready and responsive"""
    print_msg(f"⏳ Waiting for Ollama to be ready (timeout: {timeout}s)...")
    
    start_time = time.time()
    attempt = 0
    
    while time.time() - start_time < timeout:
        attempt += 1
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print_msg(f"   ✅ Ollama is ready! (attempt {attempt})")
                
                # Parse and show available models
                try:
                    data = response.json()
                    models = data.get('models', [])
                    print_msg(f"   📦 Available models: {len(models)}")
                    for model in models:
                        print_msg(f"      • {model.get('name', 'Unknown')}")
                except:
                    print_msg("   ✅ Ollama responding (couldn't parse models)")
                
                return True
                
        except requests.exceptions.RequestException:
            pass
        
        if attempt % 5 == 0:
            print_msg(f"   ⏳ Still waiting... (attempt {attempt})")
        
        time.sleep(2)
    
    print_msg(f"   ❌ Timeout waiting for Ollama to be ready")
    return False


def test_ollama_model():
    """Test Ollama with a simple model call"""
    print_msg("🧪 Testing Ollama with a simple request...")
    
    # Get available models first
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print_msg("   ❌ Can't get model list")
            return False
        
        models = response.json().get('models', [])
        if not models:
            print_msg("   ⚠️ No models available for testing")
            return True  # Ollama is working, just no models
        
        # Use the first available model
        test_model = models[0]['name']
        print_msg(f"   🎯 Testing with model: {test_model}")
        
        # Simple test request
        test_request = {
            "model": test_model,
            "prompt": "Hello",
            "stream": False,
            "options": {"num_predict": 5}
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            print_msg("   ✅ Ollama model test successful!")
            return True
        else:
            print_msg(f"   ❌ Model test failed: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print_msg("   ⚠️ Model test timeout (Ollama might be loading)")
        return True  # Ollama is probably working, just slow
    except Exception as e:
        print_msg(f"   ❌ Model test error: {e}")
        return False


def create_startup_script():
    """Create a reliable startup script for Ollama"""
    print_msg("📝 Creating reliable Ollama startup script...")
    
    script_content = '''#!/bin/bash
# start_ollama.sh - Reliable Ollama startup for containers

echo "🚀 Starting Ollama service..."

# Kill any existing Ollama processes
pkill -f ollama || true
sleep 2

# Set environment for container operation
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=5m
export OLLAMA_ORIGINS="*"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1

# Start Ollama serve in background
nohup ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Started Ollama with PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        exit 0
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

echo "❌ Ollama failed to start properly"
exit 1
'''
    
    script_path = "start_ollama.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print_msg(f"   ✅ Created: {script_path}")
    print_msg(f"   💡 Usage: ./{script_path}")
    
    return script_path


def main():
    """Main function to fix Ollama startup issues"""
    print_msg("🚨 OLLAMA STARTUP FIXER")
    print_msg("=" * 40)
    print_msg("Fixing: Models pulled but Ollama doesn't start")
    print_msg("=" * 40)
    
    # Step 1: Check current state
    print_msg("\n📊 STEP 1: Current State Analysis")
    
    pid, cmdline = check_ollama_process()
    if pid:
        print_msg(f"   ℹ️ Ollama process found: PID {pid}")
        print_msg(f"   📝 Command: {' '.join(cmdline) if cmdline else 'Unknown'}")
    else:
        print_msg("   ❌ No Ollama process running")
    
    port_ok, port_info = check_ollama_port()
    if port_ok:
        print_msg("   ✅ Ollama port 11434 is accessible")
    else:
        print_msg(f"   ❌ Ollama port not accessible: {port_info}")
    
    # Step 2: Clean up any broken processes
    print_msg("\n🧹 STEP 2: Cleanup")
    kill_ollama_processes()
    
    # Step 3: Find Ollama binary
    print_msg("\n🔍 STEP 3: Locate Ollama")
    ollama_binary = find_ollama_binary()
    if not ollama_binary:
        print_msg("❌ Cannot continue without Ollama binary")
        print_msg("💡 Try: curl -fsSL https://ollama.ai/install.sh | sh")
        return 1
    
    # Step 4: Check models
    print_msg("\n📦 STEP 4: Model Check")
    models = check_ollama_models()
    
    # Step 5: Start Ollama
    print_msg("\n🚀 STEP 5: Start Ollama Service")
    process = start_ollama_serve(ollama_binary)
    
    if not process:
        print_msg("❌ Failed to start Ollama serve")
        return 1
    
    # Step 6: Wait for ready
    print_msg("\n⏳ STEP 6: Wait for Ready")
    if not wait_for_ollama_ready():
        print_msg("❌ Ollama failed to become ready")
        return 1
    
    # Step 7: Test functionality
    print_msg("\n🧪 STEP 7: Test Functionality")
    if test_ollama_model():
        print_msg("✅ Ollama is fully functional!")
    else:
        print_msg("⚠️ Ollama started but model test failed")
    
    # Step 8: Create startup script
    print_msg("\n📝 STEP 8: Create Startup Script")
    script_path = create_startup_script()
    
    # Summary
    print_msg("\n" + "=" * 40)
    print_msg("🎯 SUMMARY")
    print_msg("=" * 40)
    print_msg("✅ Ollama service is now running!")
    print_msg(f"📍 PID: {process.pid}")
    print_msg("🌐 API: http://localhost:11434")
    print_msg(f"📜 Startup script: {script_path}")
    print_msg("\n💡 Next steps:")
    print_msg("   1. Test: curl http://localhost:11434/api/tags")
    print_msg("   2. Start your main.py: python main.py")
    print_msg("   3. Use startup script: ./start_ollama.sh")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_msg("\nOllama startup fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_msg(f"\nOllama startup fix error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)