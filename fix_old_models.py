#!/usr/bin/env python3
"""
fix_old_models.py - Complete fix for ALL files with old model configurations
Windows VS Code Compatible Version - Pure Python

This script removes all deepseek-v2:7b-q4_0 references and ensures 
proper 4-model configuration across all files.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class ModelFixTool:
    def __init__(self):
        self.backup_dir = None
        self.files_fixed = []
        
    def print_status(self, message: str, status: str = "info"):
        """Print colored status messages"""
        colors = {
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "info": "ℹ️ "
        }
        icon = colors.get(status, "ℹ️ ")
        print(f"{icon} {message}")
    
    def create_backup(self):
        """Create backup directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = f"backups/complete_fix_{timestamp}"
        os.makedirs(self.backup_dir, exist_ok=True)
        self.print_status(f"Backup directory: {self.backup_dir}", "info")
    
    def backup_file(self, file_path: str):
        """Backup a file before modifying it"""
        if os.path.exists(file_path):
            backup_path = os.path.join(self.backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            return True
        return False
    
    def fix_main_master(self):
        """Fix main_master.py with complete 4-model configuration"""
        file_path = "main_master.py"
        
        if not os.path.exists(file_path):
            self.print_status(f"{file_path} not found", "warning")
            return
            
        self.backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace model config completely
        new_config = '''        # Configuration for your 4 models - Updated to match banner
        self.model_config = {
            'phi:3.5': {
                'priority': 1,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'description': 'Phi-4 Reasoning - Complex math, logic, scientific analysis'
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'description': 'Mistral 7B - Quick facts, summaries, efficient responses'
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'description': 'Gemma 7B - Technical documentation, coding, programming'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'description': 'Llama3 8B-Instruct - Creative writing, conversations, storytelling'
            }
        }'''
        
        # Replace routing logic completely
        new_routing = '''        # Model selection logic - Updated for 4 models matching banner
        
        # Math, logic, scientific analysis → Phi-4
        if any(word in text_lower for word in [
            'calculate', 'solve', 'equation', 'math', 'formula', 'logic', 
            'analyze', 'scientific', 'reasoning', 'proof', 'theorem'
        ]):
            if 'phi:3.5' in self.available_models:
                return 'phi:3.5'
        
        # Coding, technical, programming → Gemma
        elif any(word in text_lower for word in [
            'code', 'function', 'program', 'debug', 'script', 'api', 
            'technical', 'documentation', 'programming', 'development'
        ]):
            if 'gemma:7b-instruct' in self.available_models:
                return 'gemma:7b-instruct'
        
        # Creative writing, storytelling → Llama3
        elif any(word in text_lower for word in [
            'story', 'creative', 'write', 'poem', 'chat', 'narrative', 
            'storytelling', 'conversation', 'dialogue'
        ]):
            if 'llama3:8b-instruct-q4_0' in self.available_models:
                return 'llama3:8b-instruct-q4_0'
        
        # Default to Mistral for quick facts and general queries
        if 'mistral:7b-instruct-q4_0' in self.available_models:
            return 'mistral:7b-instruct-q4_0'
        
        # Fallback to first available model
        return list(self.available_models.keys())[0]'''
        
        # Replace model config
        old_config_pattern = r'        # Configuration for your.*?        }'
        content = re.sub(old_config_pattern, new_config, content, flags=re.DOTALL)
        
        # Replace routing logic
        old_routing_pattern = r'        # Model selection logic.*?return list\(self\.available_models\.keys\(\)\)\[0\]'
        content = re.sub(old_routing_pattern, new_routing, content, flags=re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Fixed {file_path}", "success")
    
    def fix_optimized_router(self):
        """Fix services/optimized_router.py - remove deepseek references"""
        file_path = "services/optimized_router.py"
        
        if not os.path.exists(file_path):
            self.print_status(f"{file_path} not found", "warning")
            return
            
        self.backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace fallback initialization
        new_fallback = '''                self.available_models = {
                    'phi:3.5': {'priority': 1, 'good_for': ['math', 'reasoning']},
                    'mistral:7b-instruct-q4_0': {'priority': 2, 'good_for': ['general']},
                    'gemma:7b-instruct': {'priority': 2, 'good_for': ['coding']},
                    'llama3:8b-instruct-q4_0': {'priority': 3, 'good_for': ['creative']}
                }'''
        
        old_fallback_pattern = r'                self\.available_models = \{[^}]+\},'
        content = re.sub(old_fallback_pattern, new_fallback, content, flags=re.DOTALL)
        
        # Remove any deepseek lines
        deepseek_pattern = r'.*\'deepseek-v2:7b-q4_0\'.*\n'
        content = re.sub(deepseek_pattern, '', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Fixed {file_path}", "success")
    
    def fix_enhanced_start(self):
        """Fix enhanced_start.sh with proper 4-model setup"""
        file_path = "enhanced_start.sh"
        
        if not os.path.exists(file_path):
            self.print_status(f"{file_path} not found", "warning")
            return
            
        self.backup_file(file_path)
        
        new_content = '''#!/bin/bash

# Add Ollama to PATH (if installed in custom location)
export PATH=/workspace/ollama/bin:$PATH

# Enhanced startup script with 4-model preloading and warmup
echo "🚀 Starting Enhanced 4-Model LLM Proxy Service..."

# Start Ollama in background
echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to start..."
for i in {1..30}; do
  if curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Attempt $i/30 - waiting 2 seconds..."
  sleep 2
done

# Check if Ollama started successfully
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "❌ Failed to start Ollama service"
  exit 1
fi

# Pull and preload 4 models in priority order
echo "📦 Pulling and preloading 4 models..."

# Priority 1: Phi for reasoning
echo "   🔄 Pulling Phi-3.5 (Reasoning - Priority 1)..."
ollama pull phi:3.5 &
PHI_PID=$!

# Priority 2: Mistral for general
echo "   🔄 Pulling Mistral 7B (General - Priority 2)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Gemma for coding  
echo "   🔄 Pulling Gemma 7B (Technical - Priority 2)..."
ollama pull gemma:7b-instruct &
GEMMA_PID=$!

# Priority 3: Llama3 for creative
echo "   🔄 Pulling Llama3 8B (Creative - Priority 3)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

# Wait for priority 1 model (Phi) to complete first
echo "   ⏳ Waiting for priority model (Phi-3.5)..."
wait $PHI_PID
echo "   ✅ Phi-3.5 ready!"

# Warm up the priority model immediately
echo "   🔥 Warming up Phi-3.5..."
curl -X POST http://localhost:11434/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "phi:3.5",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "options": {"num_predict": 5}
  }' >/dev/null 2>&1

echo "   ✅ Phi-3.5 warmed up and ready for reasoning tasks!"

# Wait for other models in background
echo "   ⏳ Waiting for remaining models..."
wait $MISTRAL_PID && echo "   ✅ Mistral 7B ready!"
wait $GEMMA_PID && echo "   ✅ Gemma 7B ready!"
wait $LLAMA_PID && echo "   ✅ Llama3 8B ready!"

echo "🎯 All 4 models loaded successfully!"
echo ""
echo "🎯 Model Routing:"
echo "├── 🧠 Math/Logic/Reasoning    → Phi-3.5"
echo "├── ⚙️  Coding/Technical        → Gemma 7B"
echo "├── 🎨 Creative/Storytelling   → Llama3 8B"
echo "└── ⚡ General/Quick Facts     → Mistral 7B"

# Start the FastAPI application
echo "🌐 Starting FastAPI application..."
python3 main_master.py

# Cleanup function
cleanup() {
  echo "🛑 Shutting down services..."
  kill $OLLAMA_PID 2>/dev/null
  exit
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Keep the script running
wait
'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Fixed {file_path}", "success")
    
    def fix_config_enhanced(self):
        """Fix config_enhanced.py model priorities"""
        file_path = "config_enhanced.py"
        
        if not os.path.exists(file_path):
            self.print_status(f"{file_path} not found", "warning")
            return
            
        self.backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update MODEL_PRIORITIES
        new_priorities = '''    MODEL_PRIORITIES: Dict[str, int] = Field(
        default={
            "phi:3.5": 1,                           # Highest priority (reasoning)
            "mistral:7b-instruct-q4_0": 2,          # High priority (general)
            "gemma:7b-instruct": 2,                 # High priority (technical)  
            "llama3:8b-instruct-q4_0": 3            # Medium priority (creative)
        },
        description="Model priorities for 4-model system"
    )'''
        
        old_priorities_pattern = r'    MODEL_PRIORITIES: Dict\[str, int\] = Field\([^)]+\)'
        content = re.sub(old_priorities_pattern, new_priorities, content, flags=re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Fixed {file_path}", "success")
    
    def fix_model_warmup(self):
        """Fix services/model_warmup.py"""
        file_path = "services/model_warmup.py"
        
        if not os.path.exists(file_path):
            self.print_status(f"{file_path} not found", "warning")
            return
            
        self.backup_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update model priorities
        new_priorities = '''        self.model_priorities = {
            'phi:3.5': 1,                           # Highest priority (reasoning)
            'mistral:7b-instruct-q4_0': 2,          # High priority (general)
            'gemma:7b-instruct': 2,                 # High priority (technical)
            'llama3:8b-instruct-q4_0': 3            # Medium priority (creative)
        }'''
        
        # Update warmup prompts
        new_prompts = '''        # Warmup prompts for each model type - Updated for 4 models
        self.warmup_prompts = {
            'phi:3.5': [
                "What is 2+2?",
                "Solve for x: 3x + 5 = 14",
                "Analyze the logic in this statement"
            ],
            'mistral:7b-instruct-q4_0': [
                "What is the capital of France?",
                "Hello, how are you?",
                "Give me a quick summary"
            ],
            'gemma:7b-instruct': [
                "Write a Python function to sort a list",
                "Explain REST API principles", 
                "Debug this code: def hello(): return 'world'"
            ],
            'llama3:8b-instruct-q4_0': [
                "Write a short story about AI",
                "Tell me about career opportunities",
                "Create a creative dialogue"
            ]
        }'''
        
        old_priorities_pattern = r'        self\.model_priorities = \{[^}]+\}'
        content = re.sub(old_priorities_pattern, new_priorities, content, flags=re.DOTALL)
        
        old_prompts_pattern = r'        # Warmup prompts for each model type.*?        }'
        content = re.sub(old_prompts_pattern, new_prompts, content, flags=re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Fixed {file_path}", "success")
    
    def create_download_script(self):
        """Create download_4_models.py Python script"""
        file_path = "download_4_models.py"
        
        download_script = '''#!/usr/bin/env python3
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
'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(download_script)
        
        self.files_fixed.append(file_path)
        self.print_status(f"Created {file_path}", "success")
    
    def create_download_script_bash(self):
        """Create download_4_models.sh for non-Windows users"""
        file_path = "download_4_models.sh"
        
        bash_script = '''#!/bin/bash
# download_4_models.sh - Download the correct 4 models

set -e

echo "📦 Downloading 4 Models for LLM Proxy"
echo "====================================="
echo ""
echo "🎯 Target Models:"
echo "├── 🧠 Phi-3.5 (Reasoning)"
echo "├── ⚡ Mistral 7B (General)"  
echo "├── ⚙️  Gemma 7B (Technical)"
echo "└── 🎨 Llama3 8B (Creative)"
echo ""

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Starting Ollama..."
    ollama serve &
    sleep 10
fi

echo "📥 Downloading models in priority order..."

# Priority 1: Phi for reasoning
echo "🧠 Downloading Phi-3.5 (Reasoning model)..."
ollama pull phi:3.5 &
PHI_PID=$!

# Priority 2: Mistral for general use
echo "⚡ Downloading Mistral 7B (General model)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Gemma for coding
echo "⚙️  Downloading Gemma 7B (Technical model)..."
ollama pull gemma:7b-instruct &
GEMMA_PID=$!

# Priority 3: Llama3 for creative
echo "🎨 Downloading Llama3 8B (Creative model)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

echo "⏳ Waiting for downloads to complete..."

# Wait for all downloads
wait $PHI_PID && echo "✅ Phi-3.5 ready"
wait $MISTRAL_PID && echo "✅ Mistral 7B ready"  
wait $GEMMA_PID && echo "✅ Gemma 7B ready"
wait $LLAMA_PID && echo "✅ Llama3 8B ready"

echo ""
echo "🎉 All 4 models downloaded successfully!"
echo ""
echo "📊 Verify with: ollama list"
ollama list

echo ""
echo "🚀 Ready to start your 4-model LLM proxy:"
echo "   python main_master.py"
'''
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(bash_script)
        
        # Make executable on Unix systems
        try:
            os.chmod(file_path, 0o755)
        except:
            pass  # Windows doesn't need chmod
        
        self.files_fixed.append(file_path)
        self.print_status(f"Created {file_path}", "success")
    
    def run_complete_fix(self):
        """Run the complete fix process"""
        print("🔧 COMPLETE MODEL UPDATE - Fixing ALL Files")
        print("============================================")
        print("")
        
        try:
            # Create backup
            self.create_backup()
            print("")
            
            # Fix all files
            self.print_status("🔧 Fixing main_master.py", "info")
            self.fix_main_master()
            print("")
            
            self.print_status("🔧 Fixing services/optimized_router.py", "info")
            self.fix_optimized_router()
            print("")
            
            self.print_status("🔧 Fixing enhanced_start.sh", "info")
            self.fix_enhanced_start()
            print("")
            
            self.print_status("🔧 Fixing config_enhanced.py", "info")
            self.fix_config_enhanced()
            print("")
            
            self.print_status("🔧 Fixing services/model_warmup.py", "info")
            self.fix_model_warmup()
            print("")
            
            self.print_status("🔧 Creating download scripts", "info")
            self.create_download_script()
            self.create_download_script_bash()
            print("")
            
            # Summary
            self.print_status("🎉 COMPLETE FIX FINISHED!", "success")
            print("=========================")
            print("")
            print("📋 Files Fixed:")
            for file_path in self.files_fixed:
                print(f"├── ✅ {file_path}")
            print("")
            print(f"📁 Backup location: {self.backup_dir}")
            print("")
            print("🚀 Next Steps:")
            print("1. Download models: python download_4_models.py")
            print("2. Start service: python main_master.py")
            print("3. Test routing with different query types")
            print("")
            print("🎯 Your 4-model system now routes:")
            print("├── Math/Logic queries    → Phi-3.5")
            print("├── Coding/Technical      → Gemma 7B") 
            print("├── Creative/Writing      → Llama3 8B")
            print("└── General/Quick facts   → Mistral 7B")
            print("")
            self.print_status("All old models removed - system ready!", "success")
            
            return True
            
        except Exception as e:
            self.print_status(f"Fix failed: {str(e)}", "error")
            if self.backup_dir:
                self.print_status(f"You can restore from backups in: {self.backup_dir}", "info")
            return False

def main():
    """Main entry point"""
    fixer = ModelFixTool()
    success = fixer.run_complete_fix()
    
    if success:
        print("\n✨ All fixes completed successfully!")
        print("Run: python download_4_models.py")
    else:
        print("\n❌ Fix failed. Check error messages above.")
    
    input("\nPress Enter to exit...")
    return success

if __name__ == "__main__":
    main()