#!/usr/bin/env python3
"""
update_to_4_models.py - Update LLM Proxy to 4-Model Configuration
Windows VS Code Compatible Version - FIXED SYNTAX ERRORS

This script systematically updates all router files to match the 4-model banner
while preserving all existing functionality.
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class ModelConfigUpdater:
    def __init__(self):
        self.backup_dir = None
        self.files_updated = []

    def print_status(self, message: str, status: str = "info"):
        """Print colored status messages"""
        colors = {
            "info": "\033[94m",      # Blue
            "success": "\033[92m",   # Green
            "warning": "\033[93m",   # Yellow
            "error": "\033[91m",     # Red
            "reset": "\033[0m"       # Reset
        }

        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }

        print(
            f"{colors.get(status, '')}{icons.get(status, '')} {message}{colors['reset']}")

    def create_backup(self, files_to_backup: List[str]) -> str:
        """Create backup directory and backup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"backups/{timestamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)

        self.print_status(f"Creating backups in: {backup_dir}", "info")

        for file_path in files_to_backup:
            if Path(file_path).exists():
                shutil.copy2(file_path, backup_dir / Path(file_path).name)
                self.print_status(f"Backed up {file_path}", "success")
            else:
                self.print_status(
                    f"{file_path} not found (skipping)", "warning")

        self.backup_dir = str(backup_dir)
        return str(backup_dir)

    def update_semantic_enhanced_router(self):
        """Update services/semantic_enhanced_router.py"""
        file_path = "services/semantic_enhanced_router.py"
        if not Path(file_path).exists():
            self.print_status(f"{file_path} not found", "warning")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Updated model_config with 4 models
        new_model_config = '''        # Enhanced model configuration optimized for 4-model system
        self.model_config = {
            'phi3.5': {
                'priority': 1,  # Highest priority for math/reasoning
                'cost_per_token': 0.0002,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'specialties': ['complex_math', 'scientific_analysis', 'logical_reasoning', 'problem_solving']
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4000,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'specialties': ['quick_facts', 'efficient_responses', 'general_chat', 'summaries']
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 8192,
                'memory_mb': 4200,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'specialties': ['technical_documentation', 'programming', 'coding_help', 'api_docs']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'specialties': ['creative_writing', 'storytelling', 'conversations', 'narrative']
            }
        }'''

        # Updated intent mapping
        new_intent_mapping = '''        # Intent to model mapping - optimized for 4-model system
        self.intent_model_mapping = {
            # Math and reasoning → Phi-4 (specialized for complex reasoning)
            'math': 'phi3.5',
            'reasoning': 'phi3.5',
            'analysis': 'phi3.5',
            'logic': 'phi3.5',
            'scientific': 'phi3.5',
            
            # Coding and technical → Gemma (technical specialist)
            'coding': 'gemma:7b-instruct',
            'technical': 'gemma:7b-instruct',
            'programming': 'gemma:7b-instruct',
            'documentation': 'gemma:7b-instruct',
            
            # Creative tasks → Llama3 (creative specialist)
            'creative': 'llama3:8b-instruct-q4_0',
            'storytelling': 'llama3:8b-instruct-q4_0',
            'writing': 'llama3:8b-instruct-q4_0',
            'interview': 'llama3:8b-instruct-q4_0',
            'resume': 'llama3:8b-instruct-q4_0',
            
            # Quick facts and general → Mistral (efficient responses)
            'factual': 'mistral:7b-instruct-q4_0',
            'general': 'mistral:7b-instruct-q4_0',
            'summary': 'mistral:7b-instruct-q4_0'
        }'''

        # Replace model_config section
        content = re.sub(
            r'        # Enhanced model configuration.*?        }',
            new_model_config,
            content,
            flags=re.DOTALL
        )

        # Replace intent mapping section
        content = re.sub(
            r'        # Intent to model mapping.*?        }',
            new_intent_mapping,
            content,
            flags=re.DOTALL
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_updated.append(file_path)
        self.print_status(f"Updated {file_path}", "success")

    def update_optimized_router(self):
        """Update services/optimized_router.py"""
        file_path = "services/optimized_router.py"
        if not Path(file_path).exists():
            self.print_status(f"{file_path} not found", "warning")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Updated model capabilities
        new_capabilities = '''        # Model capabilities mapping - Updated for 4 models
        self.model_capabilities = {
            'phi3.5': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
            'mistral:7b-instruct-q4_0': ['factual', 'general', 'translation', 'summary'],
            'gemma:7b-instruct': ['coding', 'technical', 'programming', 'documentation'],
            'llama3:8b-instruct-q4_0': ['creative', 'storytelling', 'writing', 'conversation']
        }'''

        # Replace model capabilities
        content = re.sub(
            r'        # Model capabilities mapping.*?        }',
            new_capabilities,
            content,
            flags=re.DOTALL
        )

        # Update fallback initialization
        new_fallback = '''                self.available_models = {
                    'phi3.5': {'priority': 1, 'good_for': ['math', 'reasoning']},
                    'mistral:7b-instruct-q4_0': {'priority': 2, 'good_for': ['general']},
                    'gemma:7b-instruct': {'priority': 2, 'good_for': ['coding']},
                    'llama3:8b-instruct-q4_0': {'priority': 3, 'good_for': ['creative']}
                }'''

        content = re.sub(
            r"                self\.available_models = \{[^}]+\}",
            new_fallback,
            content,
            flags=re.DOTALL
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_updated.append(file_path)
        self.print_status(f"Updated {file_path}", "success")

    def update_main_router(self, file_path: str):
        """Update ModelRouter class in main.py or main_master.py"""
        if not Path(file_path).exists():
            self.print_status(f"{file_path} not found", "warning")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # New model config for main files
        new_model_config = '''        # Configuration for your 4 models - Updated to match banner
        self.model_config = {
            'phi3.5': {
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

        # Replace model config
        content = re.sub(
            r'        # Configuration for your.*?        }',
            new_model_config,
            content,
            flags=re.DOTALL
        )

        # New select_model logic
        new_select_logic = '''        # Model selection logic - Updated for 4 models matching banner
        
        # Math, logic, scientific analysis → Phi-4
        if any(word in text_lower for word in [
            'calculate', 'solve', 'equation', 'math', 'formula', 'logic', 
            'analyze', 'scientific', 'reasoning', 'proof', 'theorem'
        ]):
            if 'phi3.5' in self.available_models:
                return 'phi3.5'
        
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
        
        # Default to Mistral for quick facts and general queries'''

        # Find and replace the old routing logic
        old_logic_pattern = r'        # Model selection logic.*?if \'deepseek-v2:7b-q4_0\' in self\.available_models:\s+return \'deepseek-v2:7b-q4_0\''
        if re.search(old_logic_pattern, content, re.DOTALL):
            content = re.sub(old_logic_pattern, new_select_logic,
                             content, flags=re.DOTALL)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_updated.append(file_path)
        self.print_status(f"Updated {file_path}", "success")

    def update_model_warmup(self):
        """Update services/model_warmup.py"""
        file_path = "services/model_warmup.py"
        if not Path(file_path).exists():
            self.print_status(f"{file_path} not found", "warning")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update model priorities
        new_priorities = '''        self.model_priorities = {
            'phi3.5': 1,                           # Highest priority (reasoning)
            'mistral:7b-instruct-q4_0': 2,          # High priority (general)
            'gemma:7b-instruct': 2,                 # High priority (technical)
            'llama3:8b-instruct-q4_0': 3            # Medium priority (creative)
        }'''

        # Update warmup prompts
        new_prompts = '''        # Warmup prompts for each model type - Updated for 4 models
        self.warmup_prompts = {
            'phi3.5': [
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

        # Replace priorities
        content = re.sub(
            r'        self\.model_priorities = \{[^}]+\}',
            new_priorities,
            content,
            flags=re.DOTALL
        )

        # Replace prompts
        content = re.sub(
            r'        # Warmup prompts for each model type.*?        }',
            new_prompts,
            content,
            flags=re.DOTALL
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_updated.append(file_path)
        self.print_status(f"Updated {file_path}", "success")

    def update_enhanced_config(self):
        """Update config_enhanced.py"""
        file_path = "config_enhanced.py"
        if not Path(file_path).exists():
            self.print_status(f"{file_path} not found", "warning")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update model priorities in config
        new_config_priorities = '''    MODEL_PRIORITIES: Dict[str, int] = Field(
        default={
            "phi3.5": 1,                           # Highest priority (reasoning)
            "mistral:7b-instruct-q4_0": 2,          # High priority (general)
            "gemma:7b-instruct": 2,                 # High priority (technical)  
            "llama3:8b-instruct-q4_0": 3            # Medium priority (creative)
        },
        description="Model priorities for 4-model system"
    )'''

        content = re.sub(
            r'    MODEL_PRIORITIES: Dict\[str, int\] = Field\([^)]+\)',
            new_config_priorities,
            content,
            flags=re.DOTALL
        )

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.files_updated.append(file_path)
        self.print_status(f"Updated {file_path}", "success")

    def create_model_download_script(self):
        """Create download_4_models.py script - FIXED VERSION"""
        # Create the download script content without complex string escaping
        script_lines = [
            '#!/usr/bin/env python3',
            '"""',
            'download_4_models.py - Download the 4 models for the updated LLM Proxy system',
            '"""',
            '',
            'import subprocess',
            'import sys',
            'import time',
            '',
            'def print_status(message, status="info"):',
            '    """Print status messages with colors"""',
            '    colors = {',
            '        "info": "\\033[94m",',
            '        "success": "\\033[92m",',
            '        "warning": "\\033[93m",',
            '        "error": "\\033[91m",',
            '        "reset": "\\033[0m"',
            '    }',
            '    icons = {',
            '        "info": "ℹ️",',
            '        "success": "✅",',
            '        "warning": "⚠️",',
            '        "error": "❌"',
            '    }',
            '    print(f"{colors.get(status, \'\')}{icons.get(status, \'\')} {message}{colors[\'reset\']}")',
            '',
            'def run_ollama_command(command):',
            '    """Run ollama command and return result"""',
            '    try:',
            '        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=1800)',
            '        return result.returncode == 0, result.stdout, result.stderr',
            '    except subprocess.TimeoutExpired:',
            '        return False, "", "Command timed out"',
            '    except Exception as e:',
            '        return False, "", str(e)',
            '',
            'def check_ollama_running():',
            '    """Check if Ollama service is running"""',
            '    success, _, _ = run_ollama_command("ollama list")',
            '    return success',
            '',
            'def start_ollama():',
            '    """Start Ollama service"""',
            '    print_status("Starting Ollama service...", "info")',
            '    subprocess.Popen("ollama serve", shell=True)',
            '    time.sleep(10)',
            '',
            'def download_model(model_name, description):',
            '    """Download a single model"""',
            '    print_status(f"Downloading {description}...", "info")',
            '    success, stdout, stderr = run_ollama_command(f"ollama pull {model_name}")',
            '    ',
            '    if success:',
            '        print_status(f"{description} downloaded successfully", "success")',
            '        return True',
            '    else:',
            '        print_status(f"Failed to download {description}: {stderr}", "error")',
            '        return False',
            '',
            'def main():',
            '    print("📦 Downloading 4 Models for Updated LLM Proxy")',
            '    print("==============================================")',
            '    print()',
            '    ',
            '    # Check if Ollama is running',
            '    if not check_ollama_running():',
            '        print_status("Ollama not running, attempting to start...", "warning")',
            '        start_ollama()',
            '        ',
            '        time.sleep(5)',
            '        if not check_ollama_running():',
            '            print_status("Failed to start Ollama. Please start it manually with \'ollama serve\'", "error")',
            '            return False',
            '    ',
            '    print_status("Ollama is running", "success")',
            '    print()',
            '    ',
            '    # Models to download',
            '    models = [',
            '        ("phi3.5", "🧠 Phi-3.5 (Reasoning model)"),',
            '        ("mistral:7b-instruct-q4_0", "⚡ Mistral 7B (General model)"),',
            '        ("gemma:7b-instruct", "⚙️ Gemma 7B (Technical model)"),',
            '        ("llama3:8b-instruct-q4_0", "🎨 Llama3 8B (Creative model)")',
            '    ]',
            '    ',
            '    print_status("Starting downloads (this may take a while)...", "info")',
            '    print()',
            '    ',
            '    success_count = 0',
            '    for model_name, description in models:',
            '        if download_model(model_name, description):',
            '            success_count += 1',
            '        print()',
            '    ',
            '    print("=" * 50)',
            '    if success_count == len(models):',
            '        print_status("🎉 All 4 models downloaded successfully!", "success")',
            '    else:',
            '        print_status(f"Downloaded {success_count}/{len(models)} models", "warning")',
            '    ',
            '    print()',
            '    print_status("Verifying downloaded models:", "info")',
            '    success, stdout, stderr = run_ollama_command("ollama list")',
            '    if success:',
            '        print(stdout)',
            '    else:',
            '        print_status("Failed to list models", "error")',
            '    ',
            '    print()',
            '    print("🎯 Your 4 models will now route as:")',
            '    print("├── Math/Logic queries    → Phi-3.5")',
            '    print("├── Coding/Technical      → Gemma 7B")',
            '    print("├── Creative/Writing      → Llama3 8B")',
            '    print("└── General/Quick facts   → Mistral 7B")',
            '    ',
            '    return success_count == len(models)',
            '',
            'if __name__ == "__main__":',
            '    try:',
            '        success = main()',
            '        sys.exit(0 if success else 1)',
            '    except KeyboardInterrupt:',
            '        print_status("Download interrupted by user", "warning")',
            '        sys.exit(1)',
            '    except Exception as e:',
            '        print_status(f"Unexpected error: {e}", "error")',
            '        sys.exit(1)',
        ]

        # Write the script line by line to avoid string escaping issues
        with open("download_4_models.py", 'w', encoding='utf-8') as f:
            f.write('\n'.join(script_lines))

        self.print_status("Created download_4_models.py", "success")

    def run_update(self):
        """Run the complete update process"""
        print("🚀 Updating LLM Proxy to 4-Model Configuration")
        print("=" * 50)
        print()
        print("Target Models:")
        print("├── 🧠 Phi-3.5 Reasoning     → Complex math, logic, scientific analysis")
        print("├── 🎨 Llama3 8B-Instruct   → Creative writing, conversations, storytelling")
        print("├── ⚙️  Gemma 7B-Instruct    → Technical documentation, coding, programming")
        print("└── ⚡ Mistral 7B           → Quick facts, summaries, efficient responses")
        print()

        # Files to backup and update
        files_to_backup = [
            "services/router.py",
            "services/semantic_enhanced_router.py",
            "services/optimized_router.py",
            "main.py",
            "main_master.py",
            "services/model_warmup.py",
            "config_enhanced.py"
        ]

        try:
            # Step 1: Create backups
            self.print_status("Step 1: Creating Backups", "info")
            self.create_backup(files_to_backup)
            print()

            # Step 2: Update semantic enhanced router
            self.print_status("Step 2: Updating Enhanced Router", "info")
            self.update_semantic_enhanced_router()
            print()

            # Step 3: Update optimized router
            self.print_status("Step 3: Updating Optimized Router", "info")
            self.update_optimized_router()
            print()

            # Step 4: Update main.py
            self.print_status("Step 4: Updating main.py ModelRouter", "info")
            self.update_main_router("main.py")
            print()

            # Step 5: Update main_master.py
            self.print_status(
                "Step 5: Updating main_master.py ModelRouter", "info")
            self.update_main_router("main_master.py")
            print()

            # Step 6: Update model warmup
            self.print_status("Step 6: Updating Model Warmup Service", "info")
            self.update_model_warmup()
            print()

            # Step 7: Update enhanced config
            self.print_status(
                "Step 7: Updating Enhanced Configuration", "info")
            self.update_enhanced_config()
            print()

            # Step 8: Create download script
            self.print_status("Step 8: Creating Model Download Script", "info")
            self.create_model_download_script()
            print()

            # Summary
            self.print_status("🎉 4-Model Update Complete!", "success")
            print("=" * 30)
            print()
            print("📋 What was updated:")
            for file in self.files_updated:
                print(f"├── ✅ {file}")
            print("└── ✅ download_4_models.py script created")
            print()
            print(f"📁 Backups saved to: {self.backup_dir}")
            print()
            print("🚀 Next steps:")
            print("1. Download the 4 models: python download_4_models.py")
            print("2. Start your application: python main_master.py")
            print("3. Test routing with different query types")
            print()
            print("🎯 Your 4 models will now route as:")
            print("├── Math/Logic queries    → Phi-3.5")
            print("├── Coding/Technical      → Gemma 7B")
            print("├── Creative/Writing      → Llama3 8B")
            print("└── General/Quick facts   → Mistral 7B")
            print()
            self.print_status(
                "Ready to use your 4-model LLM Proxy!", "success")

            return True

        except Exception as e:
            self.print_status(f"Update failed: {str(e)}", "error")
            if self.backup_dir:
                self.print_status(
                    f"You can restore from backups in: {self.backup_dir}", "info")
            return False


def main():
    """Main entry point - FIXED for RunPod"""
    updater = ModelConfigUpdater()
    success = updater.run_update()

    if success:
        print("\n✨ Update completed successfully!")
        print("You can now run: python download_4_models.py")
    else:
        print("\n❌ Update failed. Check the error messages above.")

    # Removed the input() line that was causing EOFError in RunPod
    return success


if __name__ == "__main__":
    main()
