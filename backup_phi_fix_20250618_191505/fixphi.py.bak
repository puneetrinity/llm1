#!/usr/bin/env python3
"""
fix_phi_model_name.py - Fix Phi Model Name Throughout Codebase
Changes 'phi:3.5' to 'phi3.5' in all relevant files

The correct Ollama model name is 'phi3.5', not 'phi:3.5'
This script systematically updates all files in the codebase.
"""

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

class PhiModelNameFixer:
    def __init__(self):
        self.backup_dir = f"backup_phi_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.files_updated = []
        self.total_changes = 0
        
        # Files that are known to contain the model reference
        self.target_files = [
            "main.py",
            "main_master.py", 
            "services/semantic_enhanced_router.py",
            "services/optimized_router.py",
            "services/model_warmup.py",
            "config_enhanced.py",
            "update_to_4_models.py",
            "fixllm.sh",
            "download_4_models.sh",
            "enhanced_start.sh",
            "Dockerfile",
            "setup.sh",
            ".env",
            ".env.template"
        ]
    
    def print_status(self, message: str, status_type: str = "info"):
        """Print colored status messages"""
        colors = {
            "success": "\033[0;32m✅",
            "error": "\033[0;31m❌", 
            "warning": "\033[1;33m⚠️",
            "info": "\033[0;34mℹ️"
        }
        reset = "\033[0m"
        icon = colors.get(status_type, "\033[0;34mℹ️")
        print(f"{icon}{reset} {message}")
    
    def create_backup_dir(self):
        """Create backup directory"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            self.print_status(f"Created backup directory: {self.backup_dir}", "success")
        except Exception as e:
            self.print_status(f"Failed to create backup directory: {e}", "error")
            raise
    
    def backup_file(self, file_path: str) -> bool:
        """Create backup of a file"""
        try:
            if not os.path.exists(file_path):
                return False
            
            backup_path = os.path.join(self.backup_dir, f"{os.path.basename(file_path)}.bak")
            shutil.copy2(file_path, backup_path)
            return True
        except Exception as e:
            self.print_status(f"Failed to backup {file_path}: {e}", "error")
            return False
    
    def update_file(self, file_path: str) -> Tuple[bool, int]:
        """Update a single file, replacing phi:3.5 with phi3.5"""
        if not os.path.exists(file_path):
            self.print_status(f"File not found: {file_path}", "warning")
            return False, 0
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count current occurrences
            before_count = len(re.findall(r'phi:3\.5', content))
            
            if before_count == 0:
                self.print_status(f"No phi:3.5 references in {file_path}", "info")
                return True, 0
            
            # Create backup
            if not self.backup_file(file_path):
                return False, 0
            
            # Replace phi:3.5 with phi3.5
            updated_content = re.sub(r'phi:3\.5', 'phi3.5', content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            # Verify changes
            after_count = len(re.findall(r'phi3\.5', updated_content))
            
            self.print_status(f"Updated {file_path}: {before_count} changes made", "success")
            self.files_updated.append(file_path)
            
            return True, before_count
            
        except Exception as e:
            self.print_status(f"Failed to update {file_path}: {e}", "error")
            return False, 0
    
    def find_additional_files(self) -> List[str]:
        """Find additional files that might contain phi:3.5"""
        additional_files = []
        
        # Search Python files
        for py_file in Path('.').rglob('*.py'):
            if any(exclude in str(py_file) for exclude in ['backup_', '.git/', '__pycache__']):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'phi:3.5' in content and str(py_file) not in self.target_files:
                        additional_files.append(str(py_file))
            except:
                continue
        
        # Search shell scripts
        for sh_file in Path('.').rglob('*.sh'):
            if any(exclude in str(sh_file) for exclude in ['backup_', '.git/']):
                continue
            
            try:
                with open(sh_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'phi:3.5' in content and str(sh_file) not in self.target_files:
                        additional_files.append(str(sh_file))
            except:
                continue
        
        # Search config files
        for config_file in Path('.').rglob('*.env*'):
            if any(exclude in str(config_file) for exclude in ['backup_', '.git/']):
                continue
            
            try:
                with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'phi:3.5' in content and str(config_file) not in self.target_files:
                        additional_files.append(str(config_file))
            except:
                continue
        
        return additional_files
    
    def test_model_pull(self):
        """Test pulling the correct phi3.5 model"""
        self.print_status("Testing correct model name...", "info")
        
        try:
            # Check if ollama is available
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.print_status("Ollama not found - test pull manually with: ollama pull phi3.5", "warning")
                return
            
            # Try to pull phi3.5
            self.print_status("Attempting to pull phi3.5...", "info")
            result = subprocess.run(['ollama', 'pull', 'phi3.5'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.print_status("✅ phi3.5 model downloaded successfully!", "success")
            else:
                self.print_status(f"⚠️ phi3.5 download failed: {result.stderr}", "warning")
                
        except subprocess.TimeoutExpired:
            self.print_status("Model download timed out - this is normal for large models", "warning")
        except FileNotFoundError:
            self.print_status("Ollama not found - test pull manually with: ollama pull phi3.5", "warning")
        except Exception as e:
            self.print_status(f"Error testing model pull: {e}", "warning")
    
    def run(self):
        """Main execution function"""
        print("🔧 Fixing Phi Model Name in Codebase")
        print("====================================")
        print("Changing 'phi:3.5' → 'phi3.5' everywhere")
        print()
        
        # Create backup directory
        self.create_backup_dir()
        
        # Update target files
        self.print_status("Updating known files...", "info")
        for file_path in self.target_files:
            success, changes = self.update_file(file_path)
            if success:
                self.total_changes += changes
        
        # Find and update additional files
        self.print_status("Searching for additional files...", "info")
        additional_files = self.find_additional_files()
        
        if additional_files:
            self.print_status(f"Found {len(additional_files)} additional files with phi:3.5", "info")
            for file_path in additional_files:
                success, changes = self.update_file(file_path)
                if success:
                    self.total_changes += changes
        
        # Summary
        print()
        self.print_status("Model name fix completed!", "success")
        print()
        self.print_status("Summary of changes:", "info")
        print(f"   📝 Changed: phi:3.5 → phi3.5")
        print(f"   📁 Backups: {self.backup_dir}/")
        print(f"   📄 Files updated: {len(self.files_updated)}")
        print(f"   🔢 Total changes: {self.total_changes}")
        
        if self.files_updated:
            print("\n📋 Updated files:")
            for file_path in self.files_updated:
                print(f"   • {file_path}")
        
        # Test model pull
        print()
        self.test_model_pull()
        
        # Next steps
        print()
        self.print_status("Next steps:", "info")
        print("1. Test the model: ollama run phi3.5 'What is 2+2?'")
        print("2. Start your app: python main_master.py")
        print("3. Test math queries to ensure phi3.5 routing works")
        print()
        self.print_status("Phi model name fix complete!", "success")

def main():
    """Main entry point"""
    try:
        fixer = PhiModelNameFixer()
        fixer.run()
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
