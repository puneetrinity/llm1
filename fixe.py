# fix_compatibility.py - Python Compatibility Fixer (Works on Windows/Linux/Mac)
import os
import sys
import re
import subprocess
from pathlib import Path
import shutil

class CompatibilityFixer:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.fixes_applied = 0
        self.fixes_skipped = 0
        self.fixes_failed = 0
    
    def print_status(self, message, status="info"):
        icons = {"success": "✅", "skip": "⏭️", "error": "❌", "info": "ℹ️"}
        print(f"{icons.get(status, 'ℹ️')} {message}")
    
    def backup_file(self, file_path):
        """Create backup of file before modifying"""
        if Path(file_path).exists():
            backup_path = f"{file_path}.backup.{int(os.path.getmtime(file_path))}"
            shutil.copy2(file_path, backup_path)
            self.print_status(f"Backed up {file_path} to {backup_path}")
    
    def fix_connection_pool(self):
        """Fix connection pool tcp_nodelay compatibility"""
        print("🔌 Checking connection pool compatibility...")
        
        file_path = Path("utils/connection_pool.py")
        if not file_path.exists():
            self.print_status("Connection pool file not found - skipping", "skip")
            self.fixes_skipped += 1
            return
        
        content = file_path.read_text(encoding='utf-8')
        
        # Check if fix is already applied
        if "# tcp_nodelay.*Compatibility fix" in content:
            self.print_status("Connection pool already fixed", "skip")
            self.fixes_skipped += 1
            return
        
        # Check if problematic line exists
        if "tcp_nodelay=self.config.tcp_nodelay," not in content:
            self.print_status("Connection pool doesn't need this fix", "skip")
            self.fixes_skipped += 1
            return
        
        # Apply fix
        if not self.dry_run:
            self.backup_file(file_path)
            new_content = content.replace(
                "tcp_nodelay=self.config.tcp_nodelay,",
                "# tcp_nodelay=self.config.tcp_nodelay,  # Compatibility fix"
            )
            file_path.write_text(new_content, encoding='utf-8')
        
        self.print_status("Connection pool tcp_nodelay fixed", "success")
        self.fixes_applied += 1
    
    def fix_websocket_dashboard(self):
        """Fix WebSocket dashboard constructor"""
        print("🌐 Checking WebSocket dashboard compatibility...")
        
        file_path = Path("utils/websocket_dashboard.py")
        if not file_path.exists():
            self.print_status("WebSocket dashboard file not found - skipping", "skip")
            self.fixes_skipped += 1
            return
        
        content = file_path.read_text(encoding='utf-8')
        
        # Check if fix is already applied
        if "def __init__(self, enhanced_dashboard, metrics_collector=None" in content:
            self.print_status("WebSocket dashboard already fixed", "skip")
            self.fixes_skipped += 1
            return
        
        # Check if problematic line exists
        if "def __init__(self, enhanced_dashboard):" not in content:
            self.print_status("WebSocket dashboard doesn't need this fix", "skip")
            self.fixes_skipped += 1
            return
        
        # Apply fix
        if not self.dry_run:
            self.backup_file(file_path)
            
            # Fix constructor signature
            new_content = content.replace(
                "def __init__(self, enhanced_dashboard):",
                "def __init__(self, enhanced_dashboard, metrics_collector=None, performance_monitor=None):"
            )
            
            # Add new instance variables
            new_content = re.sub(
                r"(self\.dashboard = enhanced_dashboard)",
                r"\1\n        self.metrics_collector = metrics_collector\n        self.performance_monitor = performance_monitor",
                new_content
            )
            
            file_path.write_text(new_content, encoding='utf-8')
        
        self.print_status("WebSocket dashboard constructor fixed", "success")
        self.fixes_applied += 1
    
    def fix_enhanced_config(self):
        """Add missing config fields"""
        print("⚙️ Checking enhanced config fields...")
        
        file_path = Path("config_enhanced.py")
        if not file_path.exists():
            self.print_status("Enhanced config file not found - skipping", "skip")
            self.fixes_skipped += 1
            return
        
        content = file_path.read_text(encoding='utf-8')
        
        # Check if fields are already added
        if "ENABLE_DASHBOARD.*Field" in content:
            self.print_status("Enhanced config already has dashboard fields", "skip")
            self.fixes_skipped += 1
            return
        
        # Check if we can add fields
        if "class" not in content or "Settings" not in content:
            self.print_status("Enhanced config doesn't match expected format", "skip")
            self.fixes_skipped += 1
            return
        
        # Apply fix
        if not self.dry_run:
            self.backup_file(file_path)
            
            dashboard_fields = '''
    # Dashboard Settings (Added for compatibility)
    ENABLE_DASHBOARD: bool = Field(default=True, description="Enable dashboard")
    ENABLE_WEBSOCKET_DASHBOARD: bool = Field(default=True, description="Enable WebSocket dashboard")
    DASHBOARD_UPDATE_INTERVAL: int = Field(default=10, description="Dashboard update interval")'''
            
            # Add before get_settings function or at end
            if "def get_settings" in content:
                new_content = content.replace("def get_settings", f"{dashboard_fields}\n\ndef get_settings")
            else:
                new_content = content + dashboard_fields
            
            file_path.write_text(new_content, encoding='utf-8')
        
        self.print_status("Enhanced config dashboard fields added", "success")
        self.fixes_applied += 1
    
    def fix_requirements(self):
        """Fix requirements.txt compatibility"""
        print("📋 Checking requirements compatibility...")
        
        file_path = Path("requirements.txt")
        if not file_path.exists():
            # Create basic requirements.txt
            basic_requirements = """# Basic requirements for LLM Proxy
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.8.6
pydantic==2.5.0
pydantic-settings==2.1.0
psutil==5.9.6
"""
            if not self.dry_run:
                file_path.write_text(basic_requirements)
            self.print_status("Created basic requirements.txt", "success")
            self.fixes_applied += 1
            return
        
        content = file_path.read_text(encoding='utf-8')
        
        # Check if already compatible
        if "aiohttp==3.8.6" in content:
            self.print_status("Requirements already has compatible aiohttp", "skip")
            self.fixes_skipped += 1
            return
        
        # Check if needs fixing
        if "aiohttp==3.9.1" not in content:
            self.print_status("Requirements doesn't have aiohttp==3.9.1 to fix", "skip")
            self.fixes_skipped += 1
            return
        
        # Apply fix
        if not self.dry_run:
            self.backup_file(file_path)
            new_content = content.replace("aiohttp==3.9.1", "aiohttp==3.8.6")
            file_path.write_text(new_content, encoding='utf-8')
        
        self.print_status("Requirements aiohttp version fixed", "success")
        self.fixes_applied += 1
    
    def install_compatible_aiohttp(self):
        """Install compatible aiohttp version"""
        print("🐍 Checking aiohttp installation...")
        
        try:
            result = subprocess.run([sys.executable, "-c", "import aiohttp; print(aiohttp.__version__)"], 
                                  capture_output=True, text=True)
            current_version = result.stdout.strip() if result.returncode == 0 else "not_installed"
            
            if current_version == "3.8.6":
                self.print_status("Compatible aiohttp version already installed", "skip")
                self.fixes_skipped += 1
                return
            
            if current_version == "not_installed":
                self.print_status("aiohttp not installed, installing compatible version...")
            else:
                self.print_status(f"Current aiohttp: {current_version}, upgrading to compatible version...")
            
            if not self.dry_run:
                result = subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp==3.8.6", "--upgrade"],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.print_status("Compatible aiohttp version installed", "success")
                    self.fixes_applied += 1
                else:
                    self.print_status(f"Failed to install aiohttp: {result.stderr}", "error")
                    self.fixes_failed += 1
            else:
                self.print_status("Would install aiohttp==3.8.6", "info")
                self.fixes_applied += 1
                
        except Exception as e:
            self.print_status(f"Error checking/installing aiohttp: {e}", "error")
            self.fixes_failed += 1
    
    def fix_missing_init_files(self):
        """Create missing __init__.py files"""
        print("📁 Checking for missing __init__.py files...")
        
        dirs = ["services", "utils", "middleware", "models"]
        created = 0
        
        for dir_name in dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and dir_path.is_dir():
                init_file = dir_path / "__init__.py"
                if not init_file.exists():
                    if not self.dry_run:
                        init_file.write_text("# Package initialization\n")
                    self.print_status(f"Created missing {init_file}")
                    created += 1
        
        if created > 0:
            self.print_status(f"Created {created} missing __init__.py files", "success")
            self.fixes_applied += 1
        else:
            self.print_status("All __init__.py files already exist", "skip")
            self.fixes_skipped += 1
    
    def fix_encoding_issues(self):
        """Fix encoding issues in Python files"""
        print("🔤 Checking for encoding issues...")
        
        fixed = 0
        checked = 0
        
        for py_file in Path(".").rglob("*.py"):
            if any(part in str(py_file) for part in [".git", "venv", "__pycache__"]):
                continue
            
            checked += 1
            
            try:
                # Try to read as UTF-8
                py_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try to fix encoding
                self.print_status(f"Fixing encoding for {py_file}")
                
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        content = py_file.read_text(encoding=encoding)
                        if not self.dry_run:
                            py_file.write_text(content, encoding='utf-8')
                        fixed += 1
                        break
                    except:
                        continue
        
        if fixed > 0:
            self.print_status(f"Fixed encoding for {fixed} files (checked {checked} total)", "success")
            self.fixes_applied += 1
        else:
            self.print_status(f"No encoding issues found (checked {checked} files)", "skip")
            self.fixes_skipped += 1
    
    def check_python_syntax(self):
        """Check Python syntax"""
        print("🐍 Checking Python syntax...")
        
        errors = 0
        files_checked = 0
        
        for py_file in Path(".").rglob("*.py"):
            if any(part in str(py_file) for part in [".git", "venv", "__pycache__"]):
                continue
            
            files_checked += 1
            
            result = subprocess.run([sys.executable, "-m", "py_compile", str(py_file)],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.print_status(f"Syntax error in {py_file}", "error")
                errors += 1
        
        if errors == 0:
            self.print_status(f"All {files_checked} Python files have valid syntax", "success")
        else:
            self.print_status(f"{errors} files have syntax errors out of {files_checked} checked", "error")
            self.fixes_failed += 1
    
    def print_summary(self):
        """Print summary of fixes"""
        print("\n" + "=" * 40)
        print("📊 Fix Summary")
        print("=" * 40)
        print(f"✅ Fixes Applied: {self.fixes_applied}")
        print(f"⏭️ Fixes Skipped: {self.fixes_skipped}")
        print(f"❌ Fixes Failed: {self.fixes_failed}")
        
        if self.dry_run:
            print("\n🔍 Dry Run Mode: No changes were made")
        
        print("\n💡 Next steps:")
        if self.fixes_applied > 0 and not self.dry_run:
            print("   1. Install dependencies: pip install -r requirements.txt")
            print("   2. Start service: python main.py")
            print("   3. Test health: curl http://localhost:8000/health")
        elif self.dry_run:
            print("   1. Run without --dry-run to apply fixes")
            print("   2. Then start with: python main.py")
        else:
            print("   1. Your setup should be ready")
            print("   2. Start with: python main.py")
    
    def run_all_fixes(self):
        """Run all compatibility fixes"""
        print("🔧 Smart Compatibility Fixer (Python)")
        print("=" * 40)
        
        if self.dry_run:
            print("🔍 Running in dry-run mode - no changes will be made\n")
        
        # Run all fixes
        self.fix_missing_init_files()
        self.fix_encoding_issues()
        self.fix_connection_pool()
        self.fix_websocket_dashboard()
        self.fix_enhanced_config()
        self.fix_requirements()
        self.install_compatible_aiohttp()
        
        print()
        self.check_python_syntax()
        
        self.print_summary()
        
        return self.fixes_failed == 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix compatibility issues for LLM Proxy")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    args = parser.parse_args()
    
    fixer = CompatibilityFixer(dry_run=args.dry_run)
    success = fixer.run_all_fixes()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())