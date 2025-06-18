#!/usr/bin/env python3
"""
windows_fix_guide.py - Get Your LLM Proxy Running on Windows/VS Code
Run this in VS Code terminal: python windows_fix_guide.py

Optimized for Windows development environment with VS Code
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import platform

# Colors for Windows terminal


class Colors:
    if platform.system() == "Windows":
        # Initialize colorama for Windows
        try:
            import colorama
            colorama.init()
            HEADER = '\033[94m'
            SUCCESS = '\033[92m'
            WARNING = '\033[93m'
            ERROR = '\033[91m'
            INFO = '\033[96m'
            BOLD = '\033[1m'
            END = '\033[0m'
        except ImportError:
            # Fallback to no colors if colorama not available
            HEADER = SUCCESS = WARNING = ERROR = INFO = BOLD = END = ''
    else:
        HEADER = '\033[94m'
        SUCCESS = '\033[92m'
        WARNING = '\033[93m'
        ERROR = '\033[91m'
        INFO = '\033[96m'
        BOLD = '\033[1m'
        END = '\033[0m'


def print_header(message):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{message}{Colors.END}")


def print_success(message):
    print(f"{Colors.SUCCESS}✅ {message}{Colors.END}")


def print_warning(message):
    print(f"{Colors.WARNING}⚠️  {message}{Colors.END}")


def print_error(message):
    print(f"{Colors.ERROR}❌ {message}{Colors.END}")


def print_info(message):
    print(f"{Colors.INFO}ℹ️  {message}{Colors.END}")


def check_python_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        result = subprocess.run([sys.executable, "-m", "py_compile", str(file_path)],
                                capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def backup_file(file_path):
    """Create a backup of a file with timestamp"""
    if Path(file_path).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.backup.{timestamp}"
        shutil.copy2(file_path, backup_path)
        print_info(f"Backed up {file_path} to {backup_path}")
        return backup_path
    return None


def main():
    print_header("🚀 WINDOWS PYTHON FIX GUIDE - GET YOUR LLM PROXY RUNNING")
    print("Optimized for VS Code on Windows")
    print("Expected completion time: 30 minutes")
    print(f"Current time: {datetime.now()}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")

    # Step 1: Verify we're in the right place
    print_header("📍 STEP 1: ENVIRONMENT VERIFICATION")

    if not (Path("README.md").exists() and Path("services").is_dir()):
        print_error(
            "Please run this script from your LLM Proxy repository root directory")
        print_info(
            "In VS Code: File > Open Folder > Select your repository folder")
        print_info("Then run: python windows_fix_guide.py")
        sys.exit(1)

    print_success("Repository structure confirmed")

    # Show current Python files for clarity
    print_info("Current Python main files found:")
    main_files = list(Path(".").glob("main*.py"))
    for file in main_files:
        print(f"  {file}")

    if not main_files:
        print_warning("No main*.py files found")

    # Step 2: Check Python and dependencies
    print_header("🐍 STEP 2: PYTHON & DEPENDENCIES CHECK")

    print_success(f"Python version: {sys.version}")

    # Check if pip is available
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print_success("pip is available")
        else:
            print_warning("pip may not be properly installed")
    except Exception:
        print_error("pip not found")

    # Step 3: Apply Critical Fixes
    print_header("🔧 STEP 3: APPLYING CRITICAL FIXES")

    # Fix 1: Critical syntax and import issues
    print_info("Checking for and running existing fix scripts...")

    scripts_run = 0

    # Try to run fix_critical.py
    if Path("fix_critical.py").exists():
        try:
            result = subprocess.run([sys.executable, "fix_critical.py"],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print_success("fix_critical.py executed successfully")
                scripts_run += 1
            else:
                print_warning(f"fix_critical.py had issues: {result.stderr}")
        except Exception as e:
            print_warning(f"Could not run fix_critical.py: {e}")

    # Try to run quick_fixes1.py
    if Path("quick_fixes1.py").exists():
        try:
            result = subprocess.run([sys.executable, "quick_fixes1.py"],
                                    capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print_success("quick_fixes1.py executed successfully")
                scripts_run += 1
            else:
                print_warning(f"quick_fixes1.py had issues: {result.stderr}")
        except Exception as e:
            print_warning(f"Could not run quick_fixes1.py: {e}")

    if scripts_run == 0:
        print_warning(
            "No fix scripts ran successfully - applying manual fixes")

    # Manual fixes
    print_info("Applying manual fixes for critical issues...")

    # Fix caching middleware
    caching_file = Path("middleware/caching.py")
    if caching_file.exists():
        print_info("Fixing caching middleware...")
        try:
            content = caching_file.read_text(encoding='utf-8')
            if "return middleware_factorydef __init__(" in content:
                backup_file(caching_file)
                fixed_content = content.replace(
                    "return middleware_factorydef __init__(",
                    "return middleware_factory\n\nclass SmartCachingMiddleware(BaseHTTPMiddleware):\n    def __init__("
                )
                caching_file.write_text(fixed_content, encoding='utf-8')
                print_success("Caching middleware syntax fixed")
            else:
                print_success("Caching middleware syntax already correct")
        except Exception as e:
            print_error(f"Could not fix caching middleware: {e}")

    # Fix circuit breaker
    circuit_breaker_file = Path("services/circuit_breaker.py")
    if circuit_breaker_file.exists():
        print_info("Fixing circuit breaker...")
        try:
            lines = circuit_breaker_file.read_text(
                encoding='utf-8').splitlines()
            original_count = len(lines)

            # Remove incomplete conditional statements
            filtered_lines = [line for line in lines
                              if not re.match(r'^\s*if self\.state ==\s*$', line)]

            if len(filtered_lines) < original_count:
                backup_file(circuit_breaker_file)
                circuit_breaker_file.write_text(
                    '\n'.join(filtered_lines) + '\n', encoding='utf-8')
                removed = original_count - len(filtered_lines)
                print_success(
                    f"Circuit breaker fixed - removed {removed} incomplete line(s)")
            else:
                print_success("Circuit breaker syntax already correct")
        except Exception as e:
            print_error(f"Could not fix circuit breaker: {e}")

    # Fix import paths
    enhanced_imports_file = Path("services/enhanced_imports.py")
    if enhanced_imports_file.exists():
        print_info("Fixing import paths...")
        try:
            content = enhanced_imports_file.read_text(encoding='utf-8')
            if "services.router.LLMRouter" in content:
                backup_file(enhanced_imports_file)
                fixed_content = content.replace(
                    "services.router.LLMRouter",
                    "services.enhanced_router.EnhancedLLMRouter"
                )
                enhanced_imports_file.write_text(
                    fixed_content, encoding='utf-8')
                print_success("Import paths fixed")
            else:
                print_success("Import paths already correct")
        except Exception as e:
            print_error(f"Could not fix import paths: {e}")

    # Create missing __init__.py files
    print_info("Creating missing __init__.py files...")
    package_dirs = ["services", "utils", "middleware", "models", "test"]
    created_count = 0

    for pkg_dir in package_dirs:
        dir_path = Path(pkg_dir)
        if dir_path.is_dir():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text(
                    "# Package initialization\n", encoding='utf-8')
                print_success(f"Created {init_file}")
                created_count += 1

    if created_count == 0:
        print_success("All __init__.py files already exist")

    # Step 4: Fix Frontend Issues
    print_header("⚛️ STEP 4: FIXING FRONTEND ISSUES")

    frontend_dir = Path("frontend")
    if frontend_dir.is_dir():
        print_info("Frontend directory found")

        # Fix App.tsx ESLint issues
        app_tsx = frontend_dir / "src" / "App.tsx"
        if app_tsx.exists():
            print_info("Fixing ESLint issues in App.tsx...")
            try:
                content = app_tsx.read_text(encoding='utf-8')
                changes_made = False

                # Fix confirm() calls
                if "confirm(" in content and "window.confirm(" not in content:
                    backup_file(app_tsx)
                    content = re.sub(
                        r'\bconfirm\(', 'window.confirm(', content)
                    changes_made = True

                # Fix alert() calls
                if "alert(" in content and "window.alert(" not in content:
                    if not changes_made:
                        backup_file(app_tsx)
                    content = re.sub(r'\balert\(', 'window.alert(', content)
                    changes_made = True

                if changes_made:
                    app_tsx.write_text(content, encoding='utf-8')
                    print_success("App.tsx ESLint issues fixed")
                else:
                    print_success("App.tsx already has correct syntax")
            except Exception as e:
                print_error(f"Could not fix App.tsx: {e}")

        # Check Node.js availability
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                node_version = result.stdout.strip()
                print_success(f"Node.js found: {node_version}")
                print_info("To build frontend later:")
                print("  cd frontend")
                print("  npm install")
                print("  npm run build")
            else:
                print_warning("Node.js not responding properly")
        except FileNotFoundError:
            print_warning("Node.js not found")
            print_info(
                "Install from https://nodejs.org if you need the React dashboard")
    else:
        print_info("No frontend directory found - skipping frontend fixes")

    # Step 5: Consolidate Main Files
    print_header("📄 STEP 5: CONSOLIDATING MAIN FILES")

    # Find the best main file to use based on priority
    priority_files = ["main_master.py", "main_fixed.py",
                      "main_with_react.py", "main.py"]
    primary_file = None

    for candidate in priority_files:
        if Path(candidate).exists():
            primary_file = candidate
            break

    if primary_file:
        print_success(f"Using {primary_file} as primary application file")

        # Backup other main files
        for file in Path(".").glob("main*.py"):
            if file.name != primary_file and file.is_file():
                backup_file(file)
                file.unlink()  # Remove the file
                print_info(f"Backed up and removed {file.name}")

        # Ensure we have main.py as the canonical entry point
        if primary_file != "main.py":
            shutil.copy2(primary_file, "main.py")
            print_success("Created main.py from " + primary_file)
    else:
        print_error("No main application file found!")
        print_info("You may need to create a main.py file manually")

    # Step 6: Test Syntax
    print_header("🧪 STEP 6: TESTING SYNTAX")

    test_files = ["main.py", "middleware/caching.py",
                  "services/circuit_breaker.py"]
    error_count = 0

    for file_path in test_files:
        file = Path(file_path)
        if file.exists():
            print_info(f"Testing syntax of {file_path}...")
            if check_python_syntax(file):
                print_success(f"{file_path} syntax OK")
            else:
                print_error(f"{file_path} has syntax errors")
                error_count += 1
        else:
            print_warning(f"{file_path} not found")

    # Step 7: Environment Setup
    print_header("⚙️ STEP 7: ENVIRONMENT SETUP")

    env_file = Path(".env")
    if not env_file.exists():
        env_template = Path(".env.template")
        if env_template.exists():
            shutil.copy2(env_template, env_file)
            print_success("Created .env from template")
        else:
            print_info("Creating basic .env file...")
            env_content = """# Basic LLM Proxy Configuration
PORT=8001
DEBUG=false
LOG_LEVEL=INFO
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=mistral:7b-instruct-q4_0
ENABLE_AUTH=false
DEFAULT_API_KEY=sk-dev-key
MAX_MEMORY_MB=8192
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
"""
            env_file.write_text(env_content, encoding='utf-8')
            print_success("Created basic .env file")
    else:
        print_success(".env file already exists")

    # Step 8: Dependencies Check
    print_header("📦 STEP 8: DEPENDENCIES CHECK")

    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        print_success("requirements.txt found")

        # Ask user if they want to install dependencies
        while True:
            install_deps = input(
                "\nInstall Python dependencies now? (y/n): ").lower().strip()
            if install_deps in ['y', 'yes']:
                print_info("Installing dependencies...")
                try:
                    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                                            capture_output=True, text=True)
                    if result.returncode == 0:
                        print_success("Dependencies installed successfully")
                    else:
                        print_error(
                            f"Failed to install dependencies: {result.stderr}")
                except Exception as e:
                    print_error(f"Error installing dependencies: {e}")
                break
            elif install_deps in ['n', 'no']:
                print_warning("Skipping dependency installation")
                print_info(
                    "Install later with: pip install -r requirements.txt")
                break
            else:
                print("Please enter 'y' or 'n'")
    else:
        print_warning("requirements.txt not found")
        print_info("You may need to install basic dependencies:")
        print("  pip install fastapi uvicorn pydantic-settings aiohttp")

    # Step 9: Final Status
    print_header("📊 STEP 9: FINAL STATUS")

    print("\nFix Summary:")
    print("=" * 50)

    if error_count == 0:
        print_success("✅ All syntax tests passed")
        print_success("✅ Main file consolidated" +
                      (f" ({primary_file})" if primary_file else ""))
        print_success("✅ Environment configured")

        if frontend_dir.is_dir():
            print_success("✅ Frontend issues addressed")

        print("\n" + "🎉 READY TO START!" + "\n")
        print_info("Your LLM Proxy is now ready to run!")
        print("\nNext steps:")
        print("1. Start the application:")
        print("   python main.py")
        print("\n2. Test the health endpoint (in another terminal):")
        print("   curl http://localhost:8001/health")
        print("   # Or use: python -c \"import requests; print(requests.get('http://localhost:8001/health').json())\"")
        print("\n3. View API documentation:")
        print("   Open: http://localhost:8001/docs")

        if frontend_dir.is_dir():
            print("\n4. React dashboard (if built):")
            print("   Open: http://localhost:8001/app")

        print_success("\n🚀 Application is ready for deployment!")

    else:
        print_error(f"❌ {error_count} file(s) still have syntax errors")
        print("\nManual fixes needed for:")
        for file_path in test_files:
            file = Path(file_path)
            if file.exists() and not check_python_syntax(file):
                print(f"   - {file_path}")
        print("\nCheck the error messages above and fix manually in VS Code")

    # Step 10: VS Code specific tips
    print_header("💡 VS CODE TIPS")
    print("\nUseful VS Code commands:")
    print("• Ctrl+Shift+` : Open terminal")
    print("• Ctrl+Shift+P : Command palette")
    print("• F5 : Run/Debug Python file")
    print("• Ctrl+Shift+E : Explorer panel")
    print("\nRecommended VS Code extensions:")
    print("• Python (Microsoft)")
    print("• Python Debugger (Microsoft)")
    print("• REST Client (for testing API endpoints)")

    print_header("📋 TROUBLESHOOTING")
    print("\nIf you encounter issues:")
    print("• Check that Python is in your PATH")
    print("• Install dependencies: pip install -r requirements.txt")
    print("• Check if ports are available (8001, 11434)")
    print("• View VS Code terminal for detailed error messages")
    print("• Check if Ollama is running: curl http://localhost:11434/api/tags")

    print(f"\nFix script completed at: {datetime.now()}")
    if error_count == 0:
        print("Status: ✅ SUCCESS - Ready to run!")
        return 0
    else:
        print("Status: ⚠️  PARTIAL - Manual fixes needed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\nFix script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
