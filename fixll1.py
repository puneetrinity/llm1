import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

SERVICE_GLOBALS = [
    'ollama_client',
    'cache_service',
    'circuit_breaker',
    'router',
    'warmup_service',
    'auth_middleware',
    'rate_limiter',
]

SERVICE_IMPORTS = [
    ('OllamaClient', 'services.ollama_client'),
    ('CacheService', 'services.cache_service'),
    ('CircuitBreaker', 'services.circuit_breaker'),
    ('EnhancedLLMRouter', 'services.optimized_router'),
    ('ModelWarmupService', 'services.model_warmup'),
    ('SemanticIntentClassifier', 'services.semantic_classifier'),
    ('AuthMiddleware', 'middleware.auth'),
    ('RateLimiter', 'middleware.rate_limiter')
]

MAIN_FILES = ['main.py', 'main_master.py']

def scan_for_duplicates():
    """Warn about duplicate class definitions across the codebase."""
    class_defs = {}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                for match in re.finditer(r'class\s+([A-Za-z0-9_]+)\s*\(', content):
                    class_name = match.group(1)
                    class_defs.setdefault(class_name, []).append(str(path))
    for k, v in class_defs.items():
        if len(v) > 1 and k.lower().endswith("client") or k.lower().endswith("service") or k.lower().endswith("router"):
            print(f"WARNING: Duplicate class definition for {k} in: {v}")

def fix_main_py():
    """Fix main.py to remove globals and use app.state."""
    for filename in MAIN_FILES:
        main_path = PROJECT_ROOT / filename
        if not main_path.exists():
            continue
        with open(main_path, "r", encoding="utf-8") as f:
            code = f.read()
        original_code = code

        # Remove global service variables
        for global_var in SERVICE_GLOBALS:
            code = re.sub(
                rf"{global_var}:[^\n]*\n", "", code, flags=re.MULTILINE
            )
            code = re.sub(
                rf"\bglobal\s+{global_var}\b[^\n]*\n", "", code, flags=re.MULTILINE
            )

        # Replace direct service usage with app.state
        for global_var in SERVICE_GLOBALS:
            code = re.sub(
                rf"\b{global_var}\b",
                f"request.app.state.{global_var}",
                code
            )
        # Insert correct imports from services/
        for cls, module in SERVICE_IMPORTS:
            if cls in code and f"from {module} import {cls}" not in code:
                code = f"from {module} import {cls}\n" + code

        # Replace service initialization to attach to app.state
        code = re.sub(
            r"(\w+)\s*=\s*([A-Za-z0-9_]+)\(([^)]*)\)",
            r"app.state.\1 = \2(\3)",
            code
        )

        # Save changes if any
        if code != original_code:
            backup = main_path.with_suffix('.bak')
            print(f"Backing up {main_path} to {backup}")
            with open(backup, "w", encoding="utf-8") as f:
                f.write(original_code)
            print(f"Writing fixed {main_path}")
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(code)
        else:
            print(f"No changes needed in {filename}")

def warn_multiple_entrypoints():
    entrypoints = [f for f in MAIN_FILES if (PROJECT_ROOT / f).exists()]
    if len(entrypoints) > 1:
        print(f"WARNING: Multiple entry points found: {entrypoints}. Consider keeping only one (e.g., main.py).")

def warn_silent_fallbacks():
    for root, dirs, files in os.walk(PROJECT_ROOT):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "try:" in content and "except ImportError" in content and "logging." not in content:
                    print(f"WARNING: Possible silent fallback in {path}. Add logging or raise explicitly.")

def main():
    print("=== LLM1 Codebase Fixer ===")
    scan_for_duplicates()
    fix_main_py()
    warn_multiple_entrypoints()
    warn_silent_fallbacks()
    print("=== Done ===")

if __name__ == "__main__":
    main()