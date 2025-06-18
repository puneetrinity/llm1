# fix_critical.py
import os
import re


def fix_caching_middleware():
    try:
        with open('middleware/caching.py', 'r') as f:
            content = f.read()

        # Fix the broken function definition
        content = content.replace(
            'return middleware_factorydef __init__(',
            'return middleware_factory\n\nclass SmartCachingMiddleware(BaseHTTPMiddleware):\n    def __init__('
        )

        with open('middleware/caching.py', 'w') as f:
            f.write(content)
        print("✅ Fixed caching middleware")
    except Exception as e:
        print(f"❌ Error fixing caching middleware: {e}")


def fix_circuit_breaker():
    try:
        with open('services/circuit_breaker.py', 'r') as f:
            lines = f.readlines()

        # Remove incomplete line
        lines = [line for line in lines if not line.strip(
        ).startswith('if self.state ==')]

        with open('services/circuit_breaker.py', 'w') as f:
            f.writelines(lines)
        print("✅ Fixed circuit breaker")
    except Exception as e:
        print(f"❌ Error fixing circuit breaker: {e}")


def fix_router_import():
    try:
        with open('services/enhanced_imports.py', 'r') as f:
            content = f.read()

        content = content.replace(
            'services.router.LLMRouter',
            'services.enhanced_router.EnhancedLLMRouter'
        )

        with open('services/enhanced_imports.py', 'w') as f:
            f.write(content)
        print("✅ Fixed router import")
    except Exception as e:
        print(f"❌ Error fixing router import: {e}")


def fix_cache_key_method():
    try:
        with open('services/enhanced_ollama_client.py', 'r') as f:
            content = f.read()

        # Replace the problematic hash method
        old_method = '''def _generate_request_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a cache key for request deduplication"""
        # Create a simple hash of the request for deduplication
        key_data = {
            'model': request_data.get('model'),
            'messages': request_data.get('messages', [])[-1:],  # Only last message
            'temperature': request_data.get('options', {}).get('temperature', 0.7)
        }
        return str(hash(str(sorted(key_data.items()))))'''

        new_method = '''def _generate_request_key(self, request_data: Dict[str, Any]) -> str:
        """Generate a deterministic cache key for request deduplication"""
        import hashlib
        import json
        key_data = {
            'model': request_data.get('model'),
            'messages': request_data.get('messages', [])[-1:],  # Only last message
            'temperature': request_data.get('options', {}).get('temperature', 0.7)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()'''

        if old_method in content:
            content = content.replace(old_method, new_method)
            print("✅ Fixed cache key method")
        else:
            print("⚠️  Cache key method not found or already fixed")

        with open('services/enhanced_ollama_client.py', 'w') as f:
            f.write(content)
    except Exception as e:
        print(f"❌ Error fixing cache key method: {e}")


if __name__ == "__main__":
    print("🔧 Starting critical fixes...")
    print(f"📁 Working directory: {os.getcwd()}")

    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ Error: main.py not found. Make sure you're in the correct repository directory.")
        input("Press Enter to exit...")
        exit(1)

    fix_caching_middleware()
    fix_circuit_breaker()
    fix_router_import()
    fix_cache_key_method()

    print("✅ All critical fixes applied!")
    print("🧪 Testing syntax...")

    # Test if files compile
    test_files = ['main.py', 'middleware/caching.py',
                  'services/circuit_breaker.py']
    for file in test_files:
        if os.path.exists(file):
            try:
                import py_compile
                py_compile.compile(file, doraise=True)
                print(f"✅ {file} - syntax OK")
            except Exception as e:
                print(f"❌ {file} - syntax error: {e}")

    print("🎉 Fix script completed!")
    input("Press Enter to close...")
