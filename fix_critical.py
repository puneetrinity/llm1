# fix_critical.py
import os
import re

def fix_caching_middleware():
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

def fix_circuit_breaker():
    with open('services/circuit_breaker.py', 'r') as f:
        lines = f.readlines()
    
    # Remove incomplete line
    lines = [line for line in lines if not line.strip().startswith('if self.state ==')]
    
    with open('services/circuit_breaker.py', 'w') as f:
        f.writelines(lines)
    print("✅ Fixed circuit breaker")

def fix_router_import():
    with open('services/enhanced_imports.py', 'r') as f:
        content = f.read()
    
    content = content.replace(
        'services.router.LLMRouter',
        'services.enhanced_router.EnhancedLLMRouter'
    )
    
    with open('services/enhanced_imports.py', 'w') as f:
        f.write(content)
    print("✅ Fixed router import")

if __name__ == "__main__":
    print("🔧 Fixing critical issues...")
    fix_caching_middleware()
    fix_circuit_breaker()
    fix_router_import()
    print("✅ All critical fixes applied!")
