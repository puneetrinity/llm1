# quick_fixes.py - Fix Critical Issues Quickly
import os
import sys
from pathlib import Path

def fix_critical_issues():
    """Fix the most critical issues preventing the app from running"""
    print("🔧 Applying critical fixes...")
    
    # Fix 1: Create missing __init__.py files
    create_missing_init_files()
    
    # Fix 2: Fix common import issues
    fix_import_issues()
    
    # Fix 3: Create a minimal working main.py if needed
    ensure_working_main()
    
    # Fix 4: Fix encoding issues
    fix_encoding_issues()
    
    print("✅ Critical fixes applied!")

def create_missing_init_files():
    """Create missing __init__.py files"""
    print("📁 Creating missing __init__.py files...")
    
    package_dirs = ['services', 'utils', 'middleware', 'models', 'test']
    
    for pkg_dir in package_dirs:
        if Path(pkg_dir).exists() and Path(pkg_dir).is_dir():
            init_file = Path(pkg_dir) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('# Package initialization\n')
                print(f"  ✅ Created {init_file}")

def fix_import_issues():
    """Fix common import issues"""
    print("📦 Fixing import issues...")
    
    # Fix services/__init__.py to avoid circular imports
    services_init = Path('services/__init__.py')
    if services_init.exists():
        content = '''# services/__init__.py - Safe imports without circular dependencies
"""
Core services package with safe import handling
"""

# Basic imports that are always safe
try:
    from .ollama_client import OllamaClient
except ImportError as e:
    print(f"Warning: Could not import OllamaClient: {e}")
    OllamaClient = None

try:
    from .router import LLMRouter
except ImportError as e:
    print(f"Warning: Could not import LLMRouter: {e}")
    LLMRouter = None

try:
    from .auth import AuthService
except ImportError as e:
    print(f"Warning: Could not import AuthService: {e}")
    AuthService = None

# Enhanced imports with fallbacks
try:
    from .circuit_breaker import CircuitBreakerManager, get_circuit_breaker_manager
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError as e:
    print(f"Info: Circuit breaker not available: {e}")
    CircuitBreakerManager = None
    get_circuit_breaker_manager = lambda: None
    CIRCUIT_BREAKER_AVAILABLE = False

__all__ = [
    "OllamaClient",
    "LLMRouter", 
    "AuthService",
    "CIRCUIT_BREAKER_AVAILABLE"
]

if CIRCUIT_BREAKER_AVAILABLE:
    __all__.extend(["CircuitBreakerManager", "get_circuit_breaker_manager"])
'''
        services_init.write_text(content)
        print("  ✅ Fixed services/__init__.py")
    
    # Fix utils/__init__.py
    utils_init = Path('utils/__init__.py')
    if utils_init.exists():
        content = '''# utils/__init__.py - Safe utility imports
"""
Utilities package with graceful import handling
"""

# Core utilities (always needed)
try:
    from .metrics import MetricsCollector
except ImportError as e:
    print(f"Warning: Could not import MetricsCollector: {e}")
    MetricsCollector = None

try:
    from .health import HealthChecker
except ImportError as e:
    print(f"Warning: Could not import HealthChecker: {e}")
    HealthChecker = None

# Memory manager (important for enhanced features)
try:
    from .memory_manager import get_memory_manager, MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Info: Memory manager not available: {e}")
    get_memory_manager = lambda: None
    MemoryManager = None
    MEMORY_MANAGER_AVAILABLE = False

__all__ = [
    "MetricsCollector",
    "HealthChecker",
    "MEMORY_MANAGER_AVAILABLE"
]

if MEMORY_MANAGER_AVAILABLE:
    __all__.extend(["get_memory_manager", "MemoryManager"])
'''
        utils_init.write_text(content)
        print("  ✅ Fixed utils/__init__.py")

def ensure_working_main():
    """Ensure there's a working main.py"""
    print("🚀 Ensuring working main.py...")
    
    if not Path('main.py').exists():
        print("  ⚠️  Creating minimal main.py...")
        create_minimal_main()
        return
    
    # Check if main.py has critical issues
    try:
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common issues
        issues = []
        if 'app = FastAPI' not in content:
            issues.append("Missing FastAPI app creation")
        
        if 'if __name__ == "__main__"' not in content:
            issues.append("Missing main guard")
        
        if issues:
            print(f"  ⚠️  Issues found in main.py: {', '.join(issues)}")
            backup_and_create_minimal_main()
        else:
            print("  ✅ main.py looks good")
            
    except Exception as e:
        print(f"  ❌ Could not read main.py: {e}")
        backup_and_create_minimal_main()

def create_minimal_main():
    """Create a minimal working main.py"""
    content = '''# main.py - Minimal Working LLM Proxy
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Proxy",
    description="Basic LLM routing proxy",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None

# Global variables
ollama_client = None
llm_router = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ollama_client, llm_router
    
    logger.info("🚀 Starting LLM Proxy...")
    
    # Try to initialize services
    try:
        # Import with fallbacks
        try:
            from services.ollama_client import OllamaClient
            ollama_client = OllamaClient()
            await ollama_client.initialize()
            logger.info("✅ Ollama client initialized")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize Ollama client: {e}")
        
        try:
            from services.router import LLMRouter
            llm_router = LLMRouter(ollama_client)
            await llm_router.initialize()
            logger.info("✅ LLM router initialized")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize router: {e}")
            
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if ollama_client else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "ollama_client": ollama_client is not None,
            "llm_router": llm_router is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Proxy API",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Basic chat completions endpoint"""
    
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    try:
        # Basic request processing
        if llm_router:
            response = await llm_router.process_chat_completion(request, request.model)
        else:
            # Fallback direct processing
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            response = await ollama_client.generate_completion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 150
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
'''
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  ✅ Created minimal working main.py")

def backup_and_create_minimal_main():
    """Backup existing main.py and create minimal version"""
    if Path('main.py').exists():
        backup_name = f'main.py.backup.{int(datetime.now().timestamp())}'
        Path('main.py').rename(backup_name)
        print(f"  📦 Backed up existing main.py to {backup_name}")
    
    create_minimal_main()

def fix_encoding_issues():
    """Fix encoding issues in Python files"""
    print("🔤 Fixing encoding issues...")
    
    python_files = list(Path('.').rglob('*.py'))
    fixed_count = 0
    
    for file_path in python_files:
        if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'venv']):
            continue
        
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # Try other common encodings
                for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        
                        # Re-save as UTF-8
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        fixed_count += 1
                        print(f"  🔧 Fixed encoding for {file_path}")
                        break
                    except:
                        continue
            except Exception as e:
                print(f"  ❌ Could not fix encoding for {file_path}: {e}")
    
    if fixed_count > 0:
        print(f"  ✅ Fixed encoding for {fixed_count} files")
    else:
        print("  ✅ No encoding issues found")

def create_basic_requirements():
    """Create basic requirements.txt if missing"""
    if not Path('requirements.txt').exists():
        print("📋 Creating basic requirements.txt...")
        
        requirements = '''# Basic requirements for LLM Proxy
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.1
pydantic==2.5.0
pydantic-settings==2.1.0
psutil==5.9.6

# Optional enhanced features
redis>=4.5.0
sentence-transformers>=2.2.0
sse-starlette>=1.6.5
'''
        
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        
        print("  ✅ Created basic requirements.txt")

if __name__ == "__main__":
    print("🚀 Running critical fixes...")
    
    fix_critical_issues()
    create_basic_requirements()
    
    print("\n✅ Critical fixes complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test the application: python main.py")
    print("3. Check health: curl http://localhost:8000/health")
    print("\nIf you still have issues, run: python repo_health_check.py")