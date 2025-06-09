# security/config.py - Enhanced Security Configuration
import os
import secrets
import hashlib
import logging
from typing import Dict, List, Optional
from pydantic import validator, Field
from pydantic_settings import BaseSettings

class SecuritySettings(BaseSettings):
    """Enhanced security settings with proper validation"""
    
    # Authentication
    ENABLE_AUTH: bool = Field(default=False, description="Enable API key authentication")
    API_KEY_HEADER: str = Field(default="X-API-Key", description="Header name for API key")
    
    # API Key validation
    DEFAULT_API_KEY: Optional[str] = Field(default=None, description="Default API key (use only for development)")
    REQUIRE_STRONG_API_KEYS: bool = Field(default=True, description="Require strong API keys in production")
    MIN_API_KEY_LENGTH: int = Field(default=32, description="Minimum API key length")
    
    # Environment detection
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    
    # Rate limiting
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Enable rate limiting")
    DEFAULT_RATE_LIMIT: int = Field(default=60, description="Default requests per minute")
    
    # CORS settings
    CORS_ORIGINS: List[str] = Field(default=["*"], description="Allowed CORS origins")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow credentials in CORS")
    
    # Security headers
    ENABLE_SECURITY_HEADERS: bool = Field(default=True, description="Enable security headers")
    
    # Request validation
    MAX_REQUEST_SIZE: int = Field(default=10_000_000, description="Maximum request size in bytes (10MB)")
    MAX_TOKENS: int = Field(default=8192, description="Maximum tokens per request")
    
    @validator('DEFAULT_API_KEY')
    def validate_api_key(cls, v, values):
        """Validate API key security"""
        environment = values.get('ENVIRONMENT', 'development')
        require_strong = values.get('REQUIRE_STRONG_API_KEYS', True)
        
        if not v:
            if environment == 'production':
                raise ValueError("DEFAULT_API_KEY must be set in production environment")
            elif values.get('ENABLE_AUTH', False):
                # Generate a secure default for development
                v = f"sk-dev-{secrets.token_urlsafe(32)}"
                logging.warning(f"Generated temporary API key for development: {v[:16]}...")
        
        if v and require_strong:
            # Validate key strength
            if len(v) < values.get('MIN_API_KEY_LENGTH', 32):
                raise ValueError(f"API key must be at least {values.get('MIN_API_KEY_LENGTH', 32)} characters long")
            
            if v in WEAK_API_KEYS:
                raise ValueError("API key is too weak. Please use a cryptographically secure key.")
            
            # Check for common patterns
            if v.lower() in ['sk-default', 'sk-test', 'test-key', 'default-key']:
                if environment == 'production':
                    raise ValueError("Cannot use default/test API keys in production")
                else:
                    logging.warning("Using weak API key in development environment")
        
        return v
    
    @validator('CORS_ORIGINS')
    def validate_cors_origins(cls, v, values):
        """Validate CORS origins for production"""
        environment = values.get('ENVIRONMENT', 'development')
        
        if environment == 'production' and '*' in v:
            logging.warning("Using wildcard CORS origins in production is not recommended")
        
        return v
    
    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Validate environment value"""
        valid_environments = ['development', 'staging', 'production']
        if v not in valid_environments:
            raise ValueError(f"ENVIRONMENT must be one of: {valid_environments}")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# List of weak/common API keys that should be rejected
WEAK_API_KEYS = {
    "sk-default",
    "sk-test", 
    "test-key",
    "default-key",
    "api-key",
    "secret-key",
    "password",
    "123456",
    "admin",
    "root",
    "changeme"
}

class APIKeyGenerator:
    """Secure API key generation and validation"""
    
    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a cryptographically secure API key"""
        return f"{prefix}-{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def validate_api_key_strength(api_key: str) -> Dict[str, bool]:
        """Validate API key strength"""
        checks = {
            "min_length": len(api_key) >= 32,
            "not_weak": api_key.lower() not in WEAK_API_KEYS,
            "has_entropy": len(set(api_key)) > 10,  # At least 10 unique characters
            "proper_format": api_key.startswith(("sk-", "pk-")) and "-" in api_key
        }
        
        return {
            **checks,
            "is_strong": all(checks.values())
        }

def setup_security_middleware(app, settings: SecuritySettings):
    """Setup security middleware based on configuration"""
    
    # Security headers middleware
    if settings.ENABLE_SECURITY_HEADERS:
        from fastapi.middleware.trustedhost import TrustedHostMiddleware
        from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
        
        @app.middleware("http")
        async def add_security_headers(request, call_next):
            response = await call_next(request)
            
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY" 
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Only add HSTS in production with HTTPS
            if settings.ENVIRONMENT == "production":
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            return response
    
    # Request size limiting
    @app.middleware("http")
    async def limit_request_size(request, call_next):
        if hasattr(request, 'headers'):
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > settings.MAX_REQUEST_SIZE:
                from fastapi import HTTPException
                raise HTTPException(status_code=413, detail="Request too large")
        
        return await call_next(request)

# Enhanced .env.template with security considerations
ENV_TEMPLATE_SECURE = """# .env.template - Secure Environment Configuration
# Copy this file to .env and modify values as needed

# CRITICAL: Change these values before deploying to production!

# Environment (REQUIRED: set to 'production' for production deployment)
ENVIRONMENT=development

# Basic Configuration
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=300
OLLAMA_MAX_RETRIES=3

# Authentication (REQUIRED for production)
ENABLE_AUTH=true
API_KEY_HEADER=X-API-Key

# SECURITY WARNING: Generate a strong API key for production!
# Use: python -c "import secrets; print(f'sk-{secrets.token_urlsafe(32)}')"
# DEFAULT_API_KEY=sk-CHANGE-THIS-IN-PRODUCTION

# Security Settings
REQUIRE_STRONG_API_KEYS=true
MIN_API_KEY_LENGTH=32
ENABLE_SECURITY_HEADERS=true
MAX_REQUEST_SIZE=10485760  # 10MB

# CORS Settings (restrict in production)
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true

# Rate Limiting
ENABLE_RATE_LIMITING=true
DEFAULT_RATE_LIMIT=60

# Cache Settings
ENABLE_CACHE=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Model Settings
DEFAULT_MODEL=mistral:7b-instruct-q4_0
MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7

# Memory Management (MB)
MAX_MEMORY_MB=8192
CACHE_MEMORY_LIMIT_MB=1024
MODEL_MEMORY_LIMIT_MB=4096

# Resource Limits
MAX_CONCURRENT_REQUESTS=10
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=300

# Enhanced Features (set carefully based on available resources)
ENABLE_SEMANTIC_CLASSIFICATION=false
ENABLE_STREAMING=true
ENABLE_MODEL_WARMUP=true
ENABLE_DETAILED_METRICS=true

# Enhanced Feature Memory Limits
SEMANTIC_MODEL_MAX_MEMORY_MB=500
FAISS_INDEX_MAX_SIZE=10000
CLASSIFICATION_CACHE_MAX_SIZE=1000

# GPU Configuration (for RunPod/GPU environments)
GPU_MEMORY_FRACTION=0.9
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Production Deployment Checklist:
# [ ] Set ENVIRONMENT=production
# [ ] Generate and set strong DEFAULT_API_KEY
# [ ] Restrict CORS_ORIGINS to your domains only
# [ ] Set appropriate rate limits
# [ ] Review memory limits for your hardware
# [ ] Enable HTTPS in reverse proxy
# [ ] Set up proper logging and monitoring
"""

# Security validation script
def validate_production_config(settings: SecuritySettings) -> List[str]:
    """Validate configuration for production deployment"""
    
    issues = []
    
    if settings.ENVIRONMENT == "production":
        # Critical security checks
        if not settings.ENABLE_AUTH:
            issues.append("CRITICAL: Authentication must be enabled in production")
        
        if not settings.DEFAULT_API_KEY or settings.DEFAULT_API_KEY in WEAK_API_KEYS:
            issues.append("CRITICAL: Strong API key required in production")
        
        if "*" in settings.CORS_ORIGINS:
            issues.append("HIGH: Wildcard CORS origins not recommended in production")
        
        if not settings.ENABLE_RATE_LIMITING:
            issues.append("MEDIUM: Rate limiting should be enabled in production")
        
        if settings.DEBUG:
            issues.append("MEDIUM: Debug mode should be disabled in production")
        
        if not settings.ENABLE_SECURITY_HEADERS:
            issues.append("MEDIUM: Security headers should be enabled in production")
    
    return issues

# CLI script for security setup
def generate_secure_config():
    """Generate secure configuration for production"""
    
    print("🔒 LLM Proxy Security Configuration Generator")
    print("=" * 50)
    
    # Generate API key
    api_key = APIKeyGenerator.generate_api_key()
    print(f"✅ Generated secure API key: {api_key}")
    
    # Validate strength
    strength = APIKeyGenerator.validate_api_key_strength(api_key)
    print(f"🔍 Key strength validation: {'✅ STRONG' if strength['is_strong'] else '❌ WEAK'}")
    
    # Generate secure .env
    secure_env = ENV_TEMPLATE_SECURE.replace(
        "# DEFAULT_API_KEY=sk-CHANGE-THIS-IN-PRODUCTION",
        f"DEFAULT_API_KEY={api_key}"
    )
    
    with open(".env.production", "w") as f:
        f.write(secure_env)
    
    print("📝 Created .env.production with secure defaults")
    print("\n🚨 IMPORTANT SECURITY REMINDERS:")
    print("1. Review and customize CORS_ORIGINS for your domains")
    print("2. Set up HTTPS with a reverse proxy (nginx/caddy)")
    print("3. Configure proper firewall rules")
    print("4. Set up log monitoring and alerting")
    print("5. Regularly rotate API keys")
    print("6. Monitor rate limiting and adjust as needed")

if __name__ == "__main__":
    generate_secure_config()
