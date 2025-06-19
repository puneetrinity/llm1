from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Optional, Literal
import secrets
import logging
import json


class Settings(BaseSettings):
    # CRITICAL: Pydantic v2 fix
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True
    )

    # --- Core server settings ---
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: str = "INFO"

    # --- Ollama configuration ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300
    DEFAULT_MODEL: str = "mistral:7b-instruct-q4_0"

    # --- CORS settings ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001", "*"]
    CORS_ALLOW_CREDENTIALS: bool = True

    # --- Authentication ---
    ENABLE_AUTH: bool = False
    DEFAULT_API_KEY: str = "sk-dev-key-change-in-production"
    API_KEY_HEADER: str = "X-API-Key"

    # --- Feature toggles ---
    ENABLE_ENHANCED_FEATURES: bool = False
    ENABLE_DASHBOARD: bool = True
    ENABLE_WEBSOCKET: bool = False

    # --- Memory and performance ---
    MAX_MEMORY_MB: int = 4096
    CACHE_MEMORY_LIMIT_MB: int = 512

    # --- Security-specific settings ---
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT tokens and encryption",
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT signing algorithm")
    JWT_EXPIRATION_MINUTES: int = Field(
        default=30, description="JWT token expiration time in minutes"
    )
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development", description="Application environment"
    )

    # --- Rate limiting ---
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    DEFAULT_RATE_LIMIT: str = Field(
        default="100/hour", description="Default rate limit per user"
    )

    # --- Security headers ---
    SECURITY_HEADERS_ENABLED: bool = Field(
        default=True, description="Enable security headers middleware"
    )

    # --- API Keys (if using API key authentication) ---
    API_KEYS: List[str] = Field(
        default=[], description="Valid API keys for authentication"
    )

    # --- Database security (if applicable) ---
    DATABASE_ENCRYPTION_KEY: Optional[str] = Field(
        default=None, description="Database encryption key"
    )

    # --- TLS/SSL settings ---
    USE_TLS: bool = Field(default=False, description="Force HTTPS/TLS")
    TLS_CERT_FILE: Optional[str] = Field(
        default=None, description="Path to TLS certificate file"
    )
    TLS_KEY_FILE: Optional[str] = Field(
        default=None, description="Path to TLS private key file"
    )

    # --- Logging and monitoring ---
    ENABLE_DETAILED_LOGGING: bool = Field(
        default=False, description="Enable detailed request/response logging"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


# Validation and middleware helpers (from security/config.py)
def validate_production_config(settings: Settings) -> List[str]:
    """Validate production security configuration"""
    issues = []

    if settings.ENVIRONMENT == "production":
        # Critical security checks
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            issues.append(
                "CRITICAL: SECRET_KEY must be at least 32 characters in production"
            )

        if not settings.USE_TLS:
            issues.append("WARNING: TLS should be enabled in production")

        if settings.CORS_ORIGINS == ["*"]:
            issues.append("WARNING: CORS origins should be restricted in production")

        if not settings.RATE_LIMIT_ENABLED:
            issues.append("WARNING: Rate limiting should be enabled in production")

        if settings.ENABLE_DETAILED_LOGGING:
            issues.append(
                "INFO: Detailed logging enabled - ensure log rotation is configured"
            )

    return issues


def setup_security_middleware(app, settings: Settings):
    """Setup security middleware based on configuration"""

    if settings.SECURITY_HEADERS_ENABLED:
        try:
            from fastapi.middleware.trustedhost import TrustedHostMiddleware
            from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

            if settings.USE_TLS:
                app.add_middleware(HTTPSRedirectMiddleware)

            # Add trusted host middleware in production
            if settings.ENVIRONMENT == "production":
                app.add_middleware(
                    TrustedHostMiddleware,
                    allowed_hosts=["*"],  # Configure with actual hosts
                )
        except ImportError:
            logging.warning("Some security middleware not available")

    logging.info(
        f"Security middleware configured for {settings.ENVIRONMENT} environment"
    )


# Global settings instance
settings = Settings()
