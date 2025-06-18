# security/config.py - Fixed Security Configuration
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Literal
import os
import secrets
import logging


class SecuritySettings(BaseSettings):
    """Security configuration with proper field definitions"""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore'  # This allows extra env vars without errors
    )

    # Security-specific settings
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT tokens and encryption"
    )

    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    JWT_EXPIRATION_MINUTES: int = Field(
        default=30,
        description="JWT token expiration time in minutes"
    )

    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    DEFAULT_RATE_LIMIT: str = Field(
        default="100/hour",
        description="Default rate limit per user"
    )

    # CORS settings
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )

    # Authentication
    AUTH_ENABLED: bool = Field(
        default=False,
        description="Enable authentication"
    )

    # Logging and monitoring
    ENABLE_DETAILED_LOGGING: bool = Field(
        default=False,
        description="Enable detailed request/response logging"
    )

    # Security headers
    SECURITY_HEADERS_ENABLED: bool = Field(
        default=True,
        description="Enable security headers middleware"
    )

    # API Keys (if using API key authentication)
    API_KEYS: List[str] = Field(
        default=[],
        description="Valid API keys for authentication"
    )

    # Database security (if applicable)
    DATABASE_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Database encryption key"
    )

    # TLS/SSL settings
    USE_TLS: bool = Field(
        default=False,
        description="Force HTTPS/TLS"
    )

    TLS_CERT_FILE: Optional[str] = Field(
        default=None,
        description="Path to TLS certificate file"
    )

    TLS_KEY_FILE: Optional[str] = Field(
        default=None,
        description="Path to TLS private key file"
    )


def validate_production_config(settings: SecuritySettings) -> List[str]:
    """Validate production security configuration"""
    issues = []

    if settings.ENVIRONMENT == "production":
        # Critical security checks
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            issues.append(
                "CRITICAL: SECRET_KEY must be at least 32 characters in production")

        if not settings.USE_TLS:
            issues.append("WARNING: TLS should be enabled in production")

        if settings.CORS_ORIGINS == ["*"]:
            issues.append(
                "WARNING: CORS origins should be restricted in production")

        if not settings.RATE_LIMIT_ENABLED:
            issues.append(
                "WARNING: Rate limiting should be enabled in production")

        if settings.ENABLE_DETAILED_LOGGING:
            issues.append(
                "INFO: Detailed logging enabled - ensure log rotation is configured")

    return issues


def setup_security_middleware(app, settings: SecuritySettings):
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
                    allowed_hosts=["*"]  # Configure with actual hosts
                )

        except ImportError:
            logging.warning("Some security middleware not available")

    logging.info(
        f"Security middleware configured for {settings.ENVIRONMENT} environment")
