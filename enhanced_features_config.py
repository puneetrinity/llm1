# enhanced_features_config.py - Standalone Enhanced Features Configuration
# Add this file to your root directory alongside config.py and config_enhanced.py

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import logging


class EnhancedFeaturesConfig(BaseModel):
    """Standalone configuration for enhanced features - no dependencies on existing config"""

    class Config:
        extra = "ignore"  # Ignore unknown env vars
        case_sensitive = False

    # Connection Pooling (Safe - no external dependencies)
    connection_pooling_enabled: bool = Field(
        default=True,
        description="Enable optimized connection pooling"
    )
    connection_pool_size: int = Field(
        default=100,
        ge=10, le=500,
        description="Total connection pool size"
    )
    connection_per_host: int = Field(
        default=20,
        ge=5, le=100,
        description="Connections per host"
    )

    # Circuit Breaker (Safe - no external dependencies)
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker protection"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        ge=1, le=20,
        description="Failures before opening circuit"
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60,
        ge=10, le=300,
        description="Recovery timeout in seconds"
    )

    # Smart Caching (Safe fallbacks)
    smart_cache_enabled: bool = Field(
        default=True,
        description="Enable intelligent caching"
    )
    redis_enabled: bool = Field(
        default=True,
        description="Use Redis if available (falls back to memory)"
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    semantic_similarity_enabled: bool = Field(
        default=True,
        description="Enable semantic similarity (falls back to exact match)"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.5, le=0.99,
        description="Semantic similarity threshold"
    )

    # Memory Management
    max_memory_mb: int = Field(
        default=8192,
        ge=1024, le=32768,
        description="Maximum memory usage in MB"
    )
    cache_memory_percent: float = Field(
        default=15.0,
        ge=5.0, le=30.0,
        description="Percentage of memory for caching"
    )

    # Global Settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for enhanced features"
    )

    @classmethod
    def from_env(cls) -> 'EnhancedFeaturesConfig':
        """Load configuration from environment variables"""
        return cls(
            # Connection Pooling
            connection_pooling_enabled=_get_bool_env(
                'ENHANCED_CONNECTION_POOLING_ENABLED', True),
            connection_pool_size=_get_int_env(
                'ENHANCED_CONNECTION_POOLING_TOTAL_LIMIT', 100),
            connection_per_host=_get_int_env(
                'ENHANCED_CONNECTION_POOLING_PER_HOST_LIMIT', 20),

            # Circuit Breaker
            circuit_breaker_enabled=_get_bool_env(
                'ENHANCED_CIRCUIT_BREAKER_ENABLED', True),
            circuit_breaker_failure_threshold=_get_int_env(
                'ENHANCED_CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5),
            circuit_breaker_recovery_timeout=_get_int_env(
                'ENHANCED_CIRCUIT_BREAKER_RECOVERY_TIMEOUT', 60),

            # Smart Caching
            smart_cache_enabled=_get_bool_env(
                'ENHANCED_SMART_CACHE_ENABLED', True),
            redis_enabled=_get_bool_env(
                'ENHANCED_SMART_CACHE_REDIS_ENABLED', True),
            redis_url=os.getenv(
                'ENHANCED_SMART_CACHE_REDIS_URL', 'redis://localhost:6379'),
            semantic_similarity_enabled=_get_bool_env(
                'ENHANCED_SMART_CACHE_SEMANTIC_ENABLED', True),
            similarity_threshold=_get_float_env(
                'ENHANCED_SMART_CACHE_SIMILARITY_THRESHOLD', 0.85),

            # Memory Management
            max_memory_mb=_get_int_env(
                'ENHANCED_MEMORY_MANAGEMENT_MAX_MB', 8192),
            cache_memory_percent=_get_float_env(
                'ENHANCED_MEMORY_CACHE_ALLOCATION_PERCENT', 15.0),

            # Global
            debug_mode=_get_bool_env('ENHANCED_DEBUG_MODE', False)
        )

    def get_feature_summary(self) -> Dict[str, bool]:
        """Get summary of enabled features"""
        return {
            'connection_pooling': self.connection_pooling_enabled,
            'circuit_breaker': self.circuit_breaker_enabled,
            'smart_cache': self.smart_cache_enabled,
            'redis_backend': self.redis_enabled,
            'semantic_similarity': self.semantic_similarity_enabled,
            'debug_mode': self.debug_mode
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return warnings/errors"""
        warnings = []
        errors = []

        # Memory validation
        cache_memory_mb = (self.max_memory_mb *
                           self.cache_memory_percent / 100)
        if cache_memory_mb < 100:
            warnings.append(
                f"Cache memory allocation is very low: {cache_memory_mb:.0f}MB")

        # Connection pool validation
        if self.connection_per_host > self.connection_pool_size:
            errors.append(
                "connection_per_host cannot exceed connection_pool_size")

        # Semantic similarity validation
        if self.semantic_similarity_enabled and cache_memory_mb < 300:
            warnings.append(
                "Semantic similarity may need at least 300MB cache memory")

        return {
            'valid': len(errors) == 0,
            'warnings': warnings,
            'errors': errors,
            'estimated_cache_memory_mb': cache_memory_mb
        }

# Helper functions for environment variable parsing


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean from environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logging.warning(
            f"Invalid integer value for {key}, using default: {default}")
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logging.warning(
            f"Invalid float value for {key}, using default: {default}")
        return default


# Singleton instance
_enhanced_config: Optional[EnhancedFeaturesConfig] = None


def get_enhanced_config() -> EnhancedFeaturesConfig:
    """Get the enhanced features configuration singleton"""
    global _enhanced_config
    if _enhanced_config is None:
        _enhanced_config = EnhancedFeaturesConfig.from_env()
    return _enhanced_config


def validate_enhanced_config() -> bool:
    """Validate enhanced configuration and log results"""
    config = get_enhanced_config()
    validation = config.validate_configuration()

    if validation['errors']:
        for error in validation['errors']:
            logging.error(f"❌ Enhanced config error: {error}")
        return False

    if validation['warnings']:
        for warning in validation['warnings']:
            logging.warning(f"⚠️ Enhanced config warning: {warning}")

    features = config.get_feature_summary()
    enabled_features = [name for name, enabled in features.items() if enabled]

    logging.info(
        f"✅ Enhanced features configured: {', '.join(enabled_features)}")
    logging.info(
        f"💾 Estimated cache memory: {validation['estimated_cache_memory_mb']:.0f}MB")

    return True


# Quick test function
if __name__ == "__main__":
    # Test the configuration
    config = get_enhanced_config()
    print("Enhanced Features Configuration:")
    print("=" * 40)

    features = config.get_feature_summary()
    for feature, enabled in features.items():
        status = "✅ Enabled" if enabled else "⏸️ Disabled"
        print(f"{feature}: {status}")

    print("\nValidation:")
    validation = config.validate_configuration()
    if validation['valid']:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors:")
        for error in validation['errors']:
            print(f"  - {error}")

    if validation['warnings']:
        print("⚠️ Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
