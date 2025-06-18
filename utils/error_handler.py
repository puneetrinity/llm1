# utils/error_handler.py - Enhanced Error Handling with Circuit Breaker Integration
import logging
import traceback
import asyncio
import time
from typing import Dict, Any, Optional, Callable, Type, Union
from datetime import datetime
from functools import wraps
from enum import Enum
import json

# Import circuit breaker components
from services.circuit_breaker import (
    get_circuit_breaker_manager,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerTimeoutError
)


class ErrorSeverity(Enum):
    """Error severity levels with circuit breaker integration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification and routing"""
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    CONFIGURATION = "configuration"
    CIRCUIT_BREAKER = "circuit_breaker"  # New category


class LLMProxyError(Exception):
    """Enhanced base exception with circuit breaker awareness"""

    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Dict[str, Any] = None,
        user_message: str = None,
        retry_after: Optional[int] = None,
        circuit_breaker_name: str = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or "An error occurred while processing your request"
        self.retry_after = retry_after
        self.circuit_breaker_name = circuit_breaker_name
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses"""
        return {
            "error_code": self.error_code,
            "message": self.user_message,
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "retry_after": self.retry_after,
            "circuit_breaker": self.circuit_breaker_name
        }

# Enhanced specific exceptions with circuit breaker integration


class OllamaConnectionError(LLMProxyError):
    """Ollama service connection error with circuit breaker awareness"""

    def __init__(self, message: str, ollama_url: str = None, circuit_breaker_triggered: bool = False):
        super().__init__(
            message=message,
            error_code="OLLAMA_CONNECTION_ERROR",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details={"ollama_url": ollama_url,
                     "circuit_breaker_triggered": circuit_breaker_triggered},
            user_message="AI service is temporarily unavailable. Please try again later.",
            retry_after=60 if circuit_breaker_triggered else 30
        )


class CircuitBreakerTriggeredError(LLMProxyError):
    """Circuit breaker has been triggered"""

    def __init__(self, service_name: str, failure_rate: float, retry_after: int = 60):
        super().__init__(
            message=f"Circuit breaker for {service_name} is open (failure rate: {failure_rate:.1f}%)",
            error_code="CIRCUIT_BREAKER_OPEN",
            category=ErrorCategory.CIRCUIT_BREAKER,
            severity=ErrorSeverity.HIGH,
            details={"service_name": service_name,
                     "failure_rate": failure_rate},
            user_message=f"Service temporarily unavailable due to high error rate. Please try again in {retry_after} seconds.",
            retry_after=retry_after,
            circuit_breaker_name=service_name
        )


class ServiceTimeoutError(LLMProxyError):
    """Service timeout with circuit breaker tracking"""

    def __init__(self, service_name: str, timeout_duration: float):
        super().__init__(
            message=f"Service {service_name} timed out after {timeout_duration:.1f}s",
            error_code="SERVICE_TIMEOUT",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            details={"service_name": service_name,
                     "timeout_duration": timeout_duration},
            user_message="Request timed out. Please try again.",
            retry_after=30
        )


class EnhancedErrorHandler:
    """Enhanced error handling with circuit breaker integration and analytics"""

    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_callbacks: Dict[str, Callable] = {}
        self.circuit_breaker_manager = get_circuit_breaker_manager()

        # Error analytics
        self.error_history = []
        self.service_health = {}

        # Setup default circuit breaker configurations
        self._setup_default_circuit_breakers()

    def _setup_default_circuit_breakers(self):
        """Setup default circuit breakers for common services"""
        configs = {
            'ollama': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60,
                timeout_threshold=30.0,
                slow_request_threshold=10.0
            ),
            'redis': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                timeout_threshold=5.0
            ),
            'semantic_model': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120,
                timeout_threshold=15.0
            )
        }

        for service_name, config in configs.items():
            self.circuit_breaker_manager.get_circuit_breaker(
                service_name, config)

    async def call_with_protection(
        self,
        service_name: str,
        func: Callable,
        *args,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection and enhanced error handling"""

        context = context or {}
        start_time = time.time()

        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker_manager.call_with_circuit_breaker(
                service_name, func, *args, **kwargs
            )

            # Update service health on success
            response_time = time.time() - start_time
            self._update_service_health(service_name, True, response_time)

            return result

        except CircuitBreakerOpenError as e:
            # Handle circuit breaker open state
            circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(
                service_name)
            status = circuit_breaker.get_status()

            error = CircuitBreakerTriggeredError(
                service_name=service_name,
                failure_rate=status['stats']['failure_rate'],
                retry_after=circuit_breaker.config.recovery_timeout
            )

            await self._handle_error(error, context, service_name)
            raise error

        except CircuitBreakerTimeoutError as e:
            # Handle timeout through circuit breaker
            response_time = time.time() - start_time
            error = ServiceTimeoutError(service_name, response_time)

            self._update_service_health(service_name, False, response_time)
            await self._handle_error(error, context, service_name)
            raise error

        except Exception as e:
            # Handle other exceptions
            response_time = time.time() - start_time
            std_error = await self._convert_to_standard_error(e, context, service_name)

            self._update_service_health(service_name, False, response_time)
            await self._handle_error(std_error, context, service_name)
            raise std_error

    async def _handle_error(
        self,
        error: LLMProxyError,
        context: Dict[str, Any] = None,
        service_name: str = None
    ):
        """Enhanced error handling with circuit breaker analytics"""

        # Track error
        self._track_error(error, service_name)

        # Log error with enhanced context
        self._log_error(error, context, service_name)

        # Execute callbacks
        if error.error_code in self.error_callbacks:
            try:
                callback = self.error_callbacks[error.error_code]
                if asyncio.iscoroutinefunction(callback):
                    await callback(error, context)
                else:
                    callback(error, context)
            except Exception as callback_error:
                logging.error(f"Error callback failed: {callback_error}")

    async def _convert_to_standard_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        service_name: str = None
    ) -> LLMProxyError:
        """Convert any exception to standardized LLMProxyError with service context"""

        error_type = type(error).__name__
        error_message = str(error)

        # Enhanced error mapping with service awareness
        if "connection" in error_message.lower() or "timeout" in error_message.lower():
            if service_name == "ollama":
                return OllamaConnectionError(
                    message=error_message,
                    ollama_url=context.get("ollama_url"),
                    circuit_breaker_triggered=False
                )
            else:
                return LLMProxyError(
                    message=error_message,
                    error_code="CONNECTION_ERROR",
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.HIGH,
                    details={"service": service_name,
                             "original_type": error_type}
                )

        elif "memory" in error_message.lower() or error_type == "MemoryError":
            return LLMProxyError(
                message=error_message,
                error_code="MEMORY_ERROR",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                details={"service": service_name, "original_type": error_type}
            )

        elif "redis" in error_message.lower() and service_name == "redis":
            return LLMProxyError(
                message=error_message,
                error_code="CACHE_ERROR",
                category=ErrorCategory.EXTERNAL_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                details={"service": service_name, "cache_backend": "redis"}
            )

        else:
            # Generic internal error
            return LLMProxyError(
                message=error_message,
                error_code="INTERNAL_ERROR",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                details={
                    "service": service_name,
                    "original_type": error_type,
                    "traceback": traceback.format_exc()
                }
            )

    def _track_error(self, error: LLMProxyError, service_name: str = None):
        """Enhanced error tracking with service analytics"""
        self.error_counts[error.error_code] = self.error_counts.get(
            error.error_code, 0) + 1

        # Track error history for analytics
        self.error_history.append({
            'timestamp': error.timestamp,
            'error_code': error.error_code,
            'category': error.category.value,
            'severity': error.severity.value,
            'service': service_name,
            'circuit_breaker': error.circuit_breaker_name
        })

        # Limit history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]

    def _update_service_health(self, service_name: str, success: bool, response_time: float):
        """Update service health metrics"""
        if service_name not in self.service_health:
            self.service_health[service_name] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0,
                'last_success': None,
                'last_failure': None
            }

        health = self.service_health[service_name]
        health['total_requests'] += 1

        if success:
            health['successful_requests'] += 1
            health['last_success'] = datetime.now()
        else:
            health['failed_requests'] += 1
            health['last_failure'] = datetime.now()

        # Update rolling average response time
        current_avg = health['avg_response_time']
        total_requests = health['total_requests']
        health['avg_response_time'] = (
            (current_avg * (total_requests - 1)) + response_time) / total_requests

    def _log_error(self, error: LLMProxyError, context: Dict[str, Any], service_name: str = None):
        """Enhanced error logging with service context"""

        log_data = {
            "error_code": error.error_code,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "service": service_name,
            "circuit_breaker": error.circuit_breaker_name,
            "context": context,
            "timestamp": error.timestamp.isoformat()
        }

        log_message = f"[{service_name or 'UNKNOWN'}] {error.error_code}: {error.message}"

        if error.severity == ErrorSeverity.CRITICAL:
            logging.critical(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logging.error(log_message, extra=log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logging.warning(log_message, extra=log_data)
        else:
            logging.info(log_message, extra=log_data)

    def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics with circuit breaker status"""
        total_errors = sum(self.error_counts.values())

        # Circuit breaker status
        circuit_breaker_status = self.circuit_breaker_manager.get_health_summary()

        # Service health summary
        service_summary = {}
        for service, health in self.service_health.items():
            failure_rate = 0
            if health['total_requests'] > 0:
                failure_rate = (health['failed_requests'] /
                                health['total_requests']) * 100

            service_summary[service] = {
                'health_status': 'healthy' if failure_rate < 5 else 'degraded' if failure_rate < 20 else 'unhealthy',
                'failure_rate': failure_rate,
                'avg_response_time': health['avg_response_time'],
                'total_requests': health['total_requests']
            }

        return {
            "total_errors": total_errors,
            "error_counts": dict(self.error_counts),
            "top_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "circuit_breakers": circuit_breaker_status,
            "service_health": service_summary,
            "recent_errors": self.error_history[-10:] if self.error_history else []
        }

    def register_callback(self, error_code: str, callback: Callable):
        """Register error callback"""
        self.error_callbacks[error_code] = callback

    def get_service_recommendations(self) -> List[str]:
        """Get service optimization recommendations based on error patterns"""
        recommendations = []

        # Analyze circuit breaker status
        cb_status = self.circuit_breaker_manager.get_health_summary()
        if cb_status['open'] > 0:
            recommendations.append(
                f"{cb_status['open']} service(s) have circuit breakers open - check service health")

        # Analyze service health
        for service, health in self.service_health.items():
            failure_rate = (health['failed_requests'] /
                            max(1, health['total_requests'])) * 100

            if failure_rate > 20:
                recommendations.append(
                    f"Service '{service}' has high failure rate ({failure_rate:.1f}%) - investigate")
            elif health['avg_response_time'] > 10:
                recommendations.append(
                    f"Service '{service}' has slow response times ({health['avg_response_time']:.1f}s)")

        return recommendations

# Enhanced decorator with circuit breaker integration


def handle_errors_with_circuit_breaker(
    service_name: str,
    context_func: Callable = None,
    reraise: bool = True
):
    """Enhanced error handling decorator with circuit breaker protection"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            context = {}
            if context_func:
                try:
                    context = context_func(*args, **kwargs)
                except Exception:
                    pass

            return await error_handler.call_with_protection(
                service_name, func, *args, context=context, **kwargs
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async wrapper
            async def async_func():
                return func(*args, **kwargs)

            context = {}
            if context_func:
                try:
                    context = context_func(*args, **kwargs)
                except Exception:
                    pass

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                error_handler.call_with_protection(
                    service_name, async_func, context=context
                )
            )

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Global enhanced error handler instance
error_handler = EnhancedErrorHandler()

# Convenience functions


async def handle_error_with_circuit_breaker(
    service_name: str,
    func: Callable,
    *args,
    context: Dict[str, Any] = None,
    **kwargs
) -> Any:
    """Convenience function for handling errors with circuit breaker protection"""
    return await error_handler.call_with_protection(service_name, func, *args, context=context, **kwargs)


def register_error_callback(error_code: str, callback: Callable):
    """Register error callback"""
    error_handler.register_callback(error_code, callback)

# Context functions for common scenarios


def ollama_context(*args, **kwargs) -> Dict[str, Any]:
    """Generate context for Ollama-related functions"""
    return {
        "component": "ollama",
        "ollama_url": getattr(args[0], 'base_url', None) if args else None
    }


def cache_context(*args, **kwargs) -> Dict[str, Any]:
    """Generate context for cache-related functions"""
    return {
        "component": "cache",
        "cache_key": kwargs.get("key") or (args[1] if len(args) > 1 else None)
    }
