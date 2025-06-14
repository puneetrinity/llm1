# services/circuit_breaker.py - Complete Circuit Breaker Implementation

import time
import asyncio
import logging
from enum import Enum
from typing import Callable, Any, Coroutine, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import statistics

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    timeout_threshold: float = 30.0
    slow_request_threshold: float = 10.0
    max_requests_half_open: int = 5

class CircuitBreakerStats:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.timeouts = 0
        self.slow_requests = 0
        self.state_changes = []
        self.recent_response_times = []
        self.last_failure_time = None
        self.last_success_time = None

    def record_success(self, response_time: float):
        self.total_requests += 1
        self.successful_requests += 1
        self.last_success_time = datetime.now()
        self._add_response_time(response_time)

    def record_failure(self, error_type: str = "general"):
        self.total_requests += 1
        self.failed_requests += 1
        self.last_failure_time = datetime.now()
        if error_type == "timeout":
            self.timeouts += 1

    def record_slow_request(self, response_time: float):
        self.slow_requests += 1
        self._add_response_time(response_time)

    def _add_response_time(self, response_time: float):
        self.recent_response_times.append(response_time)
        if len(self.recent_response_times) > 100:
            self.recent_response_times.pop(0)

    def get_failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    def get_avg_response_time(self) -> float:
        if not self.recent_response_times:
            return 0.0
        return statistics.mean(self.recent_response_times)

class CircuitBreaker:
    def __init__(self, name: str = "default", config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_requests = 0
        self._lock = asyncio.Lock()
        logging.info(f"Circuit breaker '{name}' initialized in {self.state.value} state")

    async def call(self, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        async with self._lock:
            if not await self._should_allow_request():
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is {self.state.value}. "
                    f"Failure rate: {self.stats.get_failure_rate():.1f}%"
                )
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_requests += 1

        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_threshold
            )
            response_time = time.time() - start_time
            await self._on_success(response_time)
            return result
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            await self._on_failure("timeout")
            raise CircuitBreakerTimeoutError(
                f"Request timed out after {response_time:.1f}s "
                f"(threshold: {self.config.timeout_threshold}s)"
            )
        except Exception:
            await self._on_failure("general")
            raise

    async def _should_allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                await self._transition_to_half_open()
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return self._half_open_requests < self.config.max_requests_half_open
        return False

    async def _on_success(self, response_time: float):
        async with self._lock:
            self.stats.record_success(response_time)
            if response_time > self.config.slow_request_threshold:
                self.stats.record_slow_request(response_time)
                logging.warning(
                    f"Circuit breaker '{self.name}': Slow request "
                    f"({response_time:.1f}s > {self.config.slow_request_threshold}s)"
                )
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to_closed()
            elif self.state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, error_type: str = "general"):
        async with self._lock:
            self.stats.record_failure(error_type)
            self._failure_count += 1
            self._last_failure_time = time.time()
            logging.warning(
                f"Circuit breaker '{self.name}': Failure #{self._failure_count} "
                f"(type: {error_type}, rate: {self.stats.get_failure_rate():.1f}%)"
            )
            if self.state == CircuitState.CLOSED and self._failure_count >= self.config.failure_threshold:
                await self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                await self._transition_to_open()

    def _should_attempt_reset(self) -> bool:
        if not self._last_failure_time:
            return True
        return time.time() - self._last_failure_time >= self.config.recovery_timeout

    async def _transition_to_open(self):
        previous_state = self.state
        self.state = CircuitState.OPEN
        self._half_open_requests = 0
        self._success_count = 0
        self.stats.state_changes.append({
            'from': previous_state.value,
            'to': self.state.value,
            'timestamp': datetime.now(),
            'reason': f'Failure threshold reached ({self._failure_count}/{self.config.failure_threshold})'
        })
        logging.error(
            f"Circuit breaker '{self.name}' OPENED - "
            f"Failures: {self._failure_count}/{self.config.failure_threshold}, "
            f"Failure rate: {self.stats.get_failure_rate():.1f}%"
        )

    async def _transition_to_half_open(self):
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self._half_open_requests = 0
        self._success_count = 0
        self.stats.state_changes.append({
            'from': previous_state.value,
            'to': self.state.value,
            'timestamp': datetime.now(),
            'reason': f'Recovery timeout reached ({self.config.recovery_timeout}s)'
        })
        logging.info(
            f"Circuit breaker '{self.name}' HALF-OPEN - "
            f"Testing service recovery (max {self.config.max_requests_half_open} requests)"
        )

    async def _transition_to_closed(self):
        previous_state = self.state
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self.stats.state_changes.append({
            'from': previous_state.value,
            'to': self.state.value,
            'timestamp': datetime.now(),
            'reason': f'Service recovered ({self._success_count}/{self.config.success_threshold} successes)'
        })
        logging.info(
            f"Circuit breaker '{self.name}' CLOSED - Service recovered, normal operation resumed"
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'half_open_requests': self._half_open_requests,
            'last_failure_time': self._last_failure_time,
            'stats': {
                'total_requests': self.stats.total_requests,
                'successful_requests': self.stats.successful_requests,
                'failed_requests': self.stats.failed_requests,
                'timeouts': self.stats.timeouts,
                'slow_requests': self.stats.slow_requests,
                'failure_rate': self.stats.get_failure_rate(),
                'avg_response_time': self.stats.get_avg_response_time()
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout_threshold': self.config.timeout_threshold
            },
            'recent_state_changes': self.stats.state_changes[-5:]
        }

    def reset(self):
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        logging.info(f"Circuit breaker '{self.name}' manually reset to CLOSED state")

# Custom Exceptions
class CircuitBreakerError(Exception): pass
class CircuitBreakerOpenError(CircuitBreakerError): pass
class CircuitBreakerTimeoutError(CircuitBreakerError): pass

# Manager
class CircuitBreakerManager:
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()

    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        if name not in self.circuit_breakers:
            effective_config = config or self.default_config
            self.circuit_breakers[name] = CircuitBreaker(name, effective_config)
        return self.circuit_breakers[name]

    async def call_with_circuit_breaker(self, service_name: str, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        circuit_breaker = self.get_circuit_breaker(service_name)
        return await circuit_breaker.call(func, *args, **kwargs)

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        return {name: cb.get_status() for name, cb in self.circuit_breakers.items()}

    def get_health_summary(self) -> Dict[str, Any]:
        total = len(self.circuit_breakers)
        open_count = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN)
        half_open_count = sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN)
        closed_count = total - open_count - half_open_count
        overall = "healthy"
        if open_count > 0:
            overall = "degraded" if open_count < total else "unhealthy"
        elif half_open_count > 0:
            overall = "recovering"
        return {
            "overall_health": overall,
            "total_circuit_breakers": total,
            "open": open_count,
            "half_open": half_open_count,
            "closed": closed_count,
            "circuit_breakers": list(self.circuit_breakers.keys())
        }

    def reset_all(self):
        for cb in self.circuit_breakers.values():
            cb.reset()
        logging.info("All circuit breakers reset")

# Global Manager Instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager

def get_circuit_breaker(service_name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    return get_circuit_breaker_manager().get_circuit_breaker(service_name, config)

async def call_with_circuit_breaker(service_name: str, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
    return await get_circuit_breaker_manager().call_with_circuit_breaker(service_name, func, *args, **kwargs)

def circuit_breaker(service_name: str, config: CircuitBreakerConfig = None):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await call_with_circuit_breaker(service_name, func, *args, **kwargs)
        return wrapper
    return decorator
