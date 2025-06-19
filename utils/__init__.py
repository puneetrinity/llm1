# utils/__init__.py - Safe utility imports
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

    def get_memory_manager():
        return None

    MemoryManager = None
    MEMORY_MANAGER_AVAILABLE = False

__all__ = ["MetricsCollector", "HealthChecker", "MEMORY_MANAGER_AVAILABLE"]

if MEMORY_MANAGER_AVAILABLE:
    __all__.extend(["get_memory_manager", "MemoryManager"])
