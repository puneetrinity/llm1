# utils/__init__.py
from .metrics import MetricsCollector
from .health import HealthChecker

# Optional enhanced utilities
try:
    from .performance_monitor import PerformanceMonitor
    from .dashboard import EnhancedDashboard
    from .websocket_dashboard import WebSocketDashboard
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    ENHANCED_UTILS_AVAILABLE = False

__all__ = [
    "MetricsCollector",
    "HealthChecker",
    "ENHANCED_UTILS_AVAILABLE"
]

if ENHANCED_UTILS_AVAILABLE:
    __all__.extend([
        "PerformanceMonitor",
        "EnhancedDashboard", 
        "WebSocketDashboard"
    ])
