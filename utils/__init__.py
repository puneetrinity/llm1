# utils/__init__.py
from .metrics import MetricsCollector
from .health import HealthChecker

# Optional enhanced utilities - import safely
ENHANCED_UTILS_AVAILABLE = False

try:
    from .performance_monitor import PerformanceMonitor
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    PerformanceMonitor = None

try:
    from .dashboard import EnhancedDashboard
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    EnhancedDashboard = None

try:
    from .websocket_dashboard import WebSocketDashboard
    ENHANCED_UTILS_AVAILABLE = True
except ImportError:
    WebSocketDashboard = None

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
