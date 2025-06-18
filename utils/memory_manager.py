# utils/memory_manager.py - Centralized Memory Management
import psutil
import logging
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import gc


@dataclass
class MemoryLimits:
    """Memory limits configuration"""
    total_mb: int
    cache_mb: int
    model_mb: int
    semantic_model_mb: int
    faiss_index_mb: int
    buffer_mb: int = 512  # Safety buffer


class MemoryManager:
    """Centralized memory management for all components"""

    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self.limits = self._calculate_limits()
        self.usage_tracking: Dict[str, int] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.alerts_enabled = True

        # Memory usage thresholds for alerts
        self.warning_threshold = 0.80  # 80%
        self.critical_threshold = 0.95  # 95%

        logging.info(
            f"Memory Manager initialized with {max_memory_mb}MB limit")
        logging.info(f"Memory allocation: {self._get_allocation_summary()}")

    def _calculate_limits(self) -> MemoryLimits:
        """Calculate memory limits based on total available memory"""

        # Get system memory info
        system_memory = psutil.virtual_memory()
        available_gb = system_memory.available / (1024**3)

        # Adjust max memory based on system availability
        if self.max_memory_mb > system_memory.available / (1024**2) * 0.9:
            adjusted_mb = int(system_memory.available / (1024**2) * 0.8)
            logging.warning(
                f"Reducing memory limit from {self.max_memory_mb}MB to {adjusted_mb}MB based on system availability")
            self.max_memory_mb = adjusted_mb

        # Calculate component allocations (percentages of total)
        total = self.max_memory_mb

        limits = MemoryLimits(
            total_mb=total,
            cache_mb=int(total * 0.15),      # 15% for caching
            model_mb=int(total * 0.60),      # 60% for models
            # 8% or 500MB max for semantic model
            semantic_model_mb=min(500, int(total * 0.08)),
            faiss_index_mb=int(total * 0.05),  # 5% for FAISS indices
            # 12% buffer for system and other operations
            buffer_mb=int(total * 0.12)
        )

        return limits

    def _get_allocation_summary(self) -> str:
        """Get human-readable allocation summary"""
        l = self.limits
        return (f"Total: {l.total_mb}MB, Models: {l.model_mb}MB, "
                f"Cache: {l.cache_mb}MB, Semantic: {l.semantic_model_mb}MB, "
                f"FAISS: {l.faiss_index_mb}MB, Buffer: {l.buffer_mb}MB")

    def get_limits(self) -> MemoryLimits:
        """Get current memory limits"""
        return self.limits

    def check_allocation(self, component: str, requested_mb: int) -> bool:
        """Check if a component can allocate the requested memory"""

        current_usage = self.get_current_usage()
        component_limit = self._get_component_limit(component)
        component_current = self.usage_tracking.get(component, 0)

        # Check component-specific limit
        if component_current + requested_mb > component_limit:
            logging.warning(f"Component {component} would exceed limit: "
                            f"{component_current + requested_mb}MB > {component_limit}MB")
            return False

        # Check total system limit
        total_new_usage = current_usage['total_used_mb'] + requested_mb
        if total_new_usage > self.limits.total_mb:
            logging.warning(f"Total memory would exceed limit: "
                            f"{total_new_usage}MB > {self.limits.total_mb}MB")
            return False

        return True

    def allocate(self, component: str, amount_mb: int) -> bool:
        """Allocate memory for a component"""

        if not self.check_allocation(component, amount_mb):
            return False

        self.usage_tracking[component] = self.usage_tracking.get(
            component, 0) + amount_mb
        logging.debug(
            f"Allocated {amount_mb}MB to {component}. Total: {self.usage_tracking[component]}MB")

        return True

    def deallocate(self, component: str, amount_mb: int = None):
        """Deallocate memory from a component"""

        if component not in self.usage_tracking:
            return

        if amount_mb is None:
            # Deallocate all
            amount_mb = self.usage_tracking[component]
            self.usage_tracking[component] = 0
        else:
            # Deallocate specific amount
            self.usage_tracking[component] = max(
                0, self.usage_tracking[component] - amount_mb)

        logging.debug(
            f"Deallocated {amount_mb}MB from {component}. Remaining: {self.usage_tracking[component]}MB")

    def _get_component_limit(self, component: str) -> int:
        """Get memory limit for a specific component"""

        component_limits = {
            'cache': self.limits.cache_mb,
            'models': self.limits.model_mb,
            'semantic_model': self.limits.semantic_model_mb,
            'faiss_index': self.limits.faiss_index_mb,
            'streaming': self.limits.cache_mb // 2,  # Half of cache limit
            'warmup': self.limits.cache_mb // 4,     # Quarter of cache limit
        }

        return component_limits.get(component, self.limits.buffer_mb)

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""

        # System memory
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()

        # Tracked component usage
        tracked_total = sum(self.usage_tracking.values())

        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_percent': system_memory.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2),
            'tracked_components': dict(self.usage_tracking),
            'tracked_total_mb': tracked_total,
            'total_used_mb': process_memory.rss / (1024**2),
            'limit_mb': self.limits.total_mb,
            'utilization_percent': (tracked_total / self.limits.total_mb) * 100,
            'available_mb': self.limits.total_mb - tracked_total
        }

    async def start_monitoring(self):
        """Start memory monitoring background task"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_memory())
            logging.info("Memory monitoring started")

    async def stop_monitoring(self):
        """Stop memory monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logging.info("Memory monitoring stopped")

    async def _monitor_memory(self):
        """Background memory monitoring loop"""

        while True:
            try:
                usage = self.get_current_usage()
                utilization = usage['utilization_percent'] / 100

                # Check thresholds
                if utilization > self.critical_threshold and self.alerts_enabled:
                    logging.error(f"CRITICAL: Memory usage at {utilization:.1%} "
                                  f"({usage['tracked_total_mb']}MB / {self.limits.total_mb}MB)")
                    await self._handle_critical_memory()

                elif utilization > self.warning_threshold and self.alerts_enabled:
                    logging.warning(f"WARNING: Memory usage at {utilization:.1%} "
                                    f"({usage['tracked_total_mb']}MB / {self.limits.total_mb}MB)")
                    await self._handle_warning_memory()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(60)

    async def _handle_warning_memory(self):
        """Handle warning-level memory usage"""

        # Try garbage collection
        gc.collect()

        # Log component usage for debugging
        usage = self.get_current_usage()
        logging.info("Memory usage by component:")
        for component, mb in usage['tracked_components'].items():
            percentage = (mb / self.limits.total_mb) * 100
            logging.info(f"  {component}: {mb}MB ({percentage:.1f}%)")

    async def _handle_critical_memory(self):
        """Handle critical memory usage"""

        logging.error("Attempting emergency memory cleanup...")

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Find largest memory user for potential cleanup
        if self.usage_tracking:
            largest_component = max(
                self.usage_tracking.items(), key=lambda x: x[1])
            logging.error(
                f"Largest memory user: {largest_component[0]} ({largest_component[1]}MB)")

        # Could trigger cleanup callbacks here if components register them

    def register_cleanup_callback(self, component: str, callback):
        """Register cleanup callback for emergency situations"""
        # This could be implemented to allow components to register cleanup functions
        pass

    def get_recommendations(self) -> list:
        """Get memory optimization recommendations"""

        recommendations = []
        usage = self.get_current_usage()

        # Check system memory pressure
        if usage['system_used_percent'] > 85:
            recommendations.append(
                "System memory usage is high. Consider reducing max_memory_mb setting.")

        # Check component usage
        for component, mb in usage['tracked_components'].items():
            limit = self._get_component_limit(component)
            if mb > limit * 0.8:  # Over 80% of component limit
                recommendations.append(
                    f"Component '{component}' using {mb}MB of {limit}MB limit. Consider optimization.")

        # Check total utilization
        if usage['utilization_percent'] > 80:
            recommendations.append(
                "Overall memory utilization is high. Consider enabling more aggressive cleanup.")

        return recommendations

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""

        usage = self.get_current_usage()

        return {
            'limits': {
                'total_mb': self.limits.total_mb,
                'cache_mb': self.limits.cache_mb,
                'model_mb': self.limits.model_mb,
                'semantic_model_mb': self.limits.semantic_model_mb,
                'faiss_index_mb': self.limits.faiss_index_mb,
                'buffer_mb': self.limits.buffer_mb
            },
            'usage': usage,
            'health': {
                'status': 'critical' if usage['utilization_percent'] > 95 else
                'warning' if usage['utilization_percent'] > 80 else 'healthy',
                'utilization_percent': usage['utilization_percent'],
                'available_mb': usage['available_mb']
            },
            'recommendations': self.get_recommendations(),
            'monitoring_active': self.monitoring_task is not None
        }


# Global memory manager instance
memory_manager = None


def get_memory_manager(max_memory_mb: int = None) -> MemoryManager:
    """Get or create global memory manager instance"""
    global memory_manager

    if memory_manager is None:
        if max_memory_mb is None:
            # Try to get from environment or use default
            import os
            max_memory_mb = int(os.getenv('MAX_MEMORY_MB', 8192))

        memory_manager = MemoryManager(max_memory_mb)

    return memory_manager


def check_memory_allocation(component: str, amount_mb: int) -> bool:
    """Convenience function to check memory allocation"""
    return get_memory_manager().check_allocation(component, amount_mb)


def allocate_memory(component: str, amount_mb: int) -> bool:
    """Convenience function to allocate memory"""
    return get_memory_manager().allocate(component, amount_mb)


def deallocate_memory(component: str, amount_mb: int = None):
    """Convenience function to deallocate memory"""
    get_memory_manager().deallocate(component, amount_mb)
