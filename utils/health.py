# utils/health.py - Complete Health Check Service with Perfect Indentation
import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Try to import ServiceStatus, create fallback if not available
try:
    from models.responses import ServiceStatus
except ImportError:
    @dataclass
    class ServiceStatus:
        name: str
        status: str
        last_check: str
        details: Optional[Dict[str, Any]] = None

class HealthChecker:
    def __init__(self):
        self.start_time = datetime.now()
        self.health_checks: Dict[str, ServiceStatus] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_check_time = datetime.now()
        
        # Health thresholds
        self.thresholds = {
            "response_time_p95": 10.0,  # seconds
            "cpu_usage": 80.0,          # percentage
            "memory_usage": 85.0,       # percentage
            "disk_usage": 90.0,         # percentage
            "error_rate": 5.0           # percentage
        }
        
        self.alerts = []
        self.stats = {
            'total_checks': 0,
            'failed_checks': 0,
            'alerts_triggered': 0
        }
        
    async def start_monitoring(self):
        """Start health monitoring background task"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self.monitor_system())
            logging.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            self.monitoring_task = None
            logging.info("Health monitoring stopped")
    
    async def monitor_system(self):
        """Background health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logging.info("Health monitoring cancelled")
                break
            except Exception as e:
                logging.error(f"Error in health monitoring: {str(e)}")
                await asyncio.sleep(30)  # Wait shorter on error
    
    async def _perform_health_checks(self):
        """Perform all health checks"""
        self.last_check_time = datetime.now()
        self.stats['total_checks'] += 1
        
        try:
            # System resource checks
            await self._check_system_resources()
            
            # Service-specific checks
            await self._check_services()
            
        except Exception as e:
            logging.error(f"Error performing health checks: {str(e)}")
            self.stats['failed_checks'] += 1
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_healthy = cpu_percent < self.thresholds["cpu_usage"]
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < self.thresholds["memory_usage"]
            
            # Disk usage
            try:
                disk = psutil.disk_usage('/')
                disk_healthy = disk.percent < self.thresholds["disk_usage"]
            except Exception:
                disk_healthy = True
                disk = None
            
            # Update health checks with proper string conversion
            self.health_checks["cpu"] = ServiceStatus(
                name="cpu",
                status="healthy" if cpu_healthy else "unhealthy",
                last_check=datetime.now().isoformat(),
                details={
                    "usage_percent": cpu_percent,
                    "threshold": self.thresholds["cpu_usage"],
                    "healthy": cpu_healthy
                }
            )
            
            self.health_checks["memory"] = ServiceStatus(
                name="memory",
                status="healthy" if memory_healthy else "unhealthy",
                last_check=datetime.now().isoformat(),
                details={
                    "usage_percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                    "threshold": self.thresholds["memory_usage"],
                    "healthy": memory_healthy
                }
            )
            
            if disk:
                self.health_checks["disk"] = ServiceStatus(
                    name="disk",
                    status="healthy" if disk_healthy else "unhealthy",
                    last_check=datetime.now().isoformat(),
                    details={
                        "usage_percent": disk.percent,
                        "free_gb": round(disk.free / (1024**3), 2),
                        "total_gb": round(disk.total / (1024**3), 2),
                        "threshold": self.thresholds["disk_usage"],
                        "healthy": disk_healthy
                    }
                )
            
            # Check for alerts
            if not cpu_healthy:
                self._trigger_alert("cpu", f"High CPU usage: {cpu_percent}%")
            if not memory_healthy:
                self._trigger_alert("memory", f"High memory usage: {memory.percent}%")
            if disk and not disk_healthy:
                self._trigger_alert("disk", f"High disk usage: {disk.percent}%")
                
        except Exception as e:
            logging.error(f"Error checking system resources: {str(e)}")
            self.health_checks["system"] = ServiceStatus(
                name="system",
                status="unknown",
                last_check=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    async def _check_services(self):
        """Check application services"""
        try:
            # Basic service check - this can be extended
            service_healthy = True
            
            self.health_checks["application"] = ServiceStatus(
                name="application",
                status="healthy" if service_healthy else "unhealthy",
                last_check=datetime.now().isoformat(),
                details={
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                    "healthy": service_healthy
                }
            )
            
        except Exception as e:
            logging.error(f"Error checking services: {str(e)}")
    
    def _trigger_alert(self, component: str, message: str):
        """Trigger an alert"""
        alert = {
            "component": component,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "severity": "warning"
        }
        
        self.alerts.append(alert)
        self.stats['alerts_triggered'] += 1
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
        
        logging.warning(f"Health alert: {component} - {message}")
    
    async def check_service_health(self, service_name: str, check_function) -> ServiceStatus:
        """Check health of a specific service"""
        try:
            start_time = datetime.now()
            is_healthy = await check_function()
            response_time = (datetime.now() - start_time).total_seconds()
            
            status = "healthy" if is_healthy else "unhealthy"
            if response_time > self.thresholds.get("response_time_p95", 10.0):
                status = "slow"
            
            return ServiceStatus(
                name=service_name,
                status=status,
                last_check=datetime.now().isoformat(),
                details={
                    "response_time": response_time,
                    "healthy": is_healthy,
                    "threshold": self.thresholds.get("response_time_p95", 10.0)
                }
            )
            
        except Exception as e:
            logging.error(f"Health check failed for {service_name}: {str(e)}")
            return ServiceStatus(
                name=service_name,
                status="unhealthy",
                last_check=datetime.now().isoformat(),
                details={"error": str(e)}
            )
    
    def add_external_service_check(self, service_name: str, status: ServiceStatus):
        """Add health status from external service"""
        self.health_checks[service_name] = status
    
    def remove_service_check(self, service_name: str):
        """Remove a service health check"""
        if service_name in self.health_checks:
            del self.health_checks[service_name]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            # Ensure we have recent health checks
            if datetime.now() - self.last_check_time > timedelta(minutes=2):
                await self._perform_health_checks()
            
            # Determine overall health
            overall_healthy = True
            unhealthy_services = []
            
            for service_name, status in self.health_checks.items():
                if status.status in ["unhealthy", "unknown"]:
                    overall_healthy = False
                    unhealthy_services.append(service_name)
            
            # Calculate uptime
            uptime = datetime.now() - self.start_time
            
            # Get system metrics
            try:
                system_metrics = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                }
            except Exception:
                system_metrics = {"error": "Could not collect system metrics"}
            
            return {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "uptime_seconds": uptime.total_seconds(),
                "services": [
                    {
                        "name": status.name,
                        "status": status.status,
                        "last_check": status.last_check,
                        "details": status.details
                    }
                    for status in self.health_checks.values()
                ],
                "system": system_metrics,
                "unhealthy_services": unhealthy_services,
                "last_check": self.last_check_time.isoformat(),
                "stats": self.stats,
                "recent_alerts": self.alerts[-5:] if self.alerts else []
            }
            
        except Exception as e:
            logging.error(f"Error getting health status: {str(e)}")
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "version": "2.0.0",
                "error": str(e),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a brief health summary"""
        try:
            healthy_count = sum(1 for s in self.health_checks.values() if s.status == "healthy")
            total_count = len(self.health_checks)
            
            return {
                "overall_status": "healthy" if healthy_count == total_count else "degraded",
                "healthy_services": healthy_count,
                "total_services": total_count,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
                "recent_alerts": len(self.alerts),
                "monitoring_active": self.monitoring_task is not None
            }
            
        except Exception as e:
            return {
                "overall_status": "unknown",
                "error": str(e)
            }
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current health thresholds"""
        return self.thresholds.copy()
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update health thresholds"""
        self.thresholds.update(new_thresholds)
        logging.info(f"Health thresholds updated: {new_thresholds}")
    
    def get_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:] if self.alerts else []
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        logging.info("Health alerts cleared")
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information"""
        try:
            cpu_info = {
                "physical_cores": psutil.cpu_count(logical=False),
                "total_cores": psutil.cpu_count(logical=True),
                "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
                "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "cpu_usage": psutil.cpu_percent(interval=1)
            }
            
            memory_info = psutil.virtual_memory()
            memory = {
                "total": round(memory_info.total / (1024**3), 2),
                "available": round(memory_info.available / (1024**3), 2),
                "used": round(memory_info.used / (1024**3), 2),
                "percentage": memory_info.percent
            }
            
            disk_info = psutil.disk_usage('/')
            disk = {
                "total": round(disk_info.total / (1024**3), 2),
                "used": round(disk_info.used / (1024**3), 2),
                "free": round(disk_info.free / (1024**3), 2),
                "percentage": round((disk_info.used / disk_info.total) * 100, 2)
            }
            
            return {
                "cpu": cpu_info,
                "memory": memory,
                "disk": disk,
                "uptime": (datetime.now() - self.start_time).total_seconds()
            }
            
        except Exception as e:
            logging.error(f"Error getting system info: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.stop_monitoring()
        self.health_checks.clear()
        self.alerts.clear()
        logging.info("Health checker cleaned up")
