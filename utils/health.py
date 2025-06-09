# utils/health.py - Health Check Service
import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from models.responses import ServiceStatus

class HealthChecker:
    def __init__(self):
        self.start_time = datetime.now()
        self.health_checks = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.last_check_time = datetime.now()
        
        # Health thresholds
        self.thresholds = {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "disk_percent": 95.0,
            "response_time": 30.0  # seconds
        }
    
    async def start_monitoring(self):
        """Start health monitoring background task"""
        if self.monitoring_task is None:
            self.monitoring_task = asyncio.create_task(self._monitor_health())
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
    
    async def _monitor_health(self):
        """Background health monitoring loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in health monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_checks(self):
        """Perform all health checks"""
        self.last_check_time = datetime.now()
        
        # System resource checks
        await self._check_system_resources()
        
        # Service-specific checks would go here
        # These will be populated by individual services
    
    async def _check_system_resources(self):
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_healthy = cpu_percent < self.thresholds["cpu_percent"]
            
            self.health_checks["cpu"] = ServiceStatus(
                name="cpu",
                status="healthy" if cpu_healthy else "unhealthy",
                last_check=self.last_check_time,
                details={
                    "usage_percent": cpu_percent,
                    "threshold": self.thresholds["cpu_percent"]
                }
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_healthy = memory.percent < self.thresholds["memory_percent"]
            
            self.health_checks["memory"] = ServiceStatus(
                name="memory",
                status="healthy" if memory_healthy else "unhealthy",
                last_check=self.last_check_time,
                details={
                    "usage_percent": memory.percent,
                    "available_gb": memory.available / (1024**3),
                    "threshold": self.thresholds["memory_percent"]
                }
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_healthy = disk.percent < self.thresholds["disk_percent"]
            
            self.health_checks["disk"] = ServiceStatus(
                name="disk",
                status="healthy" if disk_healthy else "unhealthy",
                last_check=self.last_check_time,
                details={
                    "usage_percent": disk.percent,
                    "free_gb": disk.free / (1024**3),
                    "threshold": self.thresholds["disk_percent"]
                }
            )
            
        except Exception as e:
            logging.error(f"Error checking system resources: {str(e)}")
            self.health_checks["system"] = ServiceStatus(
                name="system",
                status="unknown",
                last_check=self.last_check_time,
                details={"error": str(e)}
            )
    
    def register_service_check(self, service_name: str, check_function):
        """Register a custom health check for a service"""
        # This would be used by individual services to register their health checks
        pass
    
    async def check_service_health(self, service_name: str, check_function) -> ServiceStatus:
        """Check health of a specific service"""
        try:
            start_time = datetime.now()
            is_healthy = await check_function()
            response_time = (datetime.now() - start_time).total_seconds()
            
            status = "healthy" if is_healthy else "unhealthy"
            if response_time > self.thresholds["response_time"]:
                status = "slow"
            
            return ServiceStatus(
                name=service_name,
                status=status,
                last_check=datetime.now(),
                details={
                    "response_time": response_time,
                    "healthy": is_healthy
                }
            )
        except Exception as e:
            logging.error(f"Health check failed for {service_name}: {str(e)}")
            return ServiceStatus(
                name=service_name,
                status="unhealthy",
                last_check=datetime.now(),
                details={"error": str(e)}
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        
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
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
            }
        except Exception:
            system_metrics = {"error": "Could not collect system metrics"}
        
        return {
            "healthy": overall_healthy,
            "timestamp": datetime.now(),
            "version": "2.0.0",
            "uptime_seconds": uptime.total_seconds(),
            "services": list(self.health_checks.values()),
            "system": system_metrics,
            "unhealthy_services": unhealthy_services,
            "last_check": self.last_check_time
        }
    
    def add_external_service_check(self, service_name: str, status: ServiceStatus):
        """Add health status from external service"""
        self.health_checks[service_name] = status
    
    def remove_service_check(self, service_name: str):
        """Remove a service health check"""
        if service_name in self.health_checks:
            del self.health_checks[service_name]
