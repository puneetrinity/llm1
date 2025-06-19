# Performance monitoring and optimization
# utils/performance_monitor.py - Advanced Performance Monitoring
import time
import asyncio
import psutil
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json


class PerformanceMonitor:
    def __init__(self):
        self.request_timings = deque(maxlen=1000)
        self.model_performance = defaultdict(
            lambda: {
                "response_times": deque(maxlen=100),
                "throughput": deque(maxlen=100),
                "error_count": 0,
                "total_requests": 0,
            }
        )

        self.system_metrics = deque(maxlen=288)  # 24 hours at 5-min intervals
        self.monitoring_task = None

        # Performance thresholds
        self.thresholds = {
            "response_time_p95": 10.0,  # seconds
            "cpu_usage": 80.0,  # percentage
            "memory_usage": 85.0,  # percentage
            "gpu_memory": 90.0,  # percentage
            "error_rate": 5.0,  # percentage
        }

        self.alerts = []

    async def start_monitoring(self):
        """Start performance monitoring background task"""
        logging.info("Starting performance monitoring...")
        self.monitoring_task = asyncio.create_task(self.monitor_system())

    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

    async def monitor_system(self):
        """Monitor system performance continuously"""

        while True:
            try:
                await self.collect_system_metrics()
                await self.check_performance_thresholds()
                await asyncio.sleep(300)  # Collect every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(60)

    async def collect_system_metrics(self):
        """Collect comprehensive system metrics"""

        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Network
            network = psutil.net_io_counters()

            # GPU metrics (if available)
            gpu_metrics = await self.get_gpu_metrics()

            metrics = {
                "timestamp": datetime.now(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "gpu_metrics": gpu_metrics,
            }

            self.system_metrics.append(metrics)

        except Exception as e:
            logging.error(f"Error collecting system metrics: {str(e)}")

    async def get_gpu_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU metrics if available"""

        try:
            import GPUtil

            gpus = GPUtil.getGPUs()

            if gpus:
                gpu = gpus[0]  # Assume single GPU
                return {
                    "gpu_utilization": gpu.load * 100,
                    "gpu_memory_used": gpu.memoryUsed,
                    "gpu_memory_total": gpu.memoryTotal,
                    "gpu_memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    "gpu_temperature": gpu.temperature,
                }

        except ImportError:
            # GPUtil not available
            pass
        except Exception as e:
            logging.debug(f"Could not get GPU metrics: {str(e)}")

        return None

    def track_request_performance(
        self,
        model: str,
        response_time: float,
        tokens_generated: int,
        success: bool = True,
    ):
        """Track individual request performance"""

        timestamp = datetime.now()

        # Overall request timing
        self.request_timings.append(
            {
                "timestamp": timestamp,
                "response_time": response_time,
                "model": model,
                "success": success,
            }
        )

        # Model-specific performance
        model_stats = self.model_performance[model]
        model_stats["response_times"].append(response_time)
        model_stats["total_requests"] += 1

        if not success:
            model_stats["error_count"] += 1

        # Calculate throughput (tokens per second)
        if tokens_generated > 0:
            throughput = tokens_generated / response_time
            model_stats["throughput"].append(throughput)

    async def check_performance_thresholds(self):
        """Check if performance metrics exceed thresholds"""

        current_metrics = await self.get_current_performance_summary()
        alerts_triggered = []

        # Check response time
        if current_metrics["response_time_p95"] > self.thresholds["response_time_p95"]:
            alerts_triggered.append(
                {
                    "type": "high_response_time",
                    "value": current_metrics["response_time_p95"],
                    "threshold": self.thresholds["response_time_p95"],
                    "timestamp": datetime.now(),
                }
            )

        # Check system resources
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]

            if latest_metrics["cpu_percent"] > self.thresholds["cpu_usage"]:
                alerts_triggered.append(
                    {
                        "type": "high_cpu_usage",
                        "value": latest_metrics["cpu_percent"],
                        "threshold": self.thresholds["cpu_usage"],
                        "timestamp": datetime.now(),
                    }
                )

            if latest_metrics["memory_percent"] > self.thresholds["memory_usage"]:
                alerts_triggered.append(
                    {
                        "type": "high_memory_usage",
                        "value": latest_metrics["memory_percent"],
                        "threshold": self.thresholds["memory_usage"],
                        "timestamp": datetime.now(),
                    }
                )

            # GPU checks
            gpu_metrics = latest_metrics.get("gpu_metrics")
            if (
                gpu_metrics
                and gpu_metrics["gpu_memory_percent"] > self.thresholds["gpu_memory"]
            ):
                alerts_triggered.append(
                    {
                        "type": "high_gpu_memory",
                        "value": gpu_metrics["gpu_memory_percent"],
                        "threshold": self.thresholds["gpu_memory"],
                        "timestamp": datetime.now(),
                    }
                )

        # Store alerts
        self.alerts.extend(alerts_triggered)

        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts if alert["timestamp"] > cutoff_time
        ]

        # Log critical alerts
        for alert in alerts_triggered:
            logging.warning(
                f"Performance alert: {alert['type']} = {alert['value']:.2f} (threshold: {alert['threshold']:.2f})"
            )

    async def get_current_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""

        summary = {
            "timestamp": datetime.now(),
            "total_requests": len(self.request_timings),
            "response_time_p50": 0,
            "response_time_p95": 0,
            "response_time_p99": 0,
            "error_rate": 0,
            "models_performance": {},
            "system_health": "unknown",
            "active_alerts": len(self.alerts),
        }

        # Calculate response time percentiles
        if self.request_timings:
            response_times = [req["response_time"] for req in self.request_timings]
            response_times.sort()

            n = len(response_times)
            summary["response_time_p50"] = response_times[int(n * 0.5)]
            summary["response_time_p95"] = response_times[int(n * 0.95)]
            summary["response_time_p99"] = response_times[int(n * 0.99)]

            # Calculate error rate
            errors = sum(1 for req in self.request_timings if not req["success"])
            summary["error_rate"] = (errors / n) * 100 if n > 0 else 0

        # Model-specific performance
        for model, stats in self.model_performance.items():
            if stats["response_times"]:
                avg_response_time = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )
                avg_throughput = (
                    sum(stats["throughput"]) / len(stats["throughput"])
                    if stats["throughput"]
                    else 0
                )
                error_rate = (
                    (stats["error_count"] / stats["total_requests"]) * 100
                    if stats["total_requests"] > 0
                    else 0
                )

                summary["models_performance"][model] = {
                    "avg_response_time": avg_response_time,
                    "avg_throughput_tokens_per_sec": avg_throughput,
                    "error_rate": error_rate,
                    "total_requests": stats["total_requests"],
                }

        # System health assessment
        if self.system_metrics:
            latest = self.system_metrics[-1]

            health_score = 100
            if latest["cpu_percent"] > 70:
                health_score -= 20
            if latest["memory_percent"] > 75:
                health_score -= 20
            if summary["error_rate"] > 1:
                health_score -= 30
            if summary["response_time_p95"] > 5:
                health_score -= 30

            if health_score >= 80:
                summary["system_health"] = "excellent"
            elif health_score >= 60:
                summary["system_health"] = "good"
            elif health_score >= 40:
                summary["system_health"] = "fair"
            else:
                summary["system_health"] = "poor"

        return summary

    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data"""

        recommendations = []

        if not self.request_timings:
            return ["No performance data available yet"]

        # Analyze response times by model
        model_avg_times = {}
        for model, stats in self.model_performance.items():
            if stats["response_times"]:
                model_avg_times[model] = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )

        # Recommend model optimizations
        if model_avg_times:
            slowest_model = max(model_avg_times, key=model_avg_times.get)
            fastest_model = min(model_avg_times, key=model_avg_times.get)

            if model_avg_times[slowest_model] > model_avg_times[fastest_model] * 2:
                recommendations.append(
                    f"Consider routing more simple queries to {fastest_model} "
                    f"instead of {slowest_model} to improve average response time"
                )

        # System resource recommendations
        if self.system_metrics:
            latest = self.system_metrics[-1]

            if latest["memory_percent"] > 80:
                recommendations.append(
                    "High memory usage detected. Consider implementing more aggressive caching cleanup or increasing memory limits"
                )

            if latest["cpu_percent"] > 80:
                recommendations.append(
                    "High CPU usage detected. Consider reducing concurrent request limits or scaling horizontally"
                )

            gpu_metrics = latest.get("gpu_metrics")
            if gpu_metrics and gpu_metrics["gpu_memory_percent"] > 85:
                recommendations.append(
                    "High GPU memory usage. Consider reducing max_loaded_models or using more aggressive model quantization"
                )

        # Caching recommendations
        recent_requests = list(self.request_timings)[-100:]  # Last 100 requests
        if len(recent_requests) >= 50:
            avg_response_time = sum(
                req["response_time"] for req in recent_requests
            ) / len(recent_requests)

            if avg_response_time > 3.0:
                recommendations.append(
                    "Average response time is high. Consider implementing semantic caching or increasing cache TTL"
                )

        return recommendations if recommendations else ["System performance is optimal"]
