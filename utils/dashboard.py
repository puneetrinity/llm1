# Advanced monitoring dashboard
# utils/dashboard.py - Real-time Dashboard for Enhanced Features
import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging


class EnhancedDashboard:
    def __init__(self, metrics_collector, performance_monitor, cache_service, warmup_service, semantic_classifier):
        self.metrics = metrics_collector
        self.performance = performance_monitor
        self.cache = cache_service
        self.warmup = warmup_service
        self.classifier = semantic_classifier

    async def get_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""

        try:
            # Gather data from all services
            basic_metrics = await self.metrics.get_all_metrics()
            performance_data = await self.performance.get_current_performance_summary()
            cache_stats = await self.cache.get_stats()
            warmup_stats = self.warmup.get_warmup_stats()
            classification_stats = self.classifier.get_classification_stats()

            # Calculate derived metrics
            cost_efficiency = self._calculate_cost_efficiency(basic_metrics)
            system_health = self._assess_system_health(performance_data)
            optimization_suggestions = self._get_optimization_suggestions(
                performance_data, basic_metrics)

            dashboard = {
                "timestamp": datetime.now().isoformat(),
                "system_overview": {
                    "status": system_health["status"],
                    "health_score": system_health["score"],
                    "uptime_hours": self._calculate_uptime(),
                    "total_requests_24h": basic_metrics["overview"]["total_requests"],
                    "avg_response_time": performance_data["response_time_p50"],
                    "error_rate": performance_data["error_rate"],
                    "cost_per_request": cost_efficiency["cost_per_request"],
                    "estimated_monthly_cost": cost_efficiency["monthly_estimate"]
                },

                "performance_metrics": {
                    "response_times": {
                        "p50": performance_data["response_time_p50"],
                        "p95": performance_data["response_time_p95"],
                        "p99": performance_data["response_time_p99"]
                    },
                    "throughput": {
                        "requests_per_minute": self._calculate_rpm(),
                        "tokens_per_second": self._calculate_tps(basic_metrics)
                    },
                    "system_resources": {
                        "cpu_usage": self._get_latest_cpu(),
                        "memory_usage": self._get_latest_memory(),
                        "gpu_usage": self._get_latest_gpu()
                    }
                },

                "model_analytics": {
                    "usage_distribution": self._calculate_model_distribution(basic_metrics),
                    "performance_by_model": performance_data["models_performance"],
                    "cost_by_model": basic_metrics["cost_analysis"]["cost_by_model"],
                    "warmup_status": warmup_stats
                },

                "intelligent_features": {
                    "semantic_classification": {
                        "enabled": classification_stats["model_loaded"],
                        "total_examples": classification_stats["total_training_examples"],
                        "classification_accuracy": self._estimate_classification_accuracy(),
                        "fallback_rate": self._calculate_fallback_rate()
                    },
                    "caching_performance": {
                        "hit_rate": cache_stats["hit_rate"],
                        "cache_size": cache_stats["cache_size"],
                        "memory_usage_mb": cache_stats["memory_usage_mb"],
                        "estimated_cost_savings": self._calculate_cache_savings(cache_stats, basic_metrics)
                    },
                    "streaming_stats": {
                        "streaming_requests": self._count_streaming_requests(),
                        "avg_stream_duration": self._calculate_avg_stream_duration(),
                        "stream_completion_rate": self._calculate_stream_completion_rate()
                    }
                },

                "cost_optimization": {
                    "total_estimated_cost": basic_metrics["cost_analysis"]["total_estimated_cost"],
                    "cost_breakdown": basic_metrics["cost_analysis"]["cost_by_model"],
                    "routing_efficiency": self._calculate_routing_efficiency(basic_metrics),
                    # Estimate 30% savings per cache hit
                    "cache_savings_percent": (cache_stats["hit_rate"] * 30),
                    "optimization_score": self._calculate_optimization_score(basic_metrics, performance_data)
                },

                "alerts_and_recommendations": {
                    "active_alerts": performance_data["active_alerts"],
                    "recent_alerts": self.performance.alerts[-5:] if self.performance.alerts else [],
                    "optimization_recommendations": optimization_suggestions,
                    "next_suggested_action": self._get_next_action(optimization_suggestions)
                },

                "trends": {
                    "hourly_request_trend": basic_metrics.get("hourly_trends", []),
                    "model_usage_trend": self._calculate_model_usage_trend(),
                    "performance_trend": self._calculate_performance_trend(),
                    "cost_trend": self._calculate_cost_trend()
                }
            }

            return dashboard

        except Exception as e:
            logging.error(f"Error generating dashboard: {str(e)}")
            return self._get_error_dashboard(str(e))

    def _calculate_cost_efficiency(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cost efficiency metrics"""

        total_cost = metrics["cost_analysis"]["total_estimated_cost"]
        total_requests = metrics["overview"]["total_requests"]

        cost_per_request = total_cost / max(1, total_requests)

        # Estimate monthly cost based on current usage
        requests_per_hour = total_requests / max(1, self._calculate_uptime())
        monthly_requests = requests_per_hour * 24 * 30
        monthly_estimate = monthly_requests * cost_per_request

        return {
            "cost_per_request": cost_per_request,
            "monthly_estimate": monthly_estimate,
            "efficiency_score": self._calculate_efficiency_score(cost_per_request)
        }

    def _assess_system_health(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health"""

        score = 100

        # Response time impact
        if performance["response_time_p95"] > 10:
            score -= 30
        elif performance["response_time_p95"] > 5:
            score -= 15

        # Error rate impact
        if performance["error_rate"] > 5:
            score -= 40
        elif performance["error_rate"] > 1:
            score -= 20

        # Alert impact
        if performance["active_alerts"] > 3:
            score -= 25
        elif performance["active_alerts"] > 0:
            score -= 10

        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        elif score >= 40:
            status = "poor"
        else:
            status = "critical"

        return {"status": status, "score": score}

    def _get_optimization_suggestions(self, performance: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Generate intelligent optimization suggestions"""

        suggestions = []

        # Performance suggestions
        if performance["response_time_p95"] > 5:
            suggestions.append(
                "Consider enabling more aggressive caching to reduce response times")

        if performance["error_rate"] > 2:
            suggestions.append(
                "High error rate detected - check model availability and warmup settings")

        # Cost optimization suggestions
        model_costs = metrics["cost_analysis"]["cost_by_model"]
        if model_costs:
            most_expensive = max(model_costs.items(),
                                 key=lambda x: x[1]["cost"])
            if most_expensive[1]["percentage"] > 60:
                suggestions.append(
                    f"Consider routing more queries away from {most_expensive[0]} to reduce costs")

        # Resource suggestions
        cache_stats = self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}
        if cache_stats.get("hit_rate", 0) < 0.3:
            suggestions.append(
                "Low cache hit rate - consider tuning semantic similarity threshold")

        # Warmup suggestions
        warmup_stats = self.warmup.get_warmup_stats()
        if any(time > 600 for time in warmup_stats.get("time_since_last_warmup", {}).values()):
            suggestions.append(
                "Some models haven't been warmed up recently - consider reducing warmup interval")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours"""
        # This would track actual startup time in a real implementation
        return 24.0  # Placeholder

    def _calculate_rpm(self) -> float:
        """Calculate requests per minute"""
        if self.metrics.request_counts:
            total_requests = sum(self.metrics.request_counts.values())
            uptime_minutes = self._calculate_uptime() * 60
            return total_requests / max(1, uptime_minutes)
        return 0.0

    def _calculate_tps(self, metrics: Dict[str, Any]) -> float:
        """Calculate tokens per second"""
        total_tokens = 0
        for model_usage in metrics["models"].values():
            total_tokens += model_usage.get("tokens", 0)

        uptime_seconds = self._calculate_uptime() * 3600
        return total_tokens / max(1, uptime_seconds)

    def _get_latest_cpu(self) -> float:
        """Get latest CPU usage"""
        if self.performance.system_metrics:
            return self.performance.system_metrics[-1]["cpu_percent"]
        return 0.0

    def _get_latest_memory(self) -> float:
        """Get latest memory usage"""
        if self.performance.system_metrics:
            return self.performance.system_metrics[-1]["memory_percent"]
        return 0.0

    def _get_latest_gpu(self) -> Dict[str, float]:
        """Get latest GPU usage"""
        if self.performance.system_metrics and self.performance.system_metrics[-1].get("gpu_metrics"):
            gpu = self.performance.system_metrics[-1]["gpu_metrics"]
            return {
                "utilization": gpu.get("gpu_utilization", 0),
                "memory_percent": gpu.get("gpu_memory_percent", 0),
                "temperature": gpu.get("gpu_temperature", 0)
            }
        return {"utilization": 0, "memory_percent": 0, "temperature": 0}

    def _calculate_model_distribution(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model usage distribution"""

        total_requests = sum(
            model_data.get("requests", 0)
            for model_data in metrics["models"].values()
        )

        if total_requests == 0:
            return {}

        return {
            model: (model_data.get("requests", 0) / total_requests) * 100
            for model, model_data in metrics["models"].items()
        }

    def _estimate_classification_accuracy(self) -> float:
        """Estimate semantic classification accuracy"""
        # This would be calculated based on feedback or validation in a real implementation
        return 85.0  # Placeholder

    def _calculate_fallback_rate(self) -> float:
        """Calculate semantic classification fallback rate"""
        # Rate at which semantic classification falls back to rule-based
        return 15.0  # Placeholder

    def _calculate_cache_savings(self, cache_stats: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate estimated cost savings from caching"""

        hit_rate = cache_stats.get("hit_rate", 0)
        total_cost = metrics["cost_analysis"]["total_estimated_cost"]

        # Estimate savings (cache hits avoid model inference costs)
        estimated_savings = total_cost * hit_rate * \
            0.3  # 30% cost reduction per cache hit

        return estimated_savings

    def _count_streaming_requests(self) -> int:
        """Count streaming requests"""
        # This would track actual streaming requests in a real implementation
        return 0  # Placeholder

    def _calculate_avg_stream_duration(self) -> float:
        """Calculate average streaming duration"""
        return 0.0  # Placeholder

    def _calculate_stream_completion_rate(self) -> float:
        """Calculate streaming completion rate"""
        return 100.0  # Placeholder

    def _calculate_routing_efficiency(self, metrics: Dict[str, Any]) -> float:
        """Calculate routing efficiency score"""

        # Efficiency based on using cheaper models appropriately
        model_costs = metrics["cost_analysis"]["cost_by_model"]

        if not model_costs:
            return 0.0

        # Check if cheaper models (Mistral) are being used frequently
        mistral_usage = 0
        total_cost = 0

        for model, data in model_costs.items():
            if "mistral" in model.lower():
                mistral_usage += data["percentage"]
            total_cost += data["cost"]

        # Higher Mistral usage = better efficiency for simple tasks
        efficiency_score = min(100, mistral_usage * 2)  # Scale to 0-100

        return efficiency_score

    def _calculate_optimization_score(self, metrics: Dict[str, Any], performance: Dict[str, Any]) -> float:
        """Calculate overall optimization score"""

        score = 100

        # Cost efficiency (30% weight)
        cost_per_request = metrics["cost_analysis"]["total_estimated_cost"] / max(
            1, metrics["overview"]["total_requests"])
        if cost_per_request > 0.01:  # More than 1 cent per request
            score -= 30
        elif cost_per_request > 0.005:
            score -= 15

        # Performance efficiency (40% weight)
        if performance["response_time_p95"] > 5:
            score -= 20
        if performance["error_rate"] > 1:
            score -= 20

        # Resource efficiency (30% weight)
        cache_hit_rate = getattr(self.cache, 'hit_rate', 0.5)  # Default 50%
        if cache_hit_rate < 0.3:
            score -= 15

        routing_efficiency = self._calculate_routing_efficiency(metrics)
        if routing_efficiency < 50:
            score -= 15

        return max(0, score)

    def _get_next_action(self, suggestions: List[str]) -> str:
        """Get the next recommended action"""

        if not suggestions:
            return "System is well optimized. Monitor performance trends."

        # Prioritize suggestions based on impact
        high_priority_keywords = ["error", "critical", "cost", "memory"]

        for suggestion in suggestions:
            if any(keyword in suggestion.lower() for keyword in high_priority_keywords):
                return suggestion

        # Return first suggestion if no high priority found
        return suggestions[0]

    def _calculate_model_usage_trend(self) -> List[Dict[str, Any]]:
        """Calculate model usage trend over time"""
        # This would track actual usage over time
        return []  # Placeholder

    def _calculate_performance_trend(self) -> List[Dict[str, Any]]:
        """Calculate performance trend over time"""
        # Based on response times over time
        return []  # Placeholder

    def _calculate_cost_trend(self) -> List[Dict[str, Any]]:
        """Calculate cost trend over time"""
        # Based on costs over time
        return []  # Placeholder

    def _calculate_efficiency_score(self, cost_per_request: float) -> float:
        """Calculate efficiency score based on cost per request"""

        # Benchmark: $0.001 per request = 100 score
        benchmark_cost = 0.001

        if cost_per_request <= benchmark_cost:
            return 100.0
        else:
            # Score decreases as cost increases
            return max(0, 100 - ((cost_per_request - benchmark_cost) / benchmark_cost) * 100)

    def _get_error_dashboard(self, error_message: str) -> Dict[str, Any]:
        """Return error dashboard when main dashboard fails"""

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error_message": error_message,
            "system_overview": {
                "status": "unknown",
                "health_score": 0,
                "message": "Dashboard data collection failed"
            }
        }
