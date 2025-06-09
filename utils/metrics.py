# utils/metrics.py - Metrics Collection Service
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

class MetricsCollector:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Request metrics
        self.request_counts = defaultdict(int)
        self.response_times = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Model usage metrics
        self.model_usage = defaultdict(lambda: {
            'requests': 0,
            'tokens': 0,
            'errors': 0,
            'total_time': 0.0
        })
        
        # User metrics
        self.user_requests = defaultdict(int)
        
        # Start time for uptime calculation
        self.start_time = datetime.now()
    
    def track_request(self, endpoint: str, user_id: str = "anonymous"):
        """Track a request"""
        with self._lock:
            self.request_counts[endpoint] += 1
            self.user_requests[user_id] += 1
    
    def track_response_time(self, response_time: float):
        """Track response time"""
        with self._lock:
            self.response_times.append(response_time)
    
    def track_error(self, error_type: str):
        """Track an error"""
        with self._lock:
            self.error_counts[error_type] += 1
    
    def track_cache_hit(self):
        """Track cache hit"""
        with self._lock:
            self.cache_hits += 1
    
    def track_cache_miss(self):
        """Track cache miss"""
        with self._lock:
            self.cache_misses += 1
    
    def track_model_usage(self, model: str, prompt_tokens: int, completion_tokens: int, processing_time: float = 0.0):
        """Track model usage"""
        with self._lock:
            usage = self.model_usage[model]
            usage['requests'] += 1
            usage['tokens'] += prompt_tokens + completion_tokens
            usage['total_time'] += processing_time
    
    def track_model_error(self, model: str):
        """Track model error"""
        with self._lock:
            self.model_usage[model]['errors'] += 1
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        with self._lock:
            total_requests = sum(self.request_counts.values())
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / max(1, total_cache_requests)
            
            # Calculate average response time
            avg_response_time = sum(self.response_times) / max(1, len(self.response_times))
            
            # Calculate uptime
            uptime = datetime.now() - self.start_time
            
            # Calculate cost estimates
            cost_analysis = self._calculate_cost_analysis()
            
            return {
                "overview": {
                    "total_requests": total_requests,
                    "total_errors": sum(self.error_counts.values()),
                    "error_rate": (sum(self.error_counts.values()) / max(1, total_requests)) * 100,
                    "avg_response_time": avg_response_time,
                    "uptime_seconds": uptime.total_seconds(),
                    "cache_hit_rate": cache_hit_rate
                },
                "requests": dict(self.request_counts),
                "errors": dict(self.error_counts),
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": cache_hit_rate
                },
                "models": dict(self.model_usage),
                "users": dict(self.user_requests),
                "cost_analysis": cost_analysis,
                "response_times": {
                    "recent": list(self.response_times)[-10:],  # Last 10 response times
                    "count": len(self.response_times)
                }
            }
    
    def _calculate_cost_analysis(self) -> Dict[str, Any]:
        """Calculate cost analysis for models"""
        
        # Model cost estimates (per 1K tokens)
        model_costs = {
            'mistral:7b-instruct-q4_0': 0.0001,
            'deepseek-v2:7b-q4_0': 0.00015,
            'llama3:8b-instruct-q4_0': 0.00012
        }
        
        total_cost = 0.0
        cost_by_model = {}
        
        for model, usage in self.model_usage.items():
            cost_per_1k = model_costs.get(model, 0.0001)  # Default cost
            model_cost = (usage['tokens'] / 1000) * cost_per_1k
            total_cost += model_cost
            
            cost_by_model[model] = {
                "cost": model_cost,
                "tokens": usage['tokens'],
                "requests": usage['requests']
            }
        
        # Calculate percentages
        for model, data in cost_by_model.items():
            data["percentage"] = (data["cost"] / max(0.001, total_cost)) * 100
        
        return {
            "total_estimated_cost": total_cost,
            "cost_by_model": cost_by_model,
            "avg_cost_per_request": total_cost / max(1, sum(self.request_counts.values()))
        }
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for dashboard"""
        with self._lock:
            recent_errors = sum(1 for _ in range(min(10, len(self.response_times))))
            
            return {
                "requests_last_minute": len([t for t in self.response_times if time.time() - t < 60]),
                "avg_response_time_last_10": sum(list(self.response_times)[-10:]) / max(1, min(10, len(self.response_times))),
                "cache_hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "active_models": len([m for m, usage in self.model_usage.items() if usage['requests'] > 0])
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self.request_counts.clear()
            self.response_times.clear()
            self.error_counts.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.model_usage.clear()
            self.user_requests.clear()
            self.start_time = datetime.now()
