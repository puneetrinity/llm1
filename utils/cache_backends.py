# utils/cache_backends.py - Cache Backend Abstraction Layer
import json
import time
import logging
import hashlib
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pickle
    import gzip

    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False


@dataclass
class CacheEntry:
    """Standardized cache entry with metadata"""

    key: str
    value: Any
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.metadata is None:
            self.metadata = {}
        if self.size_bytes == 0:
            self.size_bytes = len(str(self.value).encode("utf-8"))

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return time.time() > self.expires_at

    def access(self):
        """Mark entry as accessed"""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary"""
        return cls(**data)


class CacheBackend(ABC):
    """Abstract base class for cache backends"""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry by key"""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None
    ) -> bool:
        """Set cache entry"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "memory_usage_bytes": 0,
        }

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with LRU tracking"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        entry = self.cache[key]

        # Check expiration
        if entry.is_expired():
            await self.delete(key)
            self.stats["misses"] += 1
            return None

        # Update access tracking
        entry.access()
        self._update_access_order(key)
        self.stats["hits"] += 1

        return entry

    async def set(
        self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None
    ) -> bool:
        """Set cache entry with eviction if needed"""
        try:
            # Remove old entry if exists
            if key in self.cache:
                await self.delete(key)

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=time.time() + ttl,
                metadata=metadata or {},
            )

            # Check memory limits before adding
            if not self._can_add_entry(entry):
                await self._evict_entries(entry.size_bytes)

            # Add entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.stats["sets"] += 1
            self.stats["memory_usage_bytes"] += entry.size_bytes

            # Evict if over size limit
            await self._evict_if_needed()

            return True

        except Exception as e:
            logging.error(f"Memory cache set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats["memory_usage_bytes"] -= entry.size_bytes
            self.stats["deletes"] += 1

            if key in self.access_order:
                self.access_order.remove(key)

            return True
        return False

    async def clear(self) -> bool:
        """Clear all entries"""
        self.cache.clear()
        self.access_order.clear()
        self.stats["memory_usage_bytes"] = 0
        return True

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired"""
        if key not in self.cache:
            return False

        entry = self.cache[key]
        if entry.is_expired():
            await self.delete(key)
            return False

        return True

    def _can_add_entry(self, entry: CacheEntry) -> bool:
        """Check if entry can be added without exceeding memory limit"""
        return (
            self.stats["memory_usage_bytes"] + entry.size_bytes
        ) <= self.max_memory_bytes

    async def _evict_entries(self, needed_bytes: int):
        """Evict entries to make space"""
        freed_bytes = 0

        # Evict least recently used entries
        while freed_bytes < needed_bytes and self.access_order:
            lru_key = self.access_order[0]
            if lru_key in self.cache:
                freed_bytes += self.cache[lru_key].size_bytes
                await self.delete(lru_key)
                self.stats["evictions"] += 1

    async def _evict_if_needed(self):
        """Evict entries if over size limit"""
        while len(self.cache) > self.max_size and self.access_order:
            lru_key = self.access_order[0]
            if lru_key in self.cache:
                await self.delete(lru_key)
                self.stats["evictions"] += 1

    def _update_access_order(self, key: str):
        """Update LRU access order"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(1, total_requests)) * 100

        return {
            "backend_type": "memory",
            "hit_rate": hit_rate,
            "entries": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": self.stats["memory_usage_bytes"] / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "stats": self.stats,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check memory cache health"""
        memory_usage_percent = (
            self.stats["memory_usage_bytes"] / self.max_memory_bytes
        ) * 100

        health_status = "healthy"
        if memory_usage_percent > 90:
            health_status = "critical"
        elif memory_usage_percent > 75:
            health_status = "warning"

        return {
            "status": health_status,
            "memory_usage_percent": memory_usage_percent,
            "entries": len(self.cache),
            "evictions": self.stats["evictions"],
        }


class RedisCacheBackend(CacheBackend):
    """Redis cache backend with compression and serialization"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        db: int = 0,
        timeout: int = 5,
        enable_compression: bool = True,
        key_prefix: str = "llm_cache:",
    ):
        self.redis_url = redis_url
        self.db = db
        self.timeout = timeout
        self.enable_compression = enable_compression and COMPRESSION_AVAILABLE
        self.key_prefix = key_prefix
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False

        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
            "compression_ratio": 0,
        }

    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized or not REDIS_AVAILABLE:
            return False

        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                db=self.db,
                socket_timeout=self.timeout,
                decode_responses=False,  # We handle encoding ourselves
            )

            # Test connection
            await self.redis_client.ping()
            self._initialized = True
            logging.info("Redis cache backend initialized")
            return True

        except Exception as e:
            logging.error(f"Redis initialization failed: {e}")
            self.redis_client = None
            return False

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key"""
        return f"{self.key_prefix}{key}"

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry"""
        data = entry.to_dict()
        serialized = json.dumps(data, default=str).encode("utf-8")

        if self.enable_compression:
            try:
                compressed = gzip.compress(serialized)
                # Only use compression if it saves space
                if len(compressed) < len(serialized):
                    self.stats["compression_ratio"] = len(compressed) / len(serialized)
                    return compressed
            except Exception:
                pass

        return serialized

    def _deserialize_entry(self, data: bytes) -> Optional[CacheEntry]:
        """Deserialize cache entry"""
        try:
            # Try decompression first
            if self.enable_compression:
                try:
                    data = gzip.decompress(data)
                except Exception:
                    pass  # Not compressed

            # Deserialize JSON
            entry_dict = json.loads(data.decode("utf-8"))
            return CacheEntry.from_dict(entry_dict)

        except Exception as e:
            logging.error(f"Redis deserialization error: {e}")
            return None

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry from Redis"""
        if not self._initialized:
            await self.initialize()

        if not self.redis_client:
            self.stats["misses"] += 1
            return None

        try:
            redis_key = self._make_key(key)
            data = await self.redis_client.get(redis_key)

            if not data:
                self.stats["misses"] += 1
                return None

            entry = self._deserialize_entry(data)
            if not entry:
                self.stats["misses"] += 1
                return None

            # Check expiration (Redis TTL should handle this, but double-check)
            if entry.is_expired():
                await self.delete(key)
                self.stats["misses"] += 1
                return None

            entry.access()
            self.stats["hits"] += 1
            return entry

        except Exception as e:
            logging.error(f"Redis get error: {e}")
            self.stats["errors"] += 1
            self.stats["misses"] += 1
            return None

    async def set(
        self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None
    ) -> bool:
        """Set cache entry in Redis"""
        if not self._initialized:
            await self.initialize()

        if not self.redis_client:
            return False

        try:
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                expires_at=time.time() + ttl,
                metadata=metadata or {},
            )

            redis_key = self._make_key(key)
            serialized_data = self._serialize_entry(entry)

            await self.redis_client.setex(redis_key, ttl, serialized_data)
            self.stats["sets"] += 1
            return True

        except Exception as e:
            logging.error(f"Redis set error: {e}")
            self.stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """Delete cache entry from Redis"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            result = await self.redis_client.delete(redis_key)
            if result > 0:
                self.stats["deletes"] += 1
            return result > 0

        except Exception as e:
            logging.error(f"Redis delete error: {e}")
            self.stats["errors"] += 1
            return False

    async def clear(self) -> bool:
        """Clear all cache entries with prefix"""
        if not self.redis_client:
            return False

        try:
            # Get all keys with prefix
            pattern = f"{self.key_prefix}*"
            keys = []

            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self.redis_client.delete(*keys)

            return True

        except Exception as e:
            logging.error(f"Redis clear error: {e}")
            self.stats["errors"] += 1
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.redis_client:
            return False

        try:
            redis_key = self._make_key(key)
            return await self.redis_client.exists(redis_key) > 0

        except Exception as e:
            logging.error(f"Redis exists error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(1, total_requests)) * 100

        return {
            "backend_type": "redis",
            "initialized": self._initialized,
            "hit_rate": hit_rate,
            "compression_enabled": self.enable_compression,
            "compression_ratio": self.stats["compression_ratio"],
            "stats": self.stats,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health"""
        if not self._initialized:
            return {"status": "unavailable", "message": "Redis not initialized"}

        try:
            # Test Redis connection
            pong = await self.redis_client.ping()

            # Get Redis info
            info = await self.redis_client.info()
            memory_usage = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)

            health_status = "healthy"
            if max_memory > 0 and memory_usage > max_memory * 0.9:
                health_status = "critical"
            elif max_memory > 0 and memory_usage > max_memory * 0.75:
                health_status = "warning"

            return {
                "status": health_status,
                "ping": pong,
                "memory_usage_mb": memory_usage / (1024 * 1024),
                "max_memory_mb": max_memory / (1024 * 1024) if max_memory > 0 else None,
                "connected_clients": info.get("connected_clients", 0),
                "errors": self.stats["errors"],
            }

        except Exception as e:
            logging.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


class MultiTierCacheBackend(CacheBackend):
    """Multi-tier cache backend (Memory + Redis)"""

    def __init__(
        self,
        memory_backend: MemoryCacheBackend,
        redis_backend: RedisCacheBackend,
        write_through: bool = True,
    ):
        self.memory_backend = memory_backend
        self.redis_backend = redis_backend
        self.write_through = write_through

        self.stats = {
            "l1_hits": 0,  # Memory hits
            "l2_hits": 0,  # Redis hits
            "misses": 0,
            "sets": 0,
        }

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get from memory first, then Redis"""
        # Try L1 cache (memory) first
        entry = await self.memory_backend.get(key)
        if entry:
            self.stats["l1_hits"] += 1
            return entry

        # Try L2 cache (Redis)
        entry = await self.redis_backend.get(key)
        if entry:
            self.stats["l2_hits"] += 1

            # Populate L1 cache
            await self.memory_backend.set(
                key, entry.value, int(entry.expires_at - time.time()), entry.metadata
            )
            return entry

        self.stats["misses"] += 1
        return None

    async def set(
        self, key: str, value: Any, ttl: int = 3600, metadata: Dict[str, Any] = None
    ) -> bool:
        """Set in both tiers"""
        success = True

        # Set in memory cache
        memory_success = await self.memory_backend.set(key, value, ttl, metadata)
        if not memory_success:
            success = False

        # Set in Redis cache
        if self.write_through:
            redis_success = await self.redis_backend.set(key, value, ttl, metadata)
            if not redis_success:
                success = False

        if success:
            self.stats["sets"] += 1

        return success

    async def delete(self, key: str) -> bool:
        """Delete from both tiers"""
        memory_result = await self.memory_backend.delete(key)
        redis_result = await self.redis_backend.delete(key)
        return memory_result or redis_result

    async def clear(self) -> bool:
        """Clear both tiers"""
        memory_result = await self.memory_backend.clear()
        redis_result = await self.redis_backend.clear()
        return memory_result and redis_result

    async def exists(self, key: str) -> bool:
        """Check existence in either tier"""
        return await self.memory_backend.exists(key) or await self.redis_backend.exists(
            key
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics"""
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"]
        total_requests = total_hits + self.stats["misses"]

        overall_hit_rate = (total_hits / max(1, total_requests)) * 100
        l1_hit_rate = (self.stats["l1_hits"] / max(1, total_requests)) * 100
        l2_hit_rate = (self.stats["l2_hits"] / max(1, total_requests)) * 100

        return {
            "backend_type": "multi_tier",
            "overall_hit_rate": overall_hit_rate,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "tier_stats": {
                "memory": self.memory_backend.get_stats(),
                "redis": self.redis_backend.get_stats(),
            },
            "stats": self.stats,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check health of both tiers"""
        memory_health = await self.memory_backend.health_check()
        redis_health = await self.redis_backend.health_check()

        # Overall health is healthy if at least memory tier is healthy
        overall_status = "healthy"
        if memory_health["status"] != "healthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "tiers": {"memory": memory_health, "redis": redis_health},
        }


# Factory functions for easy creation


def create_memory_cache_backend(
    max_size: int = 1000, max_memory_mb: int = 100
) -> MemoryCacheBackend:
    """Create memory cache backend"""
    return MemoryCacheBackend(max_size, max_memory_mb)


def create_redis_cache_backend(
    redis_url: str = "redis://localhost:6379", **kwargs
) -> RedisCacheBackend:
    """Create Redis cache backend"""
    return RedisCacheBackend(redis_url, **kwargs)


async def create_smart_cache_backend(
    redis_url: str = "redis://localhost:6379",
    memory_size: int = 1000,
    memory_mb: int = 100,
    prefer_redis: bool = True,
) -> CacheBackend:
    """Create the best available cache backend"""

    # Try to create Redis backend
    redis_backend = create_redis_cache_backend(redis_url)
    redis_available = await redis_backend.initialize()

    # Always create memory backend
    memory_backend = create_memory_cache_backend(memory_size, memory_mb)

    # Return best option
    if redis_available and prefer_redis:
        return MultiTierCacheBackend(memory_backend, redis_backend)
    else:
        logging.info("Using memory-only cache backend")
        return memory_backend
