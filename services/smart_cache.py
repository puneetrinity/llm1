# services/smart_cache.py - Intelligent Caching with Redis and Semantic Similarity
import json
import hashlib
import time
import logging
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

# Optional imports with fallbacks
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.info("Redis not available - using memory-only cache")

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.info("Sentence transformers not available - semantic caching disabled")

from utils.memory_manager import allocate_memory, deallocate_memory

@dataclass
class CacheConfig:
    """Configuration for smart caching system"""
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_timeout: int = 5
    
    # Memory cache settings
    memory_cache_size: int = 1000
    memory_ttl_seconds: int = 3600
    
    # Semantic settings
    semantic_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    semantic_cache_size: int = 500
    
    # Performance settings
    enable_compression: bool = True
    enable_async_write: bool = True
    cache_hit_logging: bool = False

class CacheEntry:
    """Represents a cached entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.access_count = 0
        self.last_accessed = self.created_at
        self.size_bytes = len(str(value).encode('utf-8'))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at
    
    def access(self):
        """Mark entry as accessed"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'size_bytes': self.size_bytes
        }

class MemoryCache:
    """In-memory cache with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if entry.is_expired():
            await self.delete(key)
            return None
        
        entry.access()
        # Move to end of access order (most recently used)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in memory cache"""
        # Remove old entry if exists
        if key in self.cache:
            await self.delete(key)
        
        # Create new entry
        entry = CacheEntry(key, value, ttl)
        self.cache[key] = entry
        self.access_order.append(key)
        
        # Evict if necessary
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    async def delete(self, key: str):
        """Delete entry from memory cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    async def clear(self):
        """Clear all entries"""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        return {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'total_size_bytes': total_size,
            'avg_access_count': np.mean([e.access_count for e in self.cache.values()]) if self.cache else 0
        }

class SemanticIndex:
    """Semantic similarity index for intelligent caching"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.embeddings: Dict[str, np.ndarray] = {}
        self.key_to_content: Dict[str, str] = {}
        self.model_name = model_name
        self._initialized = False
    
    async def initialize(self):
        """Initialize semantic model"""
        if self._initialized or not SEMANTIC_AVAILABLE:
            return
        
        try:
            # Allocate memory for semantic model
            if not allocate_memory('semantic_model', 500):  # 500MB for model
                logging.warning("Could not allocate memory for semantic model")
                return
            
            logging.info(f"Loading semantic model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._initialized = True
            logging.info("Semantic model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize semantic model: {e}")
            deallocate_memory('semantic_model')
    
    async def add_embedding(self, key: str, content: str):
        """Add content embedding to index"""
        if not self._initialized:
            return
        
        try:
            # Generate embedding
            embedding = self.model.encode([content])[0]
            self.embeddings[key] = embedding
            self.key_to_content[key] = content
            
            # Limit index size
            if len(self.embeddings) > 1000:
                # Remove oldest 20% of entries
                keys_to_remove = list(self.embeddings.keys())[:200]
                for k in keys_to_remove:
                    self.embeddings.pop(k, None)
                    self.key_to_content.pop(k, None)
                    
        except Exception as e:
            logging.error(f"Failed to add embedding: {e}")
    
    async def find_similar(self, content: str, threshold: float = 0.85, max_results: int = 3) -> List[tuple]:
        """Find similar cached content"""
        if not self._initialized or not self.embeddings:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([content])[0]
            
            # Calculate similarities
            similarities = []
            for key, embedding in self.embeddings.items():
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                if similarity >= threshold:
                    similarities.append((key, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            logging.error(f"Similarity search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get semantic index statistics"""
        return {
            'initialized': self._initialized,
            'model_name': self.model_name,
            'embeddings_count': len(self.embeddings),
            'content_keys': len(self.key_to_content)
        }

class SmartCache:
    """Intelligent caching system with Redis, memory cache, and semantic similarity"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache = MemoryCache(self.config.memory_cache_size)
        self.semantic_index = SemanticIndex(self.config.semantic_model)
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'redis_hits': 0,
            'semantic_hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0
        }
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all cache backends"""
        if self._initialized:
            return
        
        # Initialize Redis if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    db=self.config.redis_db,
                    socket_timeout=self.config.redis_timeout,
                    decode_responses=True
                )
                # Test connection
                await self.redis_client.ping()
                logging.info("Redis cache backend initialized")
            except Exception as e:
                logging.warning(f"Redis initialization failed: {e}")
                self.redis_client = None
        
        # Initialize semantic index
        await self.semantic_index.initialize()
        
        self._initialized = True
        logging.info("Smart cache system initialized")
    
    async def get(self, key: str, query_text: str = None) -> Optional[Any]:
        """Get value from cache with semantic fallback"""
        if not self._initialized:
            await self.initialize()
        
        # Try memory cache first (fastest)
        value = await self.memory_cache.get(key)
        if value is not None:
            self.stats['memory_hits'] += 1
            if self.config.cache_hit_logging:
                logging.debug(f"Memory cache hit: {key}")
            return value
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(key)
                if cached_data:
                    value = json.loads(cached_data)
                    # Populate memory cache
                    await self.memory_cache.set(key, value, self.config.memory_ttl_seconds)
                    self.stats['redis_hits'] += 1
                    if self.config.cache_hit_logging:
                        logging.debug(f"Redis cache hit: {key}")
                    return value
            except Exception as e:
                logging.error(f"Redis get error: {e}")
                self.stats['errors'] += 1
        
        # Try semantic similarity search
        if query_text and self.semantic_index._initialized:
            try:
                similar_keys = await self.semantic_index.find_similar(
                    query_text, 
                    self.config.similarity_threshold
                )
                
                for similar_key, similarity in similar_keys:
                    # Try to get the similar content
                    similar_value = await self.memory_cache.get(similar_key)
                    if similar_value is None and self.redis_client:
                        try:
                            cached_data = await self.redis_client.get(similar_key)
                            if cached_data:
                                similar_value = json.loads(cached_data)
                        except Exception as e:
                            logging.error(f"Redis semantic get error: {e}")
                    
                    if similar_value is not None:
                        self.stats['semantic_hits'] += 1
                        if self.config.cache_hit_logging:
                            logging.info(f"Semantic cache hit: {similar_key} (similarity: {similarity:.3f})")
                        
                        # Cache this result with the new key
                        await self.set(key, similar_value, query_text=query_text)
                        return similar_value
                        
            except Exception as e:
                logging.error(f"Semantic search error: {e}")
                self.stats['errors'] += 1
        
        self.stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None, query_text: str = None):
        """Set value in cache with semantic indexing"""
        if not self._initialized:
            await self.initialize()
        
        effective_ttl = ttl or self.config.memory_ttl_seconds
        
        try:
            # Set in memory cache
            await self.memory_cache.set(key, value, effective_ttl)
            
            # Set in Redis cache
            if self.redis_client:
                try:
                    await self.redis_client.setex(
                        key, 
                        effective_ttl, 
                        json.dumps(value, default=str)
                    )
                except Exception as e:
                    logging.error(f"Redis set error: {e}")
                    self.stats['errors'] += 1
            
            # Add to semantic index
            if query_text:
                await self.semantic_index.add_embedding(key, query_text)
            
            self.stats['sets'] += 1
            
            # Async write optimization
            if self.config.enable_async_write:
                # Don't wait for Redis write to complete
                pass
                
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            self.stats['errors'] += 1
    
    async def delete(self, key: str):
        """Delete value from all cache layers"""
        try:
            await self.memory_cache.delete(key)
            
            if self.redis_client:
                await self.redis_client.delete(key)
            
            # Remove from semantic index
            self.semantic_index.embeddings.pop(key, None)
            self.semantic_index.key_to_content.pop(key, None)
            
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            self.stats['errors'] += 1
    
    async def clear(self):
        """Clear all cache layers"""
        try:
            await self.memory_cache.clear()
            
            if self.redis_client:
                await self.redis_client.flushdb()
            
            self.semantic_index.embeddings.clear()
            self.semantic_index.key_to_content.clear()
            
        except Exception as e:
            logging.error(f"Cache clear error: {e}")
            self.stats['errors'] += 1
    
    def generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate consistent cache key from request data"""
        # Create a normalized representation of the request
        key_data = {
            'model': request_data.get('model'),
            'messages': request_data.get('messages', []),
            'temperature': request_data.get('temperature', 0.7),
            'max_tokens': request_data.get('max_tokens'),
            'top_p': request_data.get('top_p', 1.0)
        }
        
        # Create hash of the key data
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return f"llm_cache:{hashlib.sha256(key_string.encode()).hexdigest()}"
    
    def extract_query_text(self, request_data: Dict[str, Any]) -> str:
        """Extract searchable text from request for semantic indexing"""
        messages = request_data.get('messages', [])
        if messages:
            # Use the last user message for semantic search
            for message in reversed(messages):
                if message.get('role') == 'user':
                    return message.get('content', '')
        return ''
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = sum([
            self.stats['memory_hits'],
            self.stats['redis_hits'], 
            self.stats['semantic_hits'],
            self.stats['misses']
        ])
        
        hit_rate = 0
        if total_requests > 0:
            total_hits = self.stats['memory_hits'] + self.stats['redis_hits'] + self.stats['semantic_hits']
            hit_rate = (total_hits / total_requests) * 100
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'breakdown': {
                'memory_hits': self.stats['memory_hits'],
                'redis_hits': self.stats['redis_hits'],
                'semantic_hits': self.stats['semantic_hits'],
                'misses': self.stats['misses'],
                'sets': self.stats['sets'],
                'errors': self.stats['errors']
            },
            'backends': {
                'memory': self.memory_cache.get_stats(),
                'redis_available': self.redis_client is not None,
                'semantic': self.semantic_index.get_stats()
            },
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'memory_cache_size': self.config.memory_cache_size,
                'semantic_enabled': self.semantic_index._initialized
            }
        }
    
    async def cleanup(self):
        """Cleanup cache resources"""
        if self.redis_client:
            await self.redis_client.close()
        
        deallocate_memory('semantic_model')
        logging.info("Smart cache cleanup completed")

# Global cache instance
_smart_cache: Optional[SmartCache] = None

def get_smart_cache(config: CacheConfig = None) -> SmartCache:
    """Get or create the global smart cache instance"""
    global _smart_cache
    if _smart_cache is None:
        _smart_cache = SmartCache(config)
    return _smart_cache

async def initialize_smart_cache(config: CacheConfig = None):
    """Initialize the global smart cache"""
    cache = get_smart_cache(config)
    await cache.initialize()
    return cache
