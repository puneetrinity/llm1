# utils/connection_pool.py - Advanced Connection Management
import aiohttp
import asyncio
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pool optimization"""
    total_limit: int = 100
    per_host_limit: int = 20
    keepalive_timeout: int = 60
    connect_timeout: int = 10
    total_timeout: int = 300
    dns_cache_ttl: int = 300
    tcp_nodelay: bool = True
    tcp_keepalive: bool = True

class OptimizedConnectionPool:
    """High-performance connection pool with advanced features"""
    
    def __init__(self, config: ConnectionPoolConfig = None):
        self.config = config or ConnectionPoolConfig()
        self.connector: Optional[aiohttp.TCPConnector] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'dns_cache_hits': 0,
            'timeouts': 0,
            'errors': 0
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize the optimized connection pool"""
        if self._initialized:
            return
            
        # Create optimized TCP connector
        self.connector = aiohttp.TCPConnector(
            # Connection limits
            limit=self.config.total_limit,
            limit_per_host=self.config.per_host_limit,
            
            # Keepalive optimization
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True,
            
            # DNS optimization
            ttl_dns_cache=self.config.dns_cache_ttl,
            use_dns_cache=True,
            
            # TCP optimization
           # # # # tcp_nodelay=self.config.tcp_nodelay,  # Compatibility fix  # Compatibility fix  # Compatibility fix  # Compatibility fix
            
            # SSL optimization (if needed)
            ssl=False,  # Ollama typically uses HTTP
        )
        
        # Create optimized client session
        timeout = aiohttp.ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connect_timeout,
            sock_read=30,
            sock_connect=self.config.connect_timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={
                'Connection': 'keep-alive',
                'Keep-Alive': f'timeout={self.config.keepalive_timeout}',
                'User-Agent': 'LLM-Proxy-Enhanced/2.0'
            },
            # Enable compression
            auto_decompress=True,
            # Connection pooling headers
            trust_env=True
        )
        
        self._initialized = True
        logging.info(f"Optimized connection pool initialized: {self.config.total_limit} total, {self.config.per_host_limit} per host")
    
    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make an optimized HTTP request"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Track connection reuse
            connection_key = f"{url.split('//')[1].split('/')[0]}"
            
            async with self.session.request(method, url, **kwargs) as response:
                # Update statistics
                duration = time.time() - start_time
                
                if hasattr(response, 'connection') and response.connection:
                    if hasattr(response.connection, 'transport'):
                        # Connection was reused if it exists
                        self.stats['connections_reused'] += 1
                    else:
                        self.stats['connections_created'] += 1
                
                logging.debug(f"Request completed in {duration:.3f}s, status: {response.status}")
                return response
                
        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logging.warning(f"Request timeout after {time.time() - start_time:.3f}s")
            raise
        except Exception as e:
            self.stats['errors'] += 1
            logging.error(f"Request failed after {time.time() - start_time:.3f}s: {str(e)}")
            raise
    
    async def post_json(self, url: str, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Optimized JSON POST request"""
        kwargs.setdefault('json', data)
        kwargs.setdefault('headers', {}).update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        async with await self.request('POST', url, **kwargs) as response:
            if response.status != 200:
                error_text = await response.text()
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=error_text
                )
            return await response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        connector_stats = {}
        if self.connector:
            connector_stats = {
                'total_connections': len(self.connector._conns),
                'available_connections': sum(len(conns) for conns in self.connector._conns.values()),
                'acquired_connections': self.connector._acquired_per_host,
                'dns_cache_size': len(self.connector._dns_cache) if hasattr(self.connector, '_dns_cache') else 0
            }
        
        return {
            'pool_stats': self.stats,
            'connector_stats': connector_stats,
            'config': {
                'total_limit': self.config.total_limit,
                'per_host_limit': self.config.per_host_limit,
                'keepalive_timeout': self.config.keepalive_timeout
            },
            'initialized': self._initialized
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pool health and performance"""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        stats = self.get_stats()
        
        # Calculate efficiency metrics
        total_requests = self.stats['connections_created'] + self.stats['connections_reused']
        reuse_rate = (self.stats['connections_reused'] / max(1, total_requests)) * 100
        error_rate = (self.stats['errors'] / max(1, total_requests)) * 100
        
        health_status = 'healthy'
        if error_rate > 5:
            health_status = 'degraded'
        elif error_rate > 10:
            health_status = 'unhealthy'
        
        return {
            'status': health_status,
            'metrics': {
                'connection_reuse_rate': reuse_rate,
                'error_rate': error_rate,
                'total_requests': total_requests,
                'avg_connections_per_host': stats['connector_stats'].get('total_connections', 0) / max(1, len(self.connector._conns))
            },
            'recommendations': self._get_recommendations(reuse_rate, error_rate)
        }
    
    def _get_recommendations(self, reuse_rate: float, error_rate: float) -> list:
        """Get optimization recommendations"""
        recommendations = []
        
        if reuse_rate < 50:
            recommendations.append("Low connection reuse - consider increasing keepalive_timeout")
        
        if error_rate > 5:
            recommendations.append("High error rate - check network stability or increase timeout")
        
        if self.stats['timeouts'] > self.stats['connections_created'] * 0.1:
            recommendations.append("Frequent timeouts - consider increasing connect_timeout")
        
        return recommendations
    
    async def cleanup(self):
        """Clean up connection pool resources"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        if self.connector:
            await self.connector.close()
        
        self._initialized = False
        logging.info("Connection pool cleaned up")

# Global connection pool instance
_connection_pool: Optional[OptimizedConnectionPool] = None

def get_connection_pool(config: ConnectionPoolConfig = None) -> OptimizedConnectionPool:
    """Get or create the global connection pool"""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = OptimizedConnectionPool(config)
    
    return _connection_pool

async def initialize_connection_pool(config: ConnectionPoolConfig = None):
    """Initialize the global connection pool"""
    pool = get_connection_pool(config)
    await pool.initialize()
    return pool
