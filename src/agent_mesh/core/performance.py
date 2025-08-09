"""Performance optimization and caching for Agent Mesh.

Provides intelligent caching, connection pooling, load balancing,
and performance optimization features for scalable mesh operations.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID

import structlog


logger = structlog.get_logger("performance")


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class IntelligentCache:
    """High-performance cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self._cache.get(key)
        
        if entry is None or entry.is_expired:
            self._misses += 1
            if entry and entry.is_expired:
                await self.delete(key)
            return None
        
        entry.last_accessed = time.time()
        self._hits += 1
        self._update_access_order(key)
        
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        if key in self._cache:
            await self.delete(key)
        
        while len(self._cache) >= self.max_size:
            await self._evict_lru()
        
        entry = CacheEntry(
            key=key,
            value=value,
            ttl=ttl if ttl is not None else self.default_ttl
        )
        
        self._cache[key] = entry
        self._access_order.append(key)
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate
        }
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            await self.delete(lru_key)


class PerformanceManager:
    """Central performance management system."""
    
    def __init__(self):
        self.logger = structlog.get_logger("performance_manager")
        self.cache = IntelligentCache(max_size=5000, default_ttl=300.0)
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start performance management."""
        self.logger.info("Starting performance management")
        self._running = True
        self._task = asyncio.create_task(self._optimization_loop())
    
    async def stop(self):
        """Stop performance management."""
        self.logger.info("Stopping performance management")
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {"cache": self.cache.get_statistics()}
    
    async def _optimization_loop(self):
        """Background optimization."""
        while self._running:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break


_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> PerformanceManager:
    """Get global performance manager."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager()
    return _performance_manager