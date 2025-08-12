"""Advanced caching system with adaptive policies and distributed synchronization."""

import asyncio
import hashlib
import pickle
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from uuid import UUID, uuid4

import structlog


class CachePolicy(Enum):
    """Cache eviction policies."""
    
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on usage patterns


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    
    L1_MEMORY = "l1_memory"      # Fast in-memory cache
    L2_DISK = "l2_disk"          # Slower disk-based cache
    L3_DISTRIBUTED = "l3_distributed"  # Distributed cache across nodes


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class AdaptiveCacheStrategy:
    """Adaptive caching strategy that learns from usage patterns."""
    
    def __init__(self):
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.size_patterns: Dict[str, List[int]] = defaultdict(list)
        self.hit_rates: Dict[str, float] = defaultdict(float)
        self.adaptive_ttl: Dict[str, float] = {}
        
    def record_access(self, key: str, hit: bool, size: int = 0) -> None:
        """Record cache access for learning."""
        current_time = time.time()
        
        # Track access patterns
        self.access_patterns[key].append(current_time)
        if len(self.access_patterns[key]) > 100:  # Keep last 100 accesses
            self.access_patterns[key] = self.access_patterns[key][-100:]
        
        # Track size patterns
        if size > 0:
            self.size_patterns[key].append(size)
            if len(self.size_patterns[key]) > 50:
                self.size_patterns[key] = self.size_patterns[key][-50:]
        
        # Update hit rate
        pattern = self.access_patterns[key]
        if len(pattern) >= 10:
            recent_hits = sum(1 for _ in pattern[-10:])  # Simplified hit counting
            self.hit_rates[key] = recent_hits / 10
    
    def get_adaptive_ttl(self, key: str) -> float:
        """Calculate adaptive TTL based on access patterns."""
        if key not in self.access_patterns:
            return 300.0  # Default 5 minutes
        
        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 300.0
        
        # Calculate access frequency
        time_span = accesses[-1] - accesses[0]
        frequency = len(accesses) / max(time_span, 1)
        
        # Higher frequency = longer TTL
        adaptive_ttl = min(3600.0, max(60.0, 300.0 * frequency))
        self.adaptive_ttl[key] = adaptive_ttl
        
        return adaptive_ttl
    
    def should_cache(self, key: str, size: int) -> bool:
        """Determine if an item should be cached."""
        # Don't cache very large items
        if size > 10 * 1024 * 1024:  # 10MB
            return False
        
        # Cache if we've seen this key before and it has good hit rate
        if key in self.hit_rates and self.hit_rates[key] > 0.1:
            return True
        
        # Cache new items unless they're too big
        return size < 1024 * 1024  # 1MB


class MemoryCache:
    """High-performance in-memory cache with adaptive policies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        default_ttl: float = 300.0,
        policy: CachePolicy = CachePolicy.ADAPTIVE
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default time-to-live in seconds
            policy: Cache eviction policy
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.policy = policy
        
        self.logger = structlog.get_logger("memory_cache")
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory = 0
        
        # Adaptive strategy
        self._adaptive_strategy = AdaptiveCacheStrategy()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start background cache maintenance."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Memory cache started")
    
    async def stop(self) -> None:
        """Stop background cache maintenance."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Memory cache stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                self._adaptive_strategy.record_access(key, False)
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self._misses += 1
                self._adaptive_strategy.record_access(key, False)
                return None
            
            # Update access statistics
            entry.update_access()
            self._hits += 1
            self._adaptive_strategy.record_access(key, True, entry.size_bytes)
            
            # Move to end for LRU
            if self.policy in [CachePolicy.LRU, CachePolicy.ADAPTIVE]:
                self._cache.move_to_end(key)
            
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Put value in cache."""
        # Calculate size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = sys.getsizeof(value)
        
        # Check if we should cache this item
        if not self._adaptive_strategy.should_cache(key, size_bytes):
            return False
        
        with self._lock:
            # Determine TTL
            if ttl is None:
                if self.policy == CachePolicy.ADAPTIVE:
                    ttl = self._adaptive_strategy.get_adaptive_ttl(key)
                else:
                    ttl = self.default_ttl
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                size_bytes=size_bytes,
                tags=tags or []
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Ensure we have space
            self._ensure_space(size_bytes)
            
            # Add entry
            self._cache[key] = entry
            self._current_memory += size_bytes
            
            self.logger.debug("Cache entry added",
                            key=key,
                            size_bytes=size_bytes,
                            ttl=ttl)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self.logger.info("Cache cleared")
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate cache entries by tags."""
        removed_count = 0
        
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self._remove_entry(key)
                removed_count += 1
        
        self.logger.info("Cache invalidated by tags",
                        tags=tags,
                        removed_count=removed_count)
        
        return removed_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "entries": len(self._cache),
                "max_size": self.max_size,
                "memory_usage_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "policy": self.policy.value
            }
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
    
    def _ensure_space(self, required_bytes: int) -> None:
        """Ensure enough space for new entry."""
        # Check size limit
        while len(self._cache) >= self.max_size:
            self._evict_one()
        
        # Check memory limit
        while self._current_memory + required_bytes > self.max_memory_bytes:
            if not self._evict_one():
                break  # No more entries to evict
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy."""
        if not self._cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            key = next(iter(self._cache))
        elif self.policy == CachePolicy.LFU:
            key = min(self._cache.keys(), 
                     key=lambda k: self._cache[k].access_count)
        elif self.policy == CachePolicy.TTL:
            key = min(self._cache.keys(),
                     key=lambda k: self._cache[k].created_at + (self._cache[k].ttl or 0))
        else:  # ADAPTIVE
            # Use a combination of LRU and access frequency
            key = min(self._cache.keys(),
                     key=lambda k: (
                         self._cache[k].access_count * 0.3 +
                         (time.time() - self._cache[k].last_accessed) * 0.7
                     ))
        
        self._remove_entry(key)
        self._evictions += 1
        return True
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for expired entries."""
        while self._running:
            try:
                with self._lock:
                    expired_keys = []
                    for key, entry in self._cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    if expired_keys:
                        self.logger.debug("Expired entries cleaned up",
                                        count=len(expired_keys))
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache cleanup error", error=str(e))
                await asyncio.sleep(60)


class DistributedCache:
    """Distributed cache with peer-to-peer synchronization."""
    
    def __init__(
        self,
        node_id: UUID,
        local_cache: MemoryCache,
        network_manager: Optional[Any] = None
    ):
        """
        Initialize distributed cache.
        
        Args:
            node_id: Unique node identifier
            local_cache: Local memory cache instance
            network_manager: Network manager for peer communication
        """
        self.node_id = node_id
        self.local_cache = local_cache
        self.network_manager = network_manager
        
        self.logger = structlog.get_logger("distributed_cache", 
                                         node_id=str(node_id))
        
        # Peer caches tracking
        self.peer_caches: Dict[UUID, Dict[str, float]] = {}  # peer_id -> {key -> timestamp}
        self.replication_factor = 3
        
        # Synchronization state
        self._sync_interval = 30.0
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start distributed cache."""
        if self._running:
            return
        
        self._running = True
        await self.local_cache.start()
        
        self._sync_task = asyncio.create_task(self._synchronization_loop())
        self.logger.info("Distributed cache started")
    
    async def stop(self) -> None:
        """Stop distributed cache."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        await self.local_cache.stop()
        self.logger.info("Distributed cache stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        # Try local cache first
        value = self.local_cache.get(key)
        if value is not None:
            return value
        
        # Try to fetch from peers
        if self.network_manager:
            value = await self._fetch_from_peers(key)
            if value is not None:
                # Cache locally for future access
                self.local_cache.put(key, value)
                return value
        
        return None
    
    async def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        replicate: bool = True
    ) -> bool:
        """Put value in distributed cache."""
        # Store locally
        success = self.local_cache.put(key, value, ttl)
        
        if success and replicate and self.network_manager:
            # Replicate to peers
            await self._replicate_to_peers(key, value, ttl)
        
        return success
    
    async def delete(self, key: str, propagate: bool = True) -> bool:
        """Delete from distributed cache."""
        success = self.local_cache.delete(key)
        
        if success and propagate and self.network_manager:
            await self._propagate_deletion(key)
        
        return success
    
    async def _fetch_from_peers(self, key: str) -> Optional[Any]:
        """Fetch value from peer caches."""
        if not self.network_manager:
            return None
        
        # Find peers that might have this key
        candidate_peers = []
        for peer_id, cache_keys in self.peer_caches.items():
            if key in cache_keys:
                candidate_peers.append(peer_id)
        
        # Try to fetch from candidates
        for peer_id in candidate_peers:
            try:
                value = await self.network_manager.request_cache_value(peer_id, key)
                if value is not None:
                    self.logger.debug("Cache value fetched from peer",
                                    key=key, peer_id=str(peer_id))
                    return value
            except Exception as e:
                self.logger.warning("Failed to fetch from peer",
                                  key=key, peer_id=str(peer_id), error=str(e))
        
        return None
    
    async def _replicate_to_peers(
        self,
        key: str,
        value: Any,
        ttl: Optional[float]
    ) -> None:
        """Replicate value to peer nodes."""
        if not self.network_manager:
            return
        
        # Select peers for replication
        peers = await self.network_manager.get_connected_peers()
        if len(peers) < self.replication_factor:
            selected_peers = peers
        else:
            # Use consistent hashing to select peers
            selected_peers = self._select_peers_for_key(key, peers)
        
        # Send replication requests
        replication_tasks = []
        for peer in selected_peers[:self.replication_factor]:
            task = asyncio.create_task(
                self.network_manager.replicate_cache_entry(
                    peer.peer_id, key, value, ttl
                )
            )
            replication_tasks.append(task)
        
        # Wait for replication (don't block on failures)
        if replication_tasks:
            await asyncio.gather(*replication_tasks, return_exceptions=True)
    
    async def _propagate_deletion(self, key: str) -> None:
        """Propagate deletion to peer nodes."""
        if not self.network_manager:
            return
        
        peers = await self.network_manager.get_connected_peers()
        
        deletion_tasks = []
        for peer in peers:
            task = asyncio.create_task(
                self.network_manager.delete_cache_entry(peer.peer_id, key)
            )
            deletion_tasks.append(task)
        
        if deletion_tasks:
            await asyncio.gather(*deletion_tasks, return_exceptions=True)
    
    def _select_peers_for_key(self, key: str, peers: List[Any]) -> List[Any]:
        """Select peers for key using consistent hashing."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Sort peers by hash distance to key
        peer_distances = []
        for peer in peers:
            peer_hash = hashlib.sha256(str(peer.peer_id).encode()).hexdigest()
            distance = int(key_hash, 16) ^ int(peer_hash, 16)
            peer_distances.append((distance, peer))
        
        peer_distances.sort(key=lambda x: x[0])
        return [peer for _, peer in peer_distances]
    
    async def _synchronization_loop(self) -> None:
        """Background synchronization with peers."""
        while self._running:
            try:
                if self.network_manager:
                    # Exchange cache metadata with peers
                    await self._exchange_cache_metadata()
                
                await asyncio.sleep(self._sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache synchronization error", error=str(e))
                await asyncio.sleep(self._sync_interval)
    
    async def _exchange_cache_metadata(self) -> None:
        """Exchange cache metadata with peers."""
        peers = await self.network_manager.get_connected_peers()
        
        # Prepare local cache metadata
        local_metadata = {}
        for key, entry in self.local_cache._cache.items():
            local_metadata[key] = entry.last_accessed
        
        # Exchange with peers
        for peer in peers:
            try:
                peer_metadata = await self.network_manager.exchange_cache_metadata(
                    peer.peer_id, local_metadata
                )
                if peer_metadata:
                    self.peer_caches[peer.peer_id] = peer_metadata
            except Exception as e:
                self.logger.warning("Metadata exchange failed",
                                  peer_id=str(peer.peer_id), error=str(e))


class CacheManager:
    """High-level cache manager coordinating multiple cache levels."""
    
    def __init__(
        self,
        node_id: UUID,
        network_manager: Optional[Any] = None
    ):
        """Initialize cache manager."""
        self.node_id = node_id
        self.logger = structlog.get_logger("cache_manager", node_id=str(node_id))
        
        # Cache levels
        self.l1_cache = MemoryCache(
            max_size=10000,
            max_memory_mb=512,
            policy=CachePolicy.ADAPTIVE
        )
        
        self.distributed_cache = DistributedCache(
            node_id=node_id,
            local_cache=self.l1_cache,
            network_manager=network_manager
        )
        
        # Cache hierarchies for different data types
        self.model_cache = MemoryCache(
            max_size=100,
            max_memory_mb=1024,  # 1GB for models
            policy=CachePolicy.LFU
        )
        
        self.data_cache = MemoryCache(
            max_size=5000,
            max_memory_mb=256,
            policy=CachePolicy.LRU
        )
    
    async def start(self) -> None:
        """Start cache manager."""
        await self.distributed_cache.start()
        await self.model_cache.start()
        await self.data_cache.start()
        
        self.logger.info("Cache manager started")
    
    async def stop(self) -> None:
        """Stop cache manager."""
        await self.distributed_cache.stop()
        await self.model_cache.stop()
        await self.data_cache.stop()
        
        self.logger.info("Cache manager stopped")
    
    async def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """Get value from appropriate cache."""
        cache = self._get_cache_by_type(cache_type)
        
        if cache == self.distributed_cache:
            return await cache.get(key)
        else:
            return cache.get(key)
    
    async def put(
        self,
        key: str,
        value: Any,
        cache_type: str = "general",
        ttl: Optional[float] = None,
        replicate: bool = True
    ) -> bool:
        """Put value in appropriate cache."""
        cache = self._get_cache_by_type(cache_type)
        
        if cache == self.distributed_cache:
            return await cache.put(key, value, ttl, replicate)
        else:
            return cache.put(key, value, ttl)
    
    async def delete(self, key: str, cache_type: str = "general") -> bool:
        """Delete from appropriate cache."""
        cache = self._get_cache_by_type(cache_type)
        
        if cache == self.distributed_cache:
            return await cache.delete(key)
        else:
            return cache.delete(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "l1_cache": self.l1_cache.get_statistics(),
            "model_cache": self.model_cache.get_statistics(),
            "data_cache": self.data_cache.get_statistics(),
            "total_memory_mb": (
                self.l1_cache.get_statistics()["memory_usage_mb"] +
                self.model_cache.get_statistics()["memory_usage_mb"] +
                self.data_cache.get_statistics()["memory_usage_mb"]
            )
        }
    
    def _get_cache_by_type(self, cache_type: str) -> Union[MemoryCache, DistributedCache]:
        """Get cache instance by type."""
        if cache_type == "model":
            return self.model_cache
        elif cache_type == "data":
            return self.data_cache
        else:
            return self.distributed_cache


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(node_id: Optional[UUID] = None) -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(node_id or uuid4())
    return _cache_manager


async def initialize_caching(node_id: UUID, network_manager: Optional[Any] = None) -> CacheManager:
    """Initialize global caching system."""
    global _cache_manager
    _cache_manager = CacheManager(node_id, network_manager)
    await _cache_manager.start()
    return _cache_manager