"""Performance optimization and scaling system.

This module provides advanced performance optimization including:
- Adaptive caching with intelligent eviction
- Load balancing and auto-scaling
- Resource optimization and pooling
- Performance profiling and tuning
"""

import asyncio
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from uuid import UUID, uuid4
from collections import defaultdict, OrderedDict

import structlog


class CachePolicy(Enum):
    """Cache eviction policies."""
    
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ResourcePool:
    """Resource pool for connection/object pooling."""
    
    pool_name: str
    max_size: int = 100
    min_size: int = 10
    idle_timeout: float = 300.0  # 5 minutes
    
    # Pool state
    active_resources: Set[Any] = field(default_factory=set)
    idle_resources: List[Tuple[Any, float]] = field(default_factory=list)  # (resource, idle_since)
    create_resource_func: Optional[Callable] = None
    destroy_resource_func: Optional[Callable] = None
    
    def __post_init__(self):
        self._lock = asyncio.Lock()


class AdaptiveCache:
    """High-performance adaptive cache with multiple eviction policies."""
    
    def __init__(
        self, 
        max_size_mb: float = 100.0,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        default_ttl: Optional[float] = None
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.policy = policy
        self.default_ttl = default_ttl
        
        self.logger = structlog.get_logger("adaptive_cache")
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}
        self.access_order: OrderedDict[str, float] = OrderedDict()  # For LRU
        self.access_frequency: Dict[str, int] = defaultdict(int)  # For LFU
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size_bytes = 0
        
        # Adaptive policy state
        self.policy_performance: Dict[CachePolicy, float] = {
            CachePolicy.LRU: 1.0,
            CachePolicy.LFU: 1.0,
            CachePolicy.TTL: 1.0
        }
        self.current_adaptive_policy = CachePolicy.LRU
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start cache background tasks."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop cache background tasks."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        entry = self.entries.get(key)
        
        if entry is None:
            self.misses += 1
            return None
            
        # Check expiration
        if entry.is_expired():
            await self._remove_entry(key)
            self.misses += 1
            return None
            
        # Update access statistics
        entry.access_count += 1
        entry.last_accessed = time.time()
        self.access_order[key] = entry.last_accessed
        self.access_frequency[key] += 1
        
        self.hits += 1
        return entry.value
        
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[float] = None,
        size_hint: Optional[int] = None
    ):
        """Set value in cache."""
        # Calculate size
        if size_hint is not None:
            size_bytes = size_hint
        else:
            size_bytes = self._estimate_size(value)
            
        # Check if we need to evict entries
        await self._ensure_capacity(size_bytes)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds or self.default_ttl
        )
        
        # Remove existing entry if present
        if key in self.entries:
            await self._remove_entry(key)
            
        # Add new entry
        self.entries[key] = entry
        self.access_order[key] = entry.last_accessed
        self.access_frequency[key] = 1
        self.current_size_bytes += size_bytes
        
        self.logger.debug("Cache entry added", 
                         key=key, size=size_bytes, ttl=ttl_seconds)
        
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        if key in self.entries:
            await self._remove_entry(key)
            return True
        return False
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "entries": len(self.entries),
            "size_bytes": self.current_size_bytes,
            "size_mb": self.current_size_bytes / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization": self.current_size_bytes / self.max_size_bytes
        }
        
    async def _ensure_capacity(self, needed_bytes: int):
        """Ensure cache has capacity for new entry."""
        while (self.current_size_bytes + needed_bytes > self.max_size_bytes and 
               self.entries):
            
            if self.policy == CachePolicy.ADAPTIVE:
                await self._adaptive_evict()
            elif self.policy == CachePolicy.LRU:
                await self._lru_evict()
            elif self.policy == CachePolicy.LFU:
                await self._lfu_evict()
            elif self.policy == CachePolicy.FIFO:
                await self._fifo_evict()
            elif self.policy == CachePolicy.TTL:
                await self._ttl_evict()
                
    async def _adaptive_evict(self):
        """Adaptive eviction based on policy performance."""
        # Track performance of different policies
        start_time = time.time()
        
        # Try current adaptive policy
        if self.current_adaptive_policy == CachePolicy.LRU:
            await self._lru_evict()
        elif self.current_adaptive_policy == CachePolicy.LFU:
            await self._lfu_evict()
        else:
            await self._ttl_evict()
            
        # Update policy performance (simplified)
        eviction_time = time.time() - start_time
        self.policy_performance[self.current_adaptive_policy] *= 0.9
        self.policy_performance[self.current_adaptive_policy] += 0.1 * (1.0 / max(eviction_time, 0.001))
        
        # Occasionally switch to best performing policy
        if self.evictions % 100 == 0:
            best_policy = max(self.policy_performance.items(), key=lambda x: x[1])[0]
            if best_policy != self.current_adaptive_policy:
                self.current_adaptive_policy = best_policy
                self.logger.debug("Switched adaptive policy", new_policy=best_policy.value)
                
    async def _lru_evict(self):
        """Evict least recently used entry."""
        if not self.access_order:
            return
            
        # Find LRU entry
        oldest_key = next(iter(self.access_order))
        await self._remove_entry(oldest_key)
        
    async def _lfu_evict(self):
        """Evict least frequently used entry."""
        if not self.access_frequency:
            return
            
        # Find LFU entry
        lfu_key = min(self.access_frequency.items(), key=lambda x: x[1])[0]
        await self._remove_entry(lfu_key)
        
    async def _fifo_evict(self):
        """Evict first in, first out."""
        if not self.entries:
            return
            
        # Find oldest entry by creation time
        oldest_entry = min(self.entries.items(), key=lambda x: x[1].created_at)
        await self._remove_entry(oldest_entry[0])
        
    async def _ttl_evict(self):
        """Evict expired entries first, then oldest."""
        # First try to evict expired entries
        expired_keys = [
            key for key, entry in self.entries.items()
            if entry.is_expired()
        ]
        
        if expired_keys:
            await self._remove_entry(expired_keys[0])
            return
            
        # Fall back to oldest entry
        await self._fifo_evict()
        
    async def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key not in self.entries:
            return
            
        entry = self.entries[key]
        self.current_size_bytes -= entry.size_bytes
        
        del self.entries[key]
        self.access_order.pop(key, None)
        self.access_frequency.pop(key, None)
        
        self.evictions += 1
        
    async def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.entries.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                        
                for key in expired_keys:
                    await self._remove_entry(key)
                    
                if expired_keys:
                    self.logger.debug("Cleaned up expired entries", count=len(expired_keys))
                    
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache cleanup error", error=str(e))
                await asyncio.sleep(60)
                
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return len(str(value)) * 2  # Rough estimate
            else:
                return 1000  # Default estimate


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.PERFORMANCE_BASED):
        self.strategy = strategy
        self.logger = structlog.get_logger("load_balancer")
        
        # Node tracking
        self.nodes: Dict[UUID, Dict[str, Any]] = {}
        self.node_performance: Dict[UUID, PerformanceMetrics] = {}
        self.connection_counts: Dict[UUID, int] = defaultdict(int)
        
        # Round robin state
        self.round_robin_index = 0
        
        # Consistent hashing state
        self.hash_ring: List[Tuple[int, UUID]] = []
        
    def register_node(self, node_id: UUID, weight: float = 1.0, metadata: Optional[Dict] = None):
        """Register a node for load balancing."""
        self.nodes[node_id] = {
            "weight": weight,
            "metadata": metadata or {},
            "registered_at": time.time(),
            "active": True
        }
        
        # Initialize performance metrics
        self.node_performance[node_id] = PerformanceMetrics()
        self.connection_counts[node_id] = 0
        
        # Update consistent hash ring
        if self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            self._rebuild_hash_ring()
            
        self.logger.info("Node registered", node_id=str(node_id), weight=weight)
        
    def unregister_node(self, node_id: UUID):
        """Unregister a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.node_performance[node_id]
            del self.connection_counts[node_id]
            
            if self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
                self._rebuild_hash_ring()
                
            self.logger.info("Node unregistered", node_id=str(node_id))
            
    def update_node_performance(self, node_id: UUID, metrics: PerformanceMetrics):
        """Update performance metrics for a node."""
        if node_id in self.node_performance:
            self.node_performance[node_id] = metrics
            
    def select_node(self, key: Optional[str] = None) -> Optional[UUID]:
        """Select best node based on strategy."""
        active_nodes = [nid for nid, info in self.nodes.items() if info["active"]]
        
        if not active_nodes:
            return None
            
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(active_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(active_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_RANDOM:
            return self._weighted_random_select(active_nodes)
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(key or "", active_nodes)
        elif self.strategy == LoadBalanceStrategy.PERFORMANCE_BASED:
            return self._performance_based_select(active_nodes)
        else:
            return active_nodes[0]  # Default to first node
            
    def record_connection(self, node_id: UUID):
        """Record new connection to node."""
        self.connection_counts[node_id] += 1
        
    def record_disconnection(self, node_id: UUID):
        """Record disconnection from node."""
        self.connection_counts[node_id] = max(0, self.connection_counts[node_id] - 1)
        
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        total_connections = sum(self.connection_counts.values())
        
        node_stats = {}
        for node_id in self.nodes:
            connections = self.connection_counts[node_id]
            metrics = self.node_performance.get(node_id, PerformanceMetrics())
            
            node_stats[str(node_id)] = {
                "connections": connections,
                "load_percentage": (connections / total_connections * 100) if total_connections > 0 else 0,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "throughput": metrics.throughput_ops_sec,
                "latency_ms": metrics.network_latency_ms
            }
            
        return {
            "total_nodes": len(self.nodes),
            "total_connections": total_connections,
            "strategy": self.strategy.value,
            "node_statistics": node_stats
        }
        
    def _round_robin_select(self, nodes: List[UUID]) -> UUID:
        """Round robin selection."""
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected
        
    def _least_connections_select(self, nodes: List[UUID]) -> UUID:
        """Select node with least connections."""
        return min(nodes, key=lambda nid: self.connection_counts[nid])
        
    def _weighted_random_select(self, nodes: List[UUID]) -> UUID:
        """Weighted random selection based on node weights."""
        import random
        
        weights = [self.nodes[nid]["weight"] for nid in nodes]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(nodes)
            
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return nodes[i]
                
        return nodes[-1]
        
    def _consistent_hash_select(self, key: str, nodes: List[UUID]) -> UUID:
        """Consistent hash selection."""
        if not self.hash_ring:
            self._rebuild_hash_ring()
            
        # Hash the key
        key_hash = hash(key)
        
        # Find position in ring
        for ring_hash, node_id in self.hash_ring:
            if key_hash <= ring_hash and node_id in nodes:
                return node_id
                
        # Wrap around to first node
        for ring_hash, node_id in self.hash_ring:
            if node_id in nodes:
                return node_id
                
        return nodes[0]
        
    def _performance_based_select(self, nodes: List[UUID]) -> UUID:
        """Select node based on performance metrics."""
        # Calculate performance scores
        scores = {}
        
        for node_id in nodes:
            metrics = self.node_performance.get(node_id, PerformanceMetrics())
            connections = self.connection_counts[node_id]
            
            # Lower is better for these metrics
            cpu_score = max(0, 100 - metrics.cpu_usage) / 100
            memory_score = max(0, 100 - metrics.memory_usage) / 100
            latency_score = max(0, 1000 - metrics.network_latency_ms) / 1000
            connection_score = max(0, 100 - connections) / 100
            
            # Higher is better for throughput
            throughput_score = min(1.0, metrics.throughput_ops_sec / 1000)
            
            # Combined score
            scores[node_id] = (
                cpu_score * 0.25 +
                memory_score * 0.25 +
                latency_score * 0.2 +
                connection_score * 0.2 +
                throughput_score * 0.1
            )
            
        # Return node with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
        
    def _rebuild_hash_ring(self):
        """Rebuild consistent hash ring."""
        self.hash_ring = []
        
        for node_id in self.nodes:
            # Add multiple points for each node for better distribution
            for i in range(100):
                point_hash = hash(f"{node_id}_{i}")
                self.hash_ring.append((point_hash, node_id))
                
        self.hash_ring.sort(key=lambda x: x[0])


class ResourcePoolManager:
    """Manages resource pools for connection and object pooling."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.logger = structlog.get_logger("resource_pool_manager")
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start pool management."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop pool management."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Clean up all pools
        for pool in self.pools.values():
            await self._cleanup_pool(pool)
            
    def create_pool(
        self,
        pool_name: str,
        create_func: Callable,
        destroy_func: Optional[Callable] = None,
        max_size: int = 100,
        min_size: int = 10,
        idle_timeout: float = 300.0
    ) -> ResourcePool:
        """Create a new resource pool."""
        pool = ResourcePool(
            pool_name=pool_name,
            max_size=max_size,
            min_size=min_size,
            idle_timeout=idle_timeout
        )
        
        pool.create_resource_func = create_func
        pool.destroy_resource_func = destroy_func
        
        self.pools[pool_name] = pool
        
        self.logger.info("Resource pool created", 
                        pool_name=pool_name,
                        max_size=max_size,
                        min_size=min_size)
        
        return pool
        
    async def acquire_resource(self, pool_name: str) -> Optional[Any]:
        """Acquire resource from pool."""
        pool = self.pools.get(pool_name)
        if not pool:
            return None
            
        async with pool._lock:
            # Try to get idle resource
            current_time = time.time()
            
            while pool.idle_resources:
                resource, idle_since = pool.idle_resources.pop(0)
                
                # Check if resource is still valid (not timed out)
                if current_time - idle_since < pool.idle_timeout:
                    pool.active_resources.add(resource)
                    return resource
                else:
                    # Resource timed out, destroy it
                    if pool.destroy_resource_func:
                        try:
                            await pool.destroy_resource_func(resource)
                        except Exception as e:
                            self.logger.warning("Failed to destroy timed out resource", 
                                               error=str(e))
                            
            # No idle resources available, create new one if under limit
            if len(pool.active_resources) < pool.max_size:
                if pool.create_resource_func:
                    try:
                        resource = await pool.create_resource_func()
                        pool.active_resources.add(resource)
                        return resource
                    except Exception as e:
                        self.logger.error("Failed to create resource", error=str(e))
                        
            return None  # Pool exhausted
            
    async def release_resource(self, pool_name: str, resource: Any):
        """Release resource back to pool."""
        pool = self.pools.get(pool_name)
        if not pool or resource not in pool.active_resources:
            return
            
        async with pool._lock:
            pool.active_resources.remove(resource)
            
            # Add to idle resources if under limit
            if len(pool.idle_resources) < pool.max_size - pool.min_size:
                pool.idle_resources.append((resource, time.time()))
            else:
                # Pool is full, destroy resource
                if pool.destroy_resource_func:
                    try:
                        await pool.destroy_resource_func(resource)
                    except Exception as e:
                        self.logger.warning("Failed to destroy excess resource", 
                                           error=str(e))
                        
    def get_pool_statistics(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a resource pool."""
        pool = self.pools.get(pool_name)
        if not pool:
            return None
            
        return {
            "pool_name": pool_name,
            "max_size": pool.max_size,
            "min_size": pool.min_size,
            "active_resources": len(pool.active_resources),
            "idle_resources": len(pool.idle_resources),
            "total_resources": len(pool.active_resources) + len(pool.idle_resources),
            "utilization": len(pool.active_resources) / pool.max_size
        }
        
    async def _cleanup_loop(self):
        """Background cleanup of idle resources."""
        while self._running:
            try:
                for pool in self.pools.values():
                    await self._cleanup_pool_idle_resources(pool)
                    
                await asyncio.sleep(60)  # Cleanup every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Pool cleanup error", error=str(e))
                await asyncio.sleep(60)
                
    async def _cleanup_pool_idle_resources(self, pool: ResourcePool):
        """Clean up timed out idle resources in a pool."""
        async with pool._lock:
            current_time = time.time()
            valid_resources = []
            destroyed_count = 0
            
            for resource, idle_since in pool.idle_resources:
                if current_time - idle_since < pool.idle_timeout:
                    valid_resources.append((resource, idle_since))
                else:
                    # Resource timed out, destroy it
                    if pool.destroy_resource_func:
                        try:
                            await pool.destroy_resource_func(resource)
                            destroyed_count += 1
                        except Exception as e:
                            self.logger.warning("Failed to destroy idle resource", 
                                               error=str(e))
                            
            pool.idle_resources = valid_resources
            
            if destroyed_count > 0:
                self.logger.debug("Cleaned up idle resources", 
                                 pool_name=pool.pool_name,
                                 destroyed_count=destroyed_count)
                                 
    async def _cleanup_pool(self, pool: ResourcePool):
        """Clean up entire pool."""
        async with pool._lock:
            # Destroy all active resources
            for resource in list(pool.active_resources):
                if pool.destroy_resource_func:
                    try:
                        await pool.destroy_resource_func(resource)
                    except Exception as e:
                        self.logger.warning("Failed to destroy active resource", 
                                           error=str(e))
                        
            # Destroy all idle resources
            for resource, _ in pool.idle_resources:
                if pool.destroy_resource_func:
                    try:
                        await pool.destroy_resource_func(resource)
                    except Exception as e:
                        self.logger.warning("Failed to destroy idle resource", 
                                           error=str(e))
                        
            pool.active_resources.clear()
            pool.idle_resources.clear()


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, node_id: UUID):
        self.node_id = node_id
        self.logger = structlog.get_logger("performance_optimizer", node_id=str(node_id))
        
        # Components
        self.cache = AdaptiveCache(max_size_mb=100.0)
        self.load_balancer = LoadBalancer()
        self.resource_pool_manager = ResourcePoolManager()
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_suggestions: List[Dict[str, Any]] = []
        
    async def start(self):
        """Start performance optimization."""
        self.logger.info("Starting performance optimizer")
        
        await self.cache.start()
        await self.resource_pool_manager.start()
        
        self.logger.info("Performance optimizer started")
        
    async def stop(self):
        """Stop performance optimization."""
        self.logger.info("Stopping performance optimizer")
        
        await self.resource_pool_manager.stop()
        await self.cache.stop()
        
        self.logger.info("Performance optimizer stopped")
        
    def analyze_performance(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyze performance and provide optimization suggestions."""
        suggestions = []
        
        # Cache analysis
        cache_stats = self.cache.get_statistics()
        if cache_stats["hit_rate"] < 0.7:
            suggestions.append("Consider increasing cache size or TTL values")
            
        if cache_stats["utilization"] > 0.9:
            suggestions.append("Cache is near capacity, consider increasing size")
            
        # Resource usage analysis
        if metrics.cpu_usage > 80:
            suggestions.append("High CPU usage detected, consider scaling out")
            
        if metrics.memory_usage > 85:
            suggestions.append("High memory usage, consider memory optimization")
            
        if metrics.network_latency_ms > 100:
            suggestions.append("High network latency, check network configuration")
            
        if metrics.error_rate > 0.05:
            suggestions.append("High error rate, investigate error sources")
            
        return suggestions
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive performance optimization report."""
        cache_stats = self.cache.get_statistics()
        load_stats = self.load_balancer.get_load_statistics()
        
        # Get pool statistics
        pool_stats = {}
        for pool_name in self.resource_pool_manager.pools:
            pool_stats[pool_name] = self.resource_pool_manager.get_pool_statistics(pool_name)
            
        return {
            "timestamp": time.time(),
            "node_id": str(self.node_id),
            "cache": cache_stats,
            "load_balancing": load_stats,
            "resource_pools": pool_stats,
            "optimization_suggestions": self.optimization_suggestions[-10:],  # Last 10 suggestions
            "performance_trend": self._calculate_performance_trend()
        }
        
    def _calculate_performance_trend(self) -> Dict[str, str]:
        """Calculate performance trends."""
        if len(self.performance_history) < 2:
            return {"trend": "insufficient_data"}
            
        recent = self.performance_history[-5:]  # Last 5 measurements
        older = self.performance_history[-10:-5] if len(self.performance_history) >= 10 else recent
        
        def avg_metric(metrics_list, attr):
            return sum(getattr(m, attr) for m in metrics_list) / len(metrics_list)
            
        trends = {}
        
        # CPU trend
        recent_cpu = avg_metric(recent, 'cpu_usage')
        older_cpu = avg_metric(older, 'cpu_usage')
        trends["cpu"] = "improving" if recent_cpu < older_cpu else "degrading" if recent_cpu > older_cpu else "stable"
        
        # Throughput trend
        recent_throughput = avg_metric(recent, 'throughput_ops_sec')
        older_throughput = avg_metric(older, 'throughput_ops_sec')
        trends["throughput"] = "improving" if recent_throughput > older_throughput else "degrading" if recent_throughput < older_throughput else "stable"
        
        return trends