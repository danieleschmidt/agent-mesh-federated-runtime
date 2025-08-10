"""Performance optimization and caching for Agent Mesh.

Provides intelligent caching, connection pooling, load balancing,
and performance optimization features for scalable mesh operations.
"""

import asyncio
import time
import threading
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from uuid import UUID
import statistics
import psutil

import structlog


logger = structlog.get_logger("performance")


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"
    TTL = "ttl"
    LFU = "lfu"  # Least Frequently Used
    ARC = "arc"  # Adaptive Replacement Cache


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    latency_samples: List[float] = field(default_factory=list)
    throughput_samples: List[float] = field(default_factory=list)
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: float = 0.0
    last_update: float = field(default_factory=time.time)
    
    def add_latency_sample(self, latency: float) -> None:
        """Add latency sample."""
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 1000:  # Keep only recent samples
            self.latency_samples = self.latency_samples[-1000:]
    
    def add_throughput_sample(self, throughput: float) -> None:
        """Add throughput sample."""
        self.throughput_samples.append(throughput)
        if len(self.throughput_samples) > 1000:
            self.throughput_samples = self.throughput_samples[-1000:]
    
    def get_average_latency(self) -> float:
        """Get average latency."""
        return statistics.mean(self.latency_samples) if self.latency_samples else 0.0
    
    def get_average_throughput(self) -> float:
        """Get average throughput."""
        return statistics.mean(self.throughput_samples) if self.throughput_samples else 0.0
    
    def get_p99_latency(self) -> float:
        """Get 99th percentile latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        index = int(0.99 * len(sorted_samples))
        return sorted_samples[min(index, len(sorted_samples) - 1)]


@dataclass
class CacheEntry:
    """Enhanced cache entry with comprehensive metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size_estimate: int = 0
    cache_tier: int = 0  # 0 = hot, 1 = warm, 2 = cold
    
    def __post_init__(self):
        """Calculate size estimate after initialization."""
        if self.size_estimate == 0:
            self.size_estimate = self._estimate_size(self.value)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get time since last access."""
        return time.time() - self.last_accessed
    
    def access(self) -> None:
        """Record an access to this entry."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def _estimate_size(self, obj) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                return 64  # Default estimate
        except:
            return 64


class ConnectionPool:
    """Advanced connection pooling with load balancing."""
    
    def __init__(self, max_connections: int = 50, max_idle: int = 10):
        self.max_connections = max_connections
        self.max_idle = max_idle
        self._pools: Dict[str, deque] = defaultdict(deque)
        self._active_connections: Dict[str, Set] = defaultdict(set)
        self._connection_factory: Dict[str, Callable] = {}
        self._pool_lock = asyncio.Lock()
        self._metrics = PerformanceMetrics()
        self.logger = structlog.get_logger("connection_pool")
    
    def register_factory(self, pool_name: str, factory: Callable) -> None:
        """Register connection factory for a pool."""
        self._connection_factory[pool_name] = factory
    
    async def acquire(self, pool_name: str, timeout: float = 30.0) -> Any:
        """Acquire connection from pool."""
        async with self._pool_lock:
            pool = self._pools[pool_name]
            
            # Try to get existing idle connection
            while pool:
                conn = pool.popleft()
                if await self._is_connection_healthy(conn):
                    self._active_connections[pool_name].add(conn)
                    return conn
            
            # Create new connection if under limit
            active_count = len(self._active_connections[pool_name])
            if active_count < self.max_connections:
                if pool_name in self._connection_factory:
                    conn = await self._connection_factory[pool_name]()
                    self._active_connections[pool_name].add(conn)
                    return conn
            
            # Wait for available connection
            start_time = time.time()
            while time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
                if pool:
                    conn = pool.popleft()
                    if await self._is_connection_healthy(conn):
                        self._active_connections[pool_name].add(conn)
                        return conn
        
        raise TimeoutError(f"Could not acquire connection from pool {pool_name}")
    
    async def release(self, pool_name: str, connection: Any) -> None:
        """Release connection back to pool."""
        async with self._pool_lock:
            active_set = self._active_connections[pool_name]
            if connection in active_set:
                active_set.remove(connection)
                
                # Add to idle pool if under limit
                pool = self._pools[pool_name]
                if len(pool) < self.max_idle:
                    pool.append(connection)
                else:
                    # Close excess connection
                    await self._close_connection(connection)
    
    async def _is_connection_healthy(self, connection: Any) -> bool:
        """Check if connection is healthy."""
        try:
            # Basic health check - can be overridden
            return hasattr(connection, 'is_connected') and connection.is_connected()
        except:
            return False
    
    async def _close_connection(self, connection: Any) -> None:
        """Close a connection."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
        except:
            pass


class IntelligentCache:
    """Advanced high-performance cache with multiple eviction policies."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, 
                 policy: CachePolicy = CachePolicy.LRU, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._frequency_count: Dict[str, int] = defaultdict(int)  # For LFU
        self._current_memory = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = asyncio.Lock()
        self.logger = structlog.get_logger("intelligent_cache")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired:
                self._misses += 1
                if entry and entry.is_expired:
                    await self._delete_internal(key)
                return None
            
            entry.access()
            self._hits += 1
            self._update_access_tracking(key)
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, tier: int = 0) -> None:
        """Set value in cache with intelligent eviction."""
        async with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                await self._delete_internal(key)
            
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl if ttl is not None else self.default_ttl,
                cache_tier=tier
            )
            
            # Check if we need to evict before adding
            while (len(self._cache) >= self.max_size or 
                   self._current_memory + entry.size_estimate > self.max_memory_bytes):
                evicted = await self._evict_entry()
                if not evicted:
                    # Can't evict anything, reject this entry
                    self.logger.warning("Cache full, rejecting new entry", key=key)
                    return
            
            # Add new entry
            self._cache[key] = entry
            self._current_memory += entry.size_estimate
            self._update_access_tracking(key)
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return await self._delete_internal(key)
    
    async def _delete_internal(self, key: str) -> bool:
        """Internal delete method (assumes lock is held)."""
        if key in self._cache:
            entry = self._cache[key]
            del self._cache[key]
            self._current_memory -= entry.size_estimate
            
            # Clean up tracking structures
            if key in self._access_order:
                self._access_order.remove(key)
            if key in self._frequency_count:
                del self._frequency_count[key]
            
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        memory_usage_mb = self._current_memory / (1024 * 1024)
        
        # Calculate tier distribution
        tier_distribution = defaultdict(int)
        for entry in self._cache.values():
            tier_distribution[entry.cache_tier] += 1
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": memory_usage_mb,
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "memory_utilization": memory_usage_mb / (self.max_memory_bytes / (1024 * 1024)),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "policy": self.policy.value,
            "tier_distribution": dict(tier_distribution)
        }
    
    def _update_access_tracking(self, key: str) -> None:
        """Update access tracking based on cache policy."""
        if self.policy in (CachePolicy.LRU, CachePolicy.ARC):
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
        
        if self.policy in (CachePolicy.LFU, CachePolicy.ARC):
            self._frequency_count[key] += 1
    
    async def _evict_entry(self) -> bool:
        """Evict entry based on cache policy."""
        if not self._cache:
            return False
        
        evict_key = None
        
        if self.policy == CachePolicy.LRU:
            evict_key = await self._find_lru_victim()
        elif self.policy == CachePolicy.LFU:
            evict_key = await self._find_lfu_victim()
        elif self.policy == CachePolicy.TTL:
            evict_key = await self._find_ttl_victim()
        elif self.policy == CachePolicy.ARC:
            evict_key = await self._find_arc_victim()
        
        if evict_key:
            await self._delete_internal(evict_key)
            self._evictions += 1
            return True
        
        return False
    
    async def _find_lru_victim(self) -> Optional[str]:
        """Find LRU victim for eviction."""
        if self._access_order:
            return self._access_order[0]
        return None
    
    async def _find_lfu_victim(self) -> Optional[str]:
        """Find LFU victim for eviction."""
        if not self._frequency_count:
            return next(iter(self._cache.keys())) if self._cache else None
        
        return min(self._frequency_count.keys(), key=lambda k: self._frequency_count[k])
    
    async def _find_ttl_victim(self) -> Optional[str]:
        """Find expired entry or oldest entry."""
        # First try to find expired entries
        current_time = time.time()
        for key, entry in self._cache.items():
            if entry.is_expired:
                return key
        
        # If no expired entries, find oldest
        if self._cache:
            return min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        
        return None
    
    async def _find_arc_victim(self) -> Optional[str]:
        """Adaptive replacement cache victim selection."""
        # Simplified ARC: balance between recency and frequency
        if not self._cache:
            return None
        
        # Score based on recency and frequency
        current_time = time.time()
        scores = {}
        
        for key, entry in self._cache.items():
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            frequency_score = self._frequency_count.get(key, 1) / 100.0
            scores[key] = recency_score + frequency_score
        
        return min(scores.keys(), key=lambda k: scores[k])
    
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self._delete_internal(key)
            
            return len(expired_keys)


class AdaptiveLoadBalancer:
    """Intelligent load balancer with performance-based routing."""
    
    def __init__(self):
        self.logger = structlog.get_logger("load_balancer")
        self._backend_metrics: Dict[str, PerformanceMetrics] = {}
        self._backend_weights: Dict[str, float] = {}
        self._backend_health: Dict[str, bool] = {}
        self._round_robin_index = 0
        self._lock = asyncio.Lock()
    
    def register_backend(self, backend_id: str, initial_weight: float = 1.0) -> None:
        """Register a new backend."""
        self._backend_metrics[backend_id] = PerformanceMetrics()
        self._backend_weights[backend_id] = initial_weight
        self._backend_health[backend_id] = True
    
    async def select_backend(self, strategy: str = "adaptive") -> Optional[str]:
        """Select best backend based on strategy."""
        async with self._lock:
            healthy_backends = [bid for bid, healthy in self._backend_health.items() if healthy]
            
            if not healthy_backends:
                return None
            
            if strategy == "round_robin":
                selected = healthy_backends[self._round_robin_index % len(healthy_backends)]
                self._round_robin_index += 1
                return selected
            
            elif strategy == "least_latency":
                return min(healthy_backends, key=lambda b: self._backend_metrics[b].get_average_latency())
            
            elif strategy == "adaptive":
                # Weighted selection based on performance
                scores = {}
                for backend_id in healthy_backends:
                    metrics = self._backend_metrics[backend_id]
                    latency = metrics.get_average_latency()
                    error_rate = metrics.error_count / max(len(metrics.latency_samples), 1)
                    
                    # Lower is better for both latency and error rate
                    score = 1.0 / (latency + 0.001) * (1.0 - error_rate)
                    scores[backend_id] = score * self._backend_weights[backend_id]
                
                return max(scores.keys(), key=lambda b: scores[b])
            
            return healthy_backends[0]  # Fallback
    
    def record_request_metrics(self, backend_id: str, latency: float, success: bool) -> None:
        """Record request metrics for a backend."""
        if backend_id in self._backend_metrics:
            metrics = self._backend_metrics[backend_id]
            metrics.add_latency_sample(latency)
            if not success:
                metrics.error_count += 1
    
    def update_backend_health(self, backend_id: str, healthy: bool) -> None:
        """Update backend health status."""
        self._backend_health[backend_id] = healthy


class PerformanceOptimizer:
    """Adaptive performance optimization engine."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        self.optimization_level = optimization_level
        self.logger = structlog.get_logger("performance_optimizer")
        self._optimization_history: List[Tuple[str, float]] = []
        self._current_optimizations: Set[str] = set()
    
    async def analyze_and_optimize(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyze metrics and suggest optimizations."""
        suggestions = []
        
        # CPU optimization
        if metrics.cpu_usage > 80.0:
            suggestions.append("reduce_cpu_intensive_operations")
            suggestions.append("enable_connection_pooling")
        
        # Memory optimization
        if metrics.memory_usage > 85.0:
            suggestions.append("reduce_cache_size")
            suggestions.append("enable_memory_compression")
        
        # Latency optimization
        avg_latency = metrics.get_average_latency()
        if avg_latency > 100.0:  # 100ms threshold
            suggestions.append("increase_cache_size")
            suggestions.append("enable_request_batching")
        
        # Cache optimization
        if metrics.cache_hits > 0 and metrics.cache_misses > 0:
            hit_rate = metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
            if hit_rate < 0.7:  # 70% hit rate threshold
                suggestions.append("tune_cache_policy")
                suggestions.append("increase_cache_ttl")
        
        return suggestions
    
    async def apply_optimization(self, optimization: str, context: Dict[str, Any]) -> bool:
        """Apply specific optimization."""
        try:
            if optimization == "reduce_cache_size" and "cache" in context:
                cache = context["cache"]
                cache.max_size = int(cache.max_size * 0.8)
                return True
            
            elif optimization == "increase_cache_size" and "cache" in context:
                cache = context["cache"]
                cache.max_size = int(cache.max_size * 1.2)
                return True
            
            elif optimization == "tune_cache_policy" and "cache" in context:
                cache = context["cache"]
                # Switch to more aggressive caching policy
                if cache.policy == CachePolicy.LRU:
                    cache.policy = CachePolicy.ARC
                return True
            
            # Add more optimization implementations as needed
            self.logger.info(f"Applied optimization: {optimization}")
            self._current_optimizations.add(optimization)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization {optimization}: {e}")
            return False


class PerformanceManager:
    """Advanced central performance management system."""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        self.logger = structlog.get_logger("performance_manager")
        
        # Core components
        self.cache = IntelligentCache(
            max_size=5000, 
            max_memory_mb=100,
            policy=CachePolicy.ARC,
            default_ttl=300.0
        )
        self.connection_pool = ConnectionPool(max_connections=100, max_idle=20)
        self.load_balancer = AdaptiveLoadBalancer()
        self.optimizer = PerformanceOptimizer(optimization_level)
        
        # Metrics and monitoring
        self._global_metrics = PerformanceMetrics()
        self._system_monitor: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def start(self):
        """Start comprehensive performance management."""
        self.logger.info("Starting advanced performance management")
        self._running = True
        
        # Start background tasks
        self._system_monitor = asyncio.create_task(self._system_monitoring_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Register default backends for load balancing
        self.load_balancer.register_backend("primary", 1.0)
        self.load_balancer.register_backend("secondary", 0.8)
    
    async def stop(self):
        """Stop performance management."""
        self.logger.info("Stopping performance management")
        self._running = False
        
        # Cancel background tasks
        for task in [self._system_monitor, self._optimization_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
    
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
        """Get comprehensive performance summary."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                "cache": self.cache.get_statistics(),
                "system": {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024)
                },
                "global_metrics": {
                    "average_latency": self._global_metrics.get_average_latency(),
                    "p99_latency": self._global_metrics.get_p99_latency(),
                    "average_throughput": self._global_metrics.get_average_throughput(),
                    "error_count": self._global_metrics.error_count,
                    "total_requests": len(self._global_metrics.latency_samples)
                },
                "optimizations": {
                    "level": self.optimizer.optimization_level.value,
                    "active_optimizations": list(self.optimizer._current_optimizations)
                }
            }
        except ImportError:
            # Fallback if psutil is not available
            return {
                "cache": self.cache.get_statistics(),
                "system": {"cpu_usage": 0, "memory_usage": 0},
                "global_metrics": {
                    "average_latency": self._global_metrics.get_average_latency(),
                    "p99_latency": self._global_metrics.get_p99_latency(),
                    "average_throughput": self._global_metrics.get_average_throughput(),
                    "error_count": self._global_metrics.error_count
                }
            }
    
    def record_request(self, latency: float, success: bool, throughput: float = 0.0) -> None:
        """Record request metrics."""
        self._global_metrics.add_latency_sample(latency)
        if throughput > 0:
            self._global_metrics.add_throughput_sample(throughput)
        if not success:
            self._global_metrics.error_count += 1
    
    async def execute_with_optimization(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with performance optimization."""
        start_time = time.time()
        
        try:
            # Check if this is CPU-intensive work that should be offloaded
            if kwargs.pop('cpu_intensive', False):
                result = await asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, func, *args, **kwargs
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
            
            latency = time.time() - start_time
            self.record_request(latency, True)
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.record_request(latency, False)
            raise e
    
    async def _system_monitoring_loop(self):
        """Monitor system performance metrics."""
        while self._running:
            try:
                # Update global metrics with system info
                try:
                    self._global_metrics.cpu_usage = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    self._global_metrics.memory_usage = memory.percent
                    self._global_metrics.last_update = time.time()
                except ImportError:
                    pass  # Skip if psutil not available
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_loop(self):
        """Intelligent background optimization."""
        while self._running:
            try:
                # Analyze current performance
                suggestions = await self.optimizer.analyze_and_optimize(self._global_metrics)
                
                # Apply optimizations
                for suggestion in suggestions:
                    context = {
                        "cache": self.cache,
                        "connection_pool": self.connection_pool,
                        "metrics": self._global_metrics
                    }
                    
                    success = await self.optimizer.apply_optimization(suggestion, context)
                    if success:
                        self.logger.info(f"Applied optimization: {suggestion}")
                
                await asyncio.sleep(120)  # Optimize every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_loop(self):
        """Background cleanup of expired cache entries."""
        while self._running:
            try:
                # Clean up expired cache entries
                cleaned = await self.cache.cleanup_expired()
                if cleaned > 0:
                    self.logger.debug(f"Cleaned up {cleaned} expired cache entries")
                
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(600)  # Wait 10 minutes on error


_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager(optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE) -> PerformanceManager:
    """Get global performance manager."""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager(optimization_level)
    return _performance_manager


def create_optimized_cache(cache_type: str = "intelligent", **kwargs) -> Union[IntelligentCache, None]:
    """Factory function to create optimized cache instances."""
    if cache_type == "intelligent":
        return IntelligentCache(**kwargs)
    return None


class PerformanceDecorator:
    """Performance monitoring and optimization decorator."""
    
    def __init__(self, cache_result: bool = True, cpu_intensive: bool = False):
        self.cache_result = cache_result
        self.cpu_intensive = cpu_intensive
        self.perf_manager = get_performance_manager()
    
    def __call__(self, func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key if caching is enabled
            cache_key = None
            if self.cache_result:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                cached_result = await self.perf_manager.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute with optimization
            result = await self.perf_manager.execute_with_optimization(
                func, *args, cpu_intensive=self.cpu_intensive, **kwargs
            )
            
            # Cache result if enabled
            if self.cache_result and cache_key:
                await self.perf_manager.cache.set(cache_key, result)
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


# Performance monitoring decorators
def performance_monitored(cache_result: bool = False, cpu_intensive: bool = False):
    """Decorator for performance monitoring and optimization."""
    return PerformanceDecorator(cache_result=cache_result, cpu_intensive=cpu_intensive)


def cached(ttl: Optional[float] = None, tier: int = 0):
    """Simple caching decorator."""
    def decorator(func):
        perf_manager = get_performance_manager()
        
        async def async_wrapper(*args, **kwargs):
            cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = await perf_manager.cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await perf_manager.cache.set(cache_key, result, ttl=ttl, tier=tier)
            return result
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator