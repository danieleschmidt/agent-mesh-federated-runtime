"""Advanced performance optimization and caching system for Agent Mesh.

This module provides intelligent caching, performance optimization,
resource management, and auto-scaling capabilities.
"""

import asyncio
import hashlib
import pickle
import time
import threading
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

import structlog

from .monitoring import MetricsCollector


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    
    L1_MEMORY = "l1_memory"  # Fast in-memory cache
    L2_DISTRIBUTED = "l2_distributed"  # Distributed cache
    L3_PERSISTENT = "l3_persistent"  # Persistent storage cache


class OptimizationTarget(Enum):
    """Performance optimization targets."""
    
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    BALANCED = "balanced"


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)
    
    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


@dataclass
class PerformanceProfile:
    """Performance profile for optimization."""
    
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class IntelligentCache:
    """Intelligent multi-level cache with adaptive strategies."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 100,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        default_ttl: Optional[float] = None
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        self.logger = structlog.get_logger("intelligent_cache")
        
        # Cache storage
        self._entries: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU
        self._frequency_count = defaultdict(int)  # For LFU
        
        # Statistics
        self.stats = CacheStats()
        
        # Adaptive strategy state
        self._access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._strategy_performance: Dict[CacheStrategy, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._maintenance_enabled = False
    
    async def start(self) -> None:
        """Start cache maintenance."""
        if self._maintenance_enabled:
            return
        
        self._maintenance_enabled = True
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        self.logger.info("Intelligent cache started", 
                        max_size=self.max_size,
                        strategy=self.strategy.value)
    
    async def stop(self) -> None:
        """Stop cache maintenance."""
        self._maintenance_enabled = False
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Intelligent cache stopped")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access information
            entry.touch()
            self._update_access_tracking(key)
            
            self.stats.hits += 1
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check if we need to evict
            if key not in self._entries:
                if len(self._entries) >= self.max_size:
                    self._evict_entries(1)
                
                # Check memory limit
                if self.stats.total_size + size_bytes > self.max_memory_bytes:
                    self._evict_by_memory(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )
            
            # Remove old entry if exists
            if key in self._entries:
                old_entry = self._entries[key]
                self.stats.total_size -= old_entry.size_bytes
            
            # Add new entry
            self._entries[key] = entry
            self.stats.total_size += size_bytes
            self.stats.entry_count = len(self._entries)
            
            # Update tracking structures
            self._access_order[key] = time.time()
            self._frequency_count[key] += 1
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._entries:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._access_order.clear()
            self._frequency_count.clear()
            self.stats = CacheStats()
    
    def get_statistics(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size=self.stats.total_size,
                entry_count=self.stats.entry_count
            )
    
    def optimize_strategy(self) -> CacheStrategy:
        """Optimize cache strategy based on access patterns."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return self.strategy
        
        # Analyze access patterns
        if len(self._access_patterns) < 10:
            return CacheStrategy.LRU  # Default for small datasets
        
        # Calculate strategy effectiveness
        recency_score = self._calculate_recency_score()
        frequency_score = self._calculate_frequency_score()
        temporal_score = self._calculate_temporal_score()
        
        # Choose best strategy
        if recency_score > 0.7:
            return CacheStrategy.LRU
        elif frequency_score > 0.7:
            return CacheStrategy.LFU
        elif temporal_score > 0.5:
            return CacheStrategy.TTL
        else:
            return CacheStrategy.LRU  # Default fallback
    
    # Private methods
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update tracking."""
        if key in self._entries:
            entry = self._entries[key]
            self.stats.total_size -= entry.size_bytes
            del self._entries[key]
            
            if key in self._access_order:
                del self._access_order[key]
            
            if key in self._frequency_count:
                del self._frequency_count[key]
            
            self.stats.entry_count = len(self._entries)
    
    def _evict_entries(self, count: int) -> None:
        """Evict entries based on current strategy."""
        current_strategy = self.optimize_strategy()
        
        for _ in range(count):
            if not self._entries:
                break
            
            if current_strategy == CacheStrategy.LRU:
                key = next(iter(self._access_order))
            elif current_strategy == CacheStrategy.LFU:
                key = min(self._frequency_count.keys(), key=self._frequency_count.get)
            elif current_strategy == CacheStrategy.FIFO:
                key = next(iter(self._entries))
            elif current_strategy == CacheStrategy.TTL:
                # Find expired entries first, then oldest
                expired_keys = [k for k, e in self._entries.items() if e.is_expired()]
                if expired_keys:
                    key = expired_keys[0]
                else:
                    key = min(self._entries.keys(), key=lambda k: self._entries[k].created_at)
            else:
                key = next(iter(self._entries))
            
            self._remove_entry(key)
            self.stats.evictions += 1
    
    def _evict_by_memory(self, needed_bytes: int) -> None:
        """Evict entries to free up memory."""
        freed_bytes = 0
        
        while freed_bytes < needed_bytes and self._entries:
            # Find largest entry or least valuable entry
            key = max(self._entries.keys(), 
                     key=lambda k: self._entries[k].size_bytes / max(1, self._entries[k].access_count))
            
            freed_bytes += self._entries[key].size_bytes
            self._remove_entry(key)
            self.stats.evictions += 1
    
    def _update_access_tracking(self, key: str) -> None:
        """Update access pattern tracking."""
        current_time = time.time()
        self._access_patterns[key].append(current_time)
        
        # Keep only recent access times (last hour)
        cutoff_time = current_time - 3600
        self._access_patterns[key] = [
            t for t in self._access_patterns[key] if t > cutoff_time
        ]
        
        # Update access order for LRU
        if key in self._access_order:
            del self._access_order[key]
        self._access_order[key] = current_time
    
    def _calculate_recency_score(self) -> float:
        """Calculate how much recency matters in access patterns."""
        if not self._access_patterns:
            return 0.0
        
        recent_access_weight = 0.0
        total_weight = 0.0
        current_time = time.time()
        
        for key, accesses in self._access_patterns.items():
            if len(accesses) < 2:
                continue
            
            # Calculate recency bias
            for i, access_time in enumerate(reversed(accesses[-10:])):  # Last 10 accesses
                age = current_time - access_time
                weight = 1.0 / (1.0 + age / 3600)  # Decay over hours
                recent_access_weight += weight * (i + 1)  # More recent = higher weight
                total_weight += weight
        
        return recent_access_weight / max(1, total_weight)
    
    def _calculate_frequency_score(self) -> float:
        """Calculate how much frequency matters in access patterns."""
        if not self._frequency_count:
            return 0.0
        
        # Calculate coefficient of variation for frequency
        frequencies = list(self._frequency_count.values())
        if not frequencies:
            return 0.0
        
        mean_freq = sum(frequencies) / len(frequencies)
        variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        
        if mean_freq == 0:
            return 0.0
        
        cv = (variance ** 0.5) / mean_freq
        return min(1.0, cv)  # Higher variation = more frequency-based
    
    def _calculate_temporal_score(self) -> float:
        """Calculate temporal locality in access patterns."""
        if not self._access_patterns:
            return 0.0
        
        temporal_locality = 0.0
        pattern_count = 0
        
        for accesses in self._access_patterns.values():
            if len(accesses) < 3:
                continue
            
            # Calculate inter-access time variability
            intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
            if not intervals:
                continue
            
            mean_interval = sum(intervals) / len(intervals)
            variance = sum((i - mean_interval) ** 2 for i in intervals) / len(intervals)
            
            if mean_interval > 0:
                cv = (variance ** 0.5) / mean_interval
                temporal_locality += 1.0 / (1.0 + cv)  # Lower variation = higher locality
                pattern_count += 1
        
        return temporal_locality / max(1, pattern_count)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:10])  # Sample first 10
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(obj.items())[:10])  # Sample first 10
            else:
                return 1024  # Default 1KB estimate
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self._maintenance_enabled:
            try:
                with self._lock:
                    # Remove expired entries
                    expired_keys = [
                        key for key, entry in self._entries.items() 
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                        self.stats.evictions += 1
                    
                    if expired_keys:
                        self.logger.debug("Removed expired cache entries", count=len(expired_keys))
                
                # Clean old access patterns
                current_time = time.time()
                cutoff_time = current_time - 7200  # 2 hours
                
                for key in list(self._access_patterns.keys()):
                    self._access_patterns[key] = [
                        t for t in self._access_patterns[key] if t > cutoff_time
                    ]
                    if not self._access_patterns[key]:
                        del self._access_patterns[key]
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Cache maintenance error", error=str(e))
                await asyncio.sleep(60)


class PerformanceOptimizer:
    """Performance optimizer with adaptive strategies."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("performance_optimizer", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Optimization state
        self.current_target = OptimizationTarget.BALANCED
        self.optimization_history: List[PerformanceProfile] = []
        
        # Caching system
        self.cache = IntelligentCache(max_size=5000, max_memory_mb=256)
        
        # Performance tracking
        self.operation_metrics: Dict[str, List[float]] = defaultdict(list)
        self.resource_usage: Dict[str, List[float]] = defaultdict(list)
        
        # Optimization strategies
        self.optimization_strategies = {
            OptimizationTarget.LATENCY: self._optimize_for_latency,
            OptimizationTarget.THROUGHPUT: self._optimize_for_throughput,
            OptimizationTarget.MEMORY: self._optimize_for_memory,
            OptimizationTarget.CPU: self._optimize_for_cpu,
            OptimizationTarget.BALANCED: self._optimize_balanced
        }
        
        # Background optimization
        self._optimization_task: Optional[asyncio.Task] = None
        self._optimization_enabled = False
    
    async def start(self) -> None:
        """Start performance optimization."""
        if self._optimization_enabled:
            return
        
        self._optimization_enabled = True
        await self.cache.start()
        
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Performance optimizer started", target=self.current_target.value)
    
    async def stop(self) -> None:
        """Stop performance optimization."""
        self._optimization_enabled = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        
        await self.cache.stop()
        self.logger.info("Performance optimizer stopped")
    
    def record_operation(self, operation: str, duration_ms: float, success: bool = True) -> None:
        """Record operation metrics."""
        self.operation_metrics[f"{operation}_duration"].append(duration_ms)
        self.operation_metrics[f"{operation}_success"].append(1.0 if success else 0.0)
        
        # Keep only recent data
        for key in self.operation_metrics:
            if len(self.operation_metrics[key]) > 1000:
                self.operation_metrics[key] = self.operation_metrics[key][-500:]
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached computation result."""
        return self.cache.get(key)
    
    def cache_result(
        self, 
        key: str, 
        result: Any, 
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Cache computation result."""
        self.cache.put(key, result, ttl, metadata)
    
    def optimize_for_target(self, target: OptimizationTarget) -> None:
        """Set optimization target."""
        if target != self.current_target:
            self.logger.info("Optimization target changed", 
                           old_target=self.current_target.value,
                           new_target=target.value)
            self.current_target = target
    
    def get_performance_profile(self) -> PerformanceProfile:
        """Get current performance profile."""
        profile = PerformanceProfile()
        
        # Calculate latency metrics
        all_durations = []
        for key, durations in self.operation_metrics.items():
            if key.endswith('_duration'):
                all_durations.extend(durations[-100:])  # Recent 100 operations
        
        if all_durations:
            all_durations.sort()
            profile.avg_latency_ms = sum(all_durations) / len(all_durations)
            
            if len(all_durations) >= 20:  # Need enough data for percentiles
                p95_idx = int(len(all_durations) * 0.95)
                p99_idx = int(len(all_durations) * 0.99)
                profile.p95_latency_ms = all_durations[p95_idx]
                profile.p99_latency_ms = all_durations[p99_idx]
        
        # Calculate throughput
        recent_operations = sum(len(durations[-60:]) for durations in self.operation_metrics.values())
        profile.throughput_ops_sec = recent_operations / 60.0  # Operations per second
        
        # Get cache metrics
        cache_stats = self.cache.get_statistics()
        profile.cache_hit_rate = cache_stats.hit_rate
        
        # Calculate error rate
        all_success_rates = []
        for key, success_values in self.operation_metrics.items():
            if key.endswith('_success'):
                recent_success = success_values[-100:]  # Recent 100 operations
                if recent_success:
                    success_rate = sum(recent_success) / len(recent_success)
                    all_success_rates.append(success_rate)
        
        if all_success_rates:
            avg_success_rate = sum(all_success_rates) / len(all_success_rates)
            profile.error_rate = 1.0 - avg_success_rate
        
        return profile
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        profile = self.get_performance_profile()
        cache_stats = self.cache.get_statistics()
        
        # High latency recommendation
        if profile.avg_latency_ms > 1000:  # > 1 second
            recommendations.append({
                "type": "latency",
                "severity": "high",
                "description": f"High average latency: {profile.avg_latency_ms:.1f}ms",
                "recommendation": "Consider optimizing hot paths, adding caching, or scaling resources",
                "target": OptimizationTarget.LATENCY.value
            })
        
        # Low cache hit rate
        if cache_stats.hit_rate < 0.6:  # < 60%
            recommendations.append({
                "type": "caching",
                "severity": "medium",
                "description": f"Low cache hit rate: {cache_stats.hit_rate:.1%}",
                "recommendation": "Review caching strategy, increase cache size, or adjust TTL values",
                "target": "caching"
            })
        
        # High error rate
        if profile.error_rate > 0.05:  # > 5%
            recommendations.append({
                "type": "reliability",
                "severity": "high",
                "description": f"High error rate: {profile.error_rate:.1%}",
                "recommendation": "Investigate error causes, add retry mechanisms, or improve error handling",
                "target": "reliability"
            })
        
        # Low throughput
        if profile.throughput_ops_sec < 10:  # < 10 ops/sec
            recommendations.append({
                "type": "throughput",
                "severity": "medium",
                "description": f"Low throughput: {profile.throughput_ops_sec:.1f} ops/sec",
                "recommendation": "Consider parallel processing, connection pooling, or horizontal scaling",
                "target": OptimizationTarget.THROUGHPUT.value
            })
        
        return recommendations
    
    # Private methods
    
    async def _optimization_loop(self) -> None:
        """Background optimization loop."""
        while self._optimization_enabled:
            try:
                # Get current performance profile
                profile = self.get_performance_profile()
                self.optimization_history.append(profile)
                
                # Keep history manageable
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-50:]
                
                # Apply current optimization strategy
                strategy = self.optimization_strategies.get(self.current_target)
                if strategy:
                    await strategy(profile)
                
                # Log performance periodically
                if len(self.optimization_history) % 10 == 0:
                    self.logger.info("Performance profile",
                                   avg_latency_ms=profile.avg_latency_ms,
                                   throughput_ops_sec=profile.throughput_ops_sec,
                                   cache_hit_rate=profile.cache_hit_rate,
                                   error_rate=profile.error_rate)
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Optimization loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _optimize_for_latency(self, profile: PerformanceProfile) -> None:
        """Optimize for low latency."""
        # Increase cache size if hit rate is low
        if profile.cache_hit_rate < 0.8:
            current_size = self.cache.max_size
            new_size = min(current_size * 2, 10000)
            if new_size != current_size:
                self.cache.max_size = new_size
                self.logger.info("Increased cache size for latency", 
                               old_size=current_size, new_size=new_size)
        
        # Optimize cache strategy for recency
        if self.cache.strategy == CacheStrategy.ADAPTIVE:
            optimal_strategy = self.cache.optimize_strategy()
            if optimal_strategy != CacheStrategy.LRU:
                self.cache.strategy = CacheStrategy.LRU
                self.logger.info("Switched to LRU cache strategy for latency")
    
    async def _optimize_for_throughput(self, profile: PerformanceProfile) -> None:
        """Optimize for high throughput."""
        # Adjust cache for throughput
        if profile.cache_hit_rate > 0.9:
            # High hit rate, can reduce cache size to free memory for processing
            current_size = self.cache.max_size
            new_size = max(current_size // 2, 1000)
            if new_size != current_size:
                self.cache.max_size = new_size
                self.logger.info("Reduced cache size for throughput", 
                               old_size=current_size, new_size=new_size)
    
    async def _optimize_for_memory(self, profile: PerformanceProfile) -> None:
        """Optimize for low memory usage."""
        # Reduce cache size
        current_size = self.cache.max_size
        new_size = max(current_size // 2, 500)
        if new_size != current_size:
            self.cache.max_size = new_size
            self.logger.info("Reduced cache size for memory optimization", 
                           old_size=current_size, new_size=new_size)
        
        # Force cache cleanup
        with self.cache._lock:
            self.cache._evict_entries(len(self.cache._entries) // 4)
    
    async def _optimize_for_cpu(self, profile: PerformanceProfile) -> None:
        """Optimize for low CPU usage."""
        # Increase cache hit rate to reduce CPU-intensive operations
        if profile.cache_hit_rate < 0.7:
            current_size = self.cache.max_size
            new_size = min(current_size * 2, 8000)
            if new_size != current_size:
                self.cache.max_size = new_size
                self.logger.info("Increased cache size for CPU optimization", 
                               old_size=current_size, new_size=new_size)
    
    async def _optimize_balanced(self, profile: PerformanceProfile) -> None:
        """Balanced optimization approach."""
        # Adaptive optimization based on current bottlenecks
        if profile.avg_latency_ms > 500:
            await self._optimize_for_latency(profile)
        elif profile.throughput_ops_sec < 20:
            await self._optimize_for_throughput(profile)
        elif profile.cache_hit_rate < 0.5:
            # Improve caching
            current_size = self.cache.max_size
            new_size = min(current_size + 1000, 7000)
            if new_size != current_size:
                self.cache.max_size = new_size
                self.logger.info("Adjusted cache size for balanced optimization", 
                               old_size=current_size, new_size=new_size)


class ResourceManager:
    """Intelligent resource management and allocation."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("resource_manager", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Resource pools
        self.connection_pools: Dict[str, Any] = {}
        self.thread_pools: Dict[str, Any] = {}
        self.memory_pools: Dict[str, Any] = {}
        
        # Resource limits
        self.max_connections = 1000
        self.max_threads = 100
        self.max_memory_mb = 2048
        
        # Usage tracking
        self.resource_usage: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Auto-scaling configuration
        self.auto_scaling_enabled = True
        self.scale_up_threshold = 0.8  # 80% utilization
        self.scale_down_threshold = 0.3  # 30% utilization
        
    async def get_connection_pool(self, pool_name: str, **config) -> Any:
        """Get or create connection pool."""
        if pool_name not in self.connection_pools:
            initial_size = config.get('initial_size', 10)
            max_size = config.get('max_size', 100)
            
            # Create simple connection pool simulation
            pool = {
                'name': pool_name,
                'initial_size': initial_size,
                'max_size': max_size,
                'current_size': initial_size,
                'active_connections': 0,
                'created_at': time.time()
            }
            
            self.connection_pools[pool_name] = pool
            
            self.logger.info("Connection pool created", 
                           pool_name=pool_name,
                           initial_size=initial_size,
                           max_size=max_size)
        
        return self.connection_pools[pool_name]
    
    async def scale_resources(self, resource_type: str, target_utilization: float) -> bool:
        """Auto-scale resources based on utilization."""
        if not self.auto_scaling_enabled:
            return False
        
        try:
            current_usage = self.resource_usage.get(resource_type, {})
            current_utilization = current_usage.get('utilization', 0.0)
            
            if current_utilization > self.scale_up_threshold:
                # Scale up
                return await self._scale_up_resource(resource_type)
            elif current_utilization < self.scale_down_threshold:
                # Scale down
                return await self._scale_down_resource(resource_type)
            
            return False
            
        except Exception as e:
            self.logger.error("Resource scaling error", 
                            resource_type=resource_type, error=str(e))
            return False
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        stats = {
            'connection_pools': len(self.connection_pools),
            'thread_pools': len(self.thread_pools),
            'total_connections': sum(
                pool.get('current_size', 0) for pool in self.connection_pools.values()
            ),
            'active_connections': sum(
                pool.get('active_connections', 0) for pool in self.connection_pools.values()
            ),
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'resource_usage': dict(self.resource_usage)
        }
        
        return stats
    
    async def _scale_up_resource(self, resource_type: str) -> bool:
        """Scale up specific resource type."""
        if resource_type == 'connections':
            # Scale up connection pools
            for pool_name, pool in self.connection_pools.items():
                current_size = pool.get('current_size', 0)
                max_size = pool.get('max_size', 100)
                
                if current_size < max_size:
                    new_size = min(current_size * 2, max_size)
                    pool['current_size'] = new_size
                    
                    self.logger.info("Scaled up connection pool", 
                                   pool_name=pool_name,
                                   old_size=current_size,
                                   new_size=new_size)
                    return True
        
        return False
    
    async def _scale_down_resource(self, resource_type: str) -> bool:
        """Scale down specific resource type."""
        if resource_type == 'connections':
            # Scale down connection pools
            for pool_name, pool in self.connection_pools.items():
                current_size = pool.get('current_size', 0)
                initial_size = pool.get('initial_size', 10)
                
                if current_size > initial_size:
                    new_size = max(current_size // 2, initial_size)
                    pool['current_size'] = new_size
                    
                    self.logger.info("Scaled down connection pool", 
                                   pool_name=pool_name,
                                   old_size=current_size,
                                   new_size=new_size)
                    return True
        
        return False


# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_optimizer(node_id: Optional[UUID] = None) -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(node_id)
    return _global_optimizer


async def start_optimization(node_id: Optional[UUID] = None) -> None:
    """Start global performance optimization."""
    optimizer = get_optimizer(node_id)
    await optimizer.start()


async def stop_optimization() -> None:
    """Stop global performance optimization."""
    if _global_optimizer:
        await _global_optimizer.stop()


def cached(ttl: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for automatic result caching."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            optimizer = get_optimizer()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            
            # Try to get cached result
            cached_result = optimizer.get_cached_result(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                optimizer.cache_result(cache_key, result, ttl)
                optimizer.record_operation(func.__name__, duration_ms, True)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                optimizer.record_operation(func.__name__, duration_ms, False)
                raise
        
        return wrapper
    return decorator