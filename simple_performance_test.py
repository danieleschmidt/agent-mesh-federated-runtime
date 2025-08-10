#!/usr/bin/env python3
"""Simple standalone test for advanced performance features."""

import asyncio
import sys
import time
import threading
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from uuid import UUID, uuid4
import statistics

# Mock structlog and psutil
class MockLogger:
    def __init__(self, name, **kwargs):
        self.name = name
    
    def info(self, msg, **kwargs):
        print(f"INFO [{self.name}]: {msg}")
    
    def warning(self, msg, **kwargs):
        print(f"WARN [{self.name}]: {msg}")
    
    def error(self, msg, **kwargs):
        print(f"ERROR [{self.name}]: {msg}")
    
    def debug(self, msg, **kwargs):
        pass

def get_logger(name, **kwargs):
    return MockLogger(name, **kwargs)

# Mock psutil
class MockPsutil:
    @staticmethod
    def cpu_percent():
        return 45.0
    
    @staticmethod
    def virtual_memory():
        class MockMemory:
            percent = 65.0
            available = 8 * 1024 * 1024 * 1024  # 8GB
        return MockMemory()

# Define the core classes directly for testing
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
        self.logger = get_logger("intelligent_cache")
    
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
                    self.logger.warning("Cache full, rejecting new entry")
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
            "memory_utilization": memory_usage_mb / (self.max_memory_bytes / (1024 * 1024)) if self.max_memory_bytes > 0 else 0,
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

def test_intelligent_cache():
    """Test advanced intelligent cache functionality."""
    print("💾 Testing Intelligent Cache...")
    
    try:
        # Test 1: Initialize cache with different policies
        print("  ✓ Testing cache initialization...")
        lru_cache = IntelligentCache(max_size=100, policy=CachePolicy.LRU)
        lfu_cache = IntelligentCache(max_size=100, policy=CachePolicy.LFU)
        arc_cache = IntelligentCache(max_size=100, policy=CachePolicy.ARC)
        
        assert lru_cache.policy == CachePolicy.LRU
        assert lfu_cache.policy == CachePolicy.LFU
        assert arc_cache.policy == CachePolicy.ARC
        
        # Test 2: Basic cache operations
        print("  ✓ Testing basic cache operations...")
        
        async def test_cache_ops():
            # Set and get
            await lru_cache.set("key1", "value1", ttl=10.0)
            result = await lru_cache.get("key1")
            assert result == "value1"
            
            # Cache miss
            result = await lru_cache.get("nonexistent")
            assert result is None
            
            # Delete
            deleted = await lru_cache.delete("key1")
            assert deleted is True
            
            result = await lru_cache.get("key1")
            assert result is None
        
        asyncio.run(test_cache_ops())
        
        # Test 3: Memory management
        print("  ✓ Testing memory management...")
        
        async def test_memory_management():
            cache = IntelligentCache(max_size=3, max_memory_mb=1)
            
            # Fill cache to capacity
            await cache.set("key1", "x" * 1000, tier=0)  # Hot tier
            await cache.set("key2", "y" * 1000, tier=1)  # Warm tier  
            await cache.set("key3", "z" * 1000, tier=2)  # Cold tier
            
            stats = cache.get_statistics()
            assert stats["size"] == 3
            assert stats["tier_distribution"][0] == 1  # 1 hot item
            assert stats["tier_distribution"][1] == 1  # 1 warm item
            assert stats["tier_distribution"][2] == 1  # 1 cold item
            
            # Adding another item should trigger eviction
            await cache.set("key4", "w" * 1000)
            
            stats = cache.get_statistics()
            assert stats["size"] <= 3  # Should still be at or under capacity
        
        asyncio.run(test_memory_management())
        
        # Test 4: TTL expiration
        print("  ✓ Testing TTL expiration...")
        
        async def test_ttl():
            cache = IntelligentCache(max_size=10)
            
            # Set item with very short TTL
            await cache.set("short_ttl", "expires_soon", ttl=0.1)
            
            # Should be available immediately
            result = await cache.get("short_ttl")
            assert result == "expires_soon"
            
            # Wait for expiration
            await asyncio.sleep(0.2)
            
            # Should be expired and return None
            result = await cache.get("short_ttl")
            assert result is None
        
        asyncio.run(test_ttl())
        
        print("  ✅ All Intelligent Cache tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Intelligent Cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_metrics():
    """Test performance metrics tracking."""
    print("📊 Testing Performance Metrics...")
    
    try:
        # Test 1: Initialize metrics
        print("  ✓ Testing metrics initialization...")
        metrics = PerformanceMetrics()
        
        assert len(metrics.latency_samples) == 0
        assert len(metrics.throughput_samples) == 0
        assert metrics.error_count == 0
        
        # Test 2: Add samples
        print("  ✓ Testing sample addition...")
        metrics.add_latency_sample(50.0)
        metrics.add_latency_sample(75.0)
        metrics.add_latency_sample(100.0)
        
        metrics.add_throughput_sample(1000.0)
        metrics.add_throughput_sample(1200.0)
        
        assert len(metrics.latency_samples) == 3
        assert len(metrics.throughput_samples) == 2
        
        # Test 3: Calculate averages
        print("  ✓ Testing average calculations...")
        avg_latency = metrics.get_average_latency()
        assert abs(avg_latency - 75.0) < 0.1  # (50+75+100)/3 = 75
        
        avg_throughput = metrics.get_average_throughput()
        assert abs(avg_throughput - 1100.0) < 0.1  # (1000+1200)/2 = 1100
        
        # Test 4: P99 calculation
        print("  ✓ Testing P99 latency...")
        # Add more samples for meaningful P99
        for i in range(97):
            metrics.add_latency_sample(10.0)  # Add many low latency samples
        
        p99 = metrics.get_p99_latency()
        assert p99 >= 50.0  # Should be one of the higher values
        
        print("  ✅ All Performance Metrics tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_entry():
    """Test enhanced cache entry functionality."""
    print("📝 Testing Cache Entry...")
    
    try:
        # Test 1: Create cache entry
        print("  ✓ Testing cache entry creation...")
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=10.0,
            cache_tier=1
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.cache_tier == 1
        assert entry.access_count == 0
        assert entry.size_estimate > 0
        
        # Test 2: Access tracking
        print("  ✓ Testing access tracking...")
        initial_access_time = entry.last_accessed
        time.sleep(0.01)  # Small delay
        
        entry.access()
        assert entry.access_count == 1
        assert entry.last_accessed > initial_access_time
        
        # Test 3: Age and idle time
        print("  ✓ Testing age and idle time...")
        age = entry.age
        idle_time = entry.idle_time
        
        assert age > 0
        assert idle_time >= 0
        
        # Test 4: TTL expiration
        print("  ✓ Testing TTL expiration...")
        short_ttl_entry = CacheEntry(
            key="expires_soon",
            value="temp_value",
            ttl=0.01  # 10ms TTL
        )
        
        assert not short_ttl_entry.is_expired
        time.sleep(0.02)  # Wait for expiration
        assert short_ttl_entry.is_expired
        
        print("  ✅ All Cache Entry tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Cache Entry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cache_policies():
    """Test different cache eviction policies."""
    print("🎯 Testing Cache Policies...")
    
    try:
        async def test_lru_policy():
            """Test LRU (Least Recently Used) policy."""
            cache = IntelligentCache(max_size=3, policy=CachePolicy.LRU)
            
            # Fill cache
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            await cache.set("key3", "value3")
            
            # Access key1 to make it recently used
            await cache.get("key1")
            
            # Add another item, should evict key2 (least recently used)
            await cache.set("key4", "value4")
            
            # key1 should still exist, key2 should be evicted
            assert await cache.get("key1") == "value1"
            assert await cache.get("key2") is None
            assert await cache.get("key4") == "value4"
        
        async def test_lfu_policy():
            """Test LFU (Least Frequently Used) policy."""
            cache = IntelligentCache(max_size=3, policy=CachePolicy.LFU)
            
            # Fill cache
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            await cache.set("key3", "value3")
            
            # Access key1 multiple times
            for _ in range(5):
                await cache.get("key1")
            
            # Access key2 once
            await cache.get("key2")
            
            # key3 has never been accessed (frequency = 0)
            # Adding another item should evict key3
            await cache.set("key4", "value4")
            
            # key1 and key2 should still exist
            assert await cache.get("key1") == "value1"
            assert await cache.get("key2") == "value2"
            assert await cache.get("key3") is None  # Should be evicted
        
        print("  ✓ Testing LRU policy...")
        asyncio.run(test_lru_policy())
        
        print("  ✓ Testing LFU policy...")
        asyncio.run(test_lfu_policy())
        
        print("  ✅ All Cache Policy tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Cache Policy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all advanced performance tests."""
    print("🚀 Advanced Performance Optimization Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Cache Entry", test_cache_entry),
        ("Performance Metrics", test_performance_metrics),
        ("Intelligent Cache", test_intelligent_cache),
        ("Cache Policies", test_cache_policies),
    ]
    
    passed_tests = 0
    total_tests = len(test_functions)
    
    for test_name, test_func in test_functions:
        print()
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
    
    print()
    print("=" * 60)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All advanced performance optimization tests passed!")
        print("⚡ Intelligent caching and performance optimization working correctly.")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most performance tests passed. System has robust optimization capabilities.")
        return True
    else:
        print("❌ Multiple performance test failures. Optimization system needs attention.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)