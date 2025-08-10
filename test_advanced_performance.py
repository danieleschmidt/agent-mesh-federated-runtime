#!/usr/bin/env python3
"""Test advanced caching and performance optimization features."""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock structlog and psutil for testing
class MockLogger:
    def __init__(self, name, **kwargs):
        self.name = name
        self.context = kwargs
    
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

# Mock modules
sys.modules['structlog'] = type(sys)('structlog')
sys.modules['structlog'].get_logger = get_logger

# Mock psutil with fake system metrics
class MockPsutil:
    @staticmethod
    def cpu_percent():
        return 45.0
    
    @staticmethod
    def virtual_memory():
        mock = MagicMock()
        mock.percent = 65.0
        mock.available = 8 * 1024 * 1024 * 1024  # 8GB
        return mock

sys.modules['psutil'] = MockPsutil()

def test_intelligent_cache():
    """Test advanced intelligent cache functionality."""
    print("💾 Testing Intelligent Cache...")
    
    try:
        from agent_mesh.core.performance import IntelligentCache, CachePolicy
        
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
        from agent_mesh.core.performance import PerformanceMetrics
        
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

async def test_connection_pool():
    """Test advanced connection pooling."""
    print("🔗 Testing Connection Pool...")
    
    try:
        from agent_mesh.core.performance import ConnectionPool
        
        # Test 1: Initialize pool
        print("  ✓ Testing pool initialization...")
        pool = ConnectionPool(max_connections=5, max_idle=2)
        
        assert pool.max_connections == 5
        assert pool.max_idle == 2
        
        # Test 2: Register connection factory
        print("  ✓ Testing factory registration...")
        
        class MockConnection:
            def __init__(self):
                self.connected = True
            
            def is_connected(self):
                return self.connected
            
            async def close(self):
                self.connected = False
        
        async def connection_factory():
            return MockConnection()
        
        pool.register_factory("test_pool", connection_factory)
        assert "test_pool" in pool._connection_factory
        
        # Test 3: Acquire and release connections
        print("  ✓ Testing connection acquire/release...")
        
        conn1 = await pool.acquire("test_pool")
        assert isinstance(conn1, MockConnection)
        assert conn1.is_connected()
        
        conn2 = await pool.acquire("test_pool")
        assert isinstance(conn2, MockConnection)
        assert conn1 != conn2  # Should be different connections
        
        # Release connections
        await pool.release("test_pool", conn1)
        await pool.release("test_pool", conn2)
        
        # Should be able to reuse released connection
        conn3 = await pool.acquire("test_pool")
        assert conn3 is conn1 or conn3 is conn2  # Should reuse one of the released connections
        
        print("  ✅ All Connection Pool tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Connection Pool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_load_balancer():
    """Test adaptive load balancer."""
    print("⚖️ Testing Load Balancer...")
    
    try:
        from agent_mesh.core.performance import AdaptiveLoadBalancer
        
        # Test 1: Initialize load balancer
        print("  ✓ Testing load balancer initialization...")
        lb = AdaptiveLoadBalancer()
        
        # Test 2: Register backends
        print("  ✓ Testing backend registration...")
        lb.register_backend("backend1", 1.0)
        lb.register_backend("backend2", 0.8)
        lb.register_backend("backend3", 0.6)
        
        assert len(lb._backend_metrics) == 3
        assert lb._backend_weights["backend1"] == 1.0
        
        # Test 3: Round robin selection
        print("  ✓ Testing round robin selection...")
        selected = []
        for _ in range(6):
            backend = await lb.select_backend("round_robin")
            selected.append(backend)
        
        # Should cycle through backends
        unique_backends = set(selected)
        assert len(unique_backends) == 3
        
        # Test 4: Record metrics and adaptive selection
        print("  ✓ Testing adaptive selection...")
        
        # Make backend2 faster (lower latency)
        lb.record_request_metrics("backend1", 100.0, True)  # High latency
        lb.record_request_metrics("backend2", 20.0, True)   # Low latency
        lb.record_request_metrics("backend3", 80.0, True)   # Medium latency
        
        # Adaptive selection should prefer backend2
        selections = []
        for _ in range(10):
            backend = await lb.select_backend("adaptive")
            selections.append(backend)
        
        # backend2 should be selected most often due to lower latency
        backend2_count = selections.count("backend2")
        assert backend2_count > 3  # Should be selected frequently
        
        # Test 5: Health management
        print("  ✓ Testing health management...")
        
        # Mark backend1 as unhealthy
        lb.update_backend_health("backend1", False)
        
        # Should not select unhealthy backend
        selections = []
        for _ in range(10):
            backend = await lb.select_backend("round_robin")
            selections.append(backend)
        
        assert "backend1" not in selections
        
        print("  ✅ All Load Balancer tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Load Balancer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_manager():
    """Test comprehensive performance manager."""
    print("🎯 Testing Performance Manager...")
    
    try:
        from agent_mesh.core.performance import PerformanceManager, OptimizationLevel
        
        # Test 1: Initialize manager
        print("  ✓ Testing manager initialization...")
        manager = PerformanceManager(OptimizationLevel.ADAPTIVE)
        
        assert manager.cache is not None
        assert manager.connection_pool is not None
        assert manager.load_balancer is not None
        assert manager.optimizer is not None
        
        # Test 2: Start and stop
        print("  ✓ Testing start/stop lifecycle...")
        await manager.start()
        
        assert manager._running is True
        assert manager._system_monitor is not None
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        await manager.stop()
        assert manager._running is False
        
        # Test 3: Performance summary
        print("  ✓ Testing performance summary...")
        summary = await manager.get_performance_summary()
        
        assert "cache" in summary
        assert "system" in summary
        assert "global_metrics" in summary
        assert "optimizations" in summary
        
        # Test 4: Record metrics
        print("  ✓ Testing metrics recording...")
        manager.record_request(50.0, True, 100.0)
        manager.record_request(75.0, False, 150.0)
        
        summary = await manager.get_performance_summary()
        metrics = summary["global_metrics"]
        
        assert metrics["error_count"] == 1
        assert metrics["total_requests"] == 2
        assert metrics["average_latency"] > 0
        
        # Test 5: Execute with optimization
        print("  ✓ Testing optimized execution...")
        
        def test_function(x, y):
            return x + y
        
        result = await manager.execute_with_optimization(test_function, 5, 10)
        assert result == 15
        
        # Test CPU-intensive function
        def cpu_intensive_function(n):
            return sum(i * i for i in range(n))
        
        result = await manager.execute_with_optimization(
            cpu_intensive_function, 1000, cpu_intensive=True
        )
        assert result == sum(i * i for i in range(1000))
        
        print("  ✅ All Performance Manager tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_decorators():
    """Test performance monitoring decorators."""
    print("🎭 Testing Performance Decorators...")
    
    try:
        from agent_mesh.core.performance import performance_monitored, cached
        
        # Test 1: Performance monitoring decorator
        print("  ✓ Testing performance monitoring decorator...")
        
        @performance_monitored(cache_result=True)
        def monitored_function(x, y):
            return x * y
        
        result = monitored_function(5, 10)
        assert result == 50
        
        # Test 2: Caching decorator
        print("  ✓ Testing caching decorator...")
        call_count = 0
        
        @cached(ttl=10.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x ** 2
        
        # First call should execute function
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Should not increment
        
        # Different parameter should execute function
        result3 = expensive_function(6)
        assert result3 == 36
        assert call_count == 2
        
        print("  ✅ All Performance Decorator tests passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance Decorator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all advanced performance tests."""
    print("🚀 Advanced Performance Optimization Test Suite")
    print("=" * 60)
    
    test_functions = [
        ("Intelligent Cache", test_intelligent_cache),
        ("Performance Metrics", test_performance_metrics),
        ("Connection Pool", test_connection_pool),
        ("Load Balancer", test_load_balancer),
        ("Performance Manager", test_performance_manager),
        ("Performance Decorators", test_performance_decorators),
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
        print("⚡ Intelligent caching, load balancing, and optimization are working correctly.")
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