#!/usr/bin/env python3
"""Demonstration of Generation 3 scaling features.

Tests performance optimization, auto-scaling, and intelligent resource management.
"""

import sys
sys.path.append('src')

import asyncio
import random
import time
from uuid import uuid4

# Mock structlog if not available
try:
    import structlog
except ImportError:
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def critical(self, *args, **kwargs): pass
    
    class MockStructlog:
        def get_logger(self, *args, **kwargs):
            return MockLogger()
    
    structlog = MockStructlog()

# Import our Generation 3 components
from agent_mesh.core.performance_optimizer import (
    PerformanceOptimizer, AdaptiveCache, LoadBalancer, 
    ResourcePoolManager, PerformanceMetrics, LoadBalanceStrategy
)
from agent_mesh.core.auto_scaler import (
    AutoScaler, ScalingRule, ScalingTrigger, InstanceMetrics,
    ScalingAction, ResourceRequirements
)


async def demo_adaptive_cache():
    """Demonstrate adaptive caching system."""
    print("🧠 Testing Adaptive Cache System")
    print("-" * 40)
    
    cache = AdaptiveCache(max_size_mb=10.0)  # Small cache for demo
    await cache.start()
    
    # Test basic cache operations
    await cache.set("key1", "value1", ttl_seconds=5)
    await cache.set("key2", "value2", ttl_seconds=10)
    await cache.set("key3", "value3")  # No TTL
    
    # Test cache hits
    value1 = await cache.get("key1")
    value2 = await cache.get("key2")
    value_missing = await cache.get("nonexistent")
    
    print(f"✅ Cache operations completed")
    print(f"   key1: {value1}")
    print(f"   key2: {value2}")
    print(f"   missing key: {value_missing}")
    
    # Test cache statistics
    stats = cache.get_statistics()
    print(f"✅ Cache statistics:")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.2%}")
    print(f"   Entries: {stats['entries']}")
    
    # Fill cache to trigger eviction
    for i in range(100):
        await cache.set(f"bulk_key_{i}", f"bulk_value_{i}" * 100)  # Larger values
        
    post_eviction_stats = cache.get_statistics()
    print(f"✅ Cache after bulk insertion:")
    print(f"   Entries: {post_eviction_stats['entries']}")
    print(f"   Evictions: {post_eviction_stats['evictions']}")
    print(f"   Utilization: {post_eviction_stats['utilization']:.1%}")
    
    await cache.stop()
    print()


async def demo_load_balancing():
    """Demonstrate intelligent load balancing."""
    print("⚖️ Testing Load Balancing System")  
    print("-" * 40)
    
    load_balancer = LoadBalancer(strategy=LoadBalanceStrategy.PERFORMANCE_BASED)
    
    # Register nodes with different capabilities
    nodes = []
    for i in range(5):
        node_id = uuid4()
        nodes.append(node_id)
        weight = 1.0 + random.uniform(-0.3, 0.5)  # Varied weights
        load_balancer.register_node(node_id, weight=weight)
        
        # Update with simulated performance metrics
        metrics = PerformanceMetrics(
            cpu_usage=random.uniform(20, 80),
            memory_usage=random.uniform(30, 90),
            network_latency_ms=random.uniform(5, 50),
            throughput_ops_sec=random.uniform(100, 1000)
        )
        load_balancer.update_node_performance(node_id, metrics)
        
    print(f"✅ Registered {len(nodes)} nodes")
    
    # Test different load balancing strategies
    strategies = [
        LoadBalanceStrategy.ROUND_ROBIN,
        LoadBalanceStrategy.LEAST_CONNECTIONS,
        LoadBalanceStrategy.PERFORMANCE_BASED
    ]
    
    for strategy in strategies:
        load_balancer.strategy = strategy
        selections = {}
        
        # Make 20 selections
        for i in range(20):
            selected = load_balancer.select_node(f"request_{i}")
            if selected:
                selections[selected] = selections.get(selected, 0) + 1
                load_balancer.record_connection(selected)
                
        print(f"✅ {strategy.value} distribution:")
        for node_id, count in selections.items():
            print(f"   Node {str(node_id)[:8]}: {count} selections")
            
    # Get load statistics
    stats = load_balancer.get_load_statistics()
    print(f"✅ Load balancer statistics:")
    print(f"   Total nodes: {stats['total_nodes']}")
    print(f"   Total connections: {stats['total_connections']}")
    print(f"   Strategy: {stats['strategy']}")
    print()


async def demo_resource_pooling():
    """Demonstrate resource pooling."""
    print("🏊 Testing Resource Pool Management")
    print("-" * 40)
    
    pool_manager = ResourcePoolManager()
    await pool_manager.start()
    
    # Create a mock resource creation function
    connection_counter = 0
    
    async def create_connection():
        nonlocal connection_counter
        connection_counter += 1
        await asyncio.sleep(0.1)  # Simulate connection time
        return f"connection_{connection_counter}"
        
    async def destroy_connection(conn):
        await asyncio.sleep(0.05)  # Simulate cleanup time
        
    # Create a connection pool
    pool = pool_manager.create_pool(
        "database_connections",
        create_func=create_connection,
        destroy_func=destroy_connection,
        max_size=10,
        min_size=2
    )
    
    print(f"✅ Created resource pool: {pool.pool_name}")
    
    # Test resource acquisition and release
    acquired_resources = []
    
    # Acquire multiple resources
    for i in range(5):
        resource = await pool_manager.acquire_resource("database_connections")
        if resource:
            acquired_resources.append(resource)
            print(f"   Acquired: {resource}")
            
    # Check pool statistics
    stats = pool_manager.get_pool_statistics("database_connections")
    print(f"✅ Pool statistics after acquisition:")
    print(f"   Active resources: {stats['active_resources']}")
    print(f"   Idle resources: {stats['idle_resources']}")
    print(f"   Utilization: {stats['utilization']:.1%}")
    
    # Release resources
    for resource in acquired_resources[:3]:  # Release some
        await pool_manager.release_resource("database_connections", resource)
        print(f"   Released: {resource}")
        
    # Check updated statistics
    updated_stats = pool_manager.get_pool_statistics("database_connections")
    print(f"✅ Pool statistics after release:")
    print(f"   Active resources: {updated_stats['active_resources']}")
    print(f"   Idle resources: {updated_stats['idle_resources']}")
    
    await pool_manager.stop()
    print()


async def demo_performance_optimization():
    """Demonstrate complete performance optimization."""
    print("🚀 Testing Performance Optimization System")
    print("-" * 40)
    
    node_id = uuid4()
    optimizer = PerformanceOptimizer(node_id)
    
    await optimizer.start()
    print("✅ Performance optimizer started")
    
    # Register nodes with load balancer
    for i in range(3):
        node_id = uuid4()
        optimizer.load_balancer.register_node(node_id, weight=1.0)
        
        # Simulate performance metrics
        metrics = PerformanceMetrics(
            cpu_usage=random.uniform(40, 80),
            memory_usage=random.uniform(50, 85),
            throughput_ops_sec=random.uniform(200, 800)
        )
        
        optimizer.load_balancer.update_node_performance(node_id, metrics)
        
    # Test performance analysis
    test_metrics = PerformanceMetrics(
        cpu_usage=85.0,  # High CPU
        memory_usage=90.0,  # High memory
        network_latency_ms=150.0,  # High latency
        error_rate=0.08  # High error rate
    )
    
    suggestions = optimizer.analyze_performance(test_metrics)
    print(f"✅ Performance analysis completed")
    print(f"   Suggestions: {len(suggestions)}")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
        
    # Get optimization report
    report = optimizer.get_optimization_report()
    print(f"✅ Optimization report generated")
    print(f"   Cache hit rate: {report['cache']['hit_rate']:.2%}")
    print(f"   Load balancer nodes: {report['load_balancing']['total_nodes']}")
    print(f"   Resource pools: {len(report['resource_pools'])}")
    
    await optimizer.stop()
    print()


async def demo_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    print("📈 Testing Auto-Scaling System")
    print("-" * 40)
    
    scaler = AutoScaler("test_cluster")
    
    # Set instance limits
    scaler.set_instance_limits(min_instances=2, max_instances=10)
    
    # Add scaling rules
    cpu_rule = ScalingRule(
        rule_id="cpu_scale_out",
        trigger=ScalingTrigger.CPU_UTILIZATION,
        metric_name="cpu_utilization",
        threshold_high=75.0,
        threshold_low=25.0,
        cooldown_seconds=60.0,  # Short cooldown for demo
        scaling_adjustment=2
    )
    scaler.add_scaling_rule(cpu_rule)
    
    memory_rule = ScalingRule(
        rule_id="memory_scale_out", 
        trigger=ScalingTrigger.MEMORY_UTILIZATION,
        metric_name="memory_utilization",
        threshold_high=80.0,
        threshold_low=30.0,
        scaling_adjustment=1
    )
    scaler.add_scaling_rule(memory_rule)
    
    print(f"✅ Added {len(scaler.scaling_rules)} scaling rules")
    
    # Set up scaling callbacks (simulated)
    scaled_out_count = 0
    scaled_in_instances = []
    
    async def mock_scale_out(count):
        nonlocal scaled_out_count
        scaled_out_count += count
        print(f"   📈 Scaled out: +{count} instances")
        return True
        
    async def mock_scale_in(instance_ids):
        nonlocal scaled_in_instances
        scaled_in_instances.extend(instance_ids)
        print(f"   📉 Scaled in: -{len(instance_ids)} instances")
        return True
        
    scaler.set_scaling_callbacks(mock_scale_out, mock_scale_in)
    
    # Start scaler
    await scaler.start()
    
    # Simulate instance metrics that trigger scaling
    for i in range(3):
        instance_id = uuid4()
        
        # High utilization metrics
        metrics = InstanceMetrics(
            instance_id=instance_id,
            cpu_utilization=85.0,  # Above threshold
            memory_utilization=85.0,  # Above threshold
            requests_per_second=150.0,
            response_time_ms=200.0
        )
        
        scaler.update_instance_metrics(instance_id, metrics)
        print(f"   Updated metrics for instance {str(instance_id)[:8]}")
        
    # Trigger immediate evaluation
    await scaler.trigger_immediate_evaluation()
    
    # Get scaling status
    status = scaler.get_scaling_status()
    print(f"✅ Scaling status:")
    print(f"   Current instances: {status['current_instances']}")
    print(f"   Average CPU: {status['average_cpu_utilization']:.1f}%")
    print(f"   Average memory: {status['average_memory_utilization']:.1f}%")
    print(f"   Active rules: {status['active_rules']}")
    
    # Test predictive insights
    insights = scaler.get_predictive_insights()
    print(f"✅ Predictive insights:")
    if insights['recommendations']:
        for rec in insights['recommendations']:
            print(f"   • {rec}")
    else:
        print("   • No specific recommendations at this time")
        
    await scaler.stop()
    print(f"✅ Scaling actions taken:")
    print(f"   Scale out events: {scaled_out_count}")
    print(f"   Scale in events: {len(scaled_in_instances)}")
    print()


async def demo_integrated_scaling():
    """Demonstrate integrated scaling system."""
    print("🌐 Testing Integrated Scaling System")
    print("-" * 40)
    
    # Create integrated system
    node_id = uuid4()
    optimizer = PerformanceOptimizer(node_id)
    scaler = AutoScaler("integrated_cluster")
    
    await optimizer.start()
    await scaler.start()
    
    # Set up integration
    scaler.set_instance_limits(1, 8)
    
    # Add comprehensive scaling rule
    integrated_rule = ScalingRule(
        rule_id="integrated_scaling",
        trigger=ScalingTrigger.CPU_UTILIZATION,
        metric_name="cpu_utilization",
        threshold_high=70.0,
        threshold_low=20.0,
        scaling_adjustment=1,
        cooldown_seconds=30.0
    )
    scaler.add_scaling_rule(integrated_rule)
    
    print("✅ Integrated system initialized")
    
    # Simulate load scenarios
    scenarios = [
        {"name": "Low Load", "cpu": 15.0, "memory": 25.0, "rps": 50},
        {"name": "Normal Load", "cpu": 45.0, "memory": 55.0, "rps": 200},
        {"name": "High Load", "cpu": 80.0, "memory": 85.0, "rps": 500},
        {"name": "Peak Load", "cpu": 95.0, "memory": 95.0, "rps": 800},
    ]
    
    for scenario in scenarios:
        print(f"\n🔄 Simulating {scenario['name']}:")
        
        # Update metrics for scenario
        instance_id = uuid4()
        metrics = InstanceMetrics(
            instance_id=instance_id,
            cpu_utilization=scenario["cpu"],
            memory_utilization=scenario["memory"],
            requests_per_second=scenario["rps"],
            response_time_ms=50 + (scenario["rps"] / 10)
        )
        
        scaler.update_instance_metrics(instance_id, metrics)
        
        # Get performance analysis
        perf_metrics = PerformanceMetrics(
            cpu_usage=scenario["cpu"],
            memory_usage=scenario["memory"],
            throughput_ops_sec=scenario["rps"]
        )
        
        suggestions = optimizer.analyze_performance(perf_metrics)
        
        print(f"   CPU: {scenario['cpu']}%, Memory: {scenario['memory']}%, RPS: {scenario['rps']}")
        print(f"   Performance suggestions: {len(suggestions)}")
        for suggestion in suggestions[:2]:  # Show first 2
            print(f"     • {suggestion}")
            
        # Brief pause between scenarios
        await asyncio.sleep(0.5)
        
    # Get final status
    optimizer_report = optimizer.get_optimization_report()
    scaler_status = scaler.get_scaling_status()
    
    print(f"\n✅ Integrated system final status:")
    print(f"   Cache utilization: {optimizer_report['cache']['utilization']:.1%}")
    print(f"   Active scaling rules: {scaler_status['active_rules']}")
    print(f"   Current cluster size: {scaler_status['current_instances']} instances")
    
    await optimizer.stop()
    await scaler.stop()
    print()


async def main():
    """Run all Generation 3 demonstrations."""
    print("⚡ Agent Mesh Generation 3: MAKE IT SCALE")
    print("=" * 50)
    print("Testing performance optimization and intelligent scaling")
    print()
    
    try:
        await demo_adaptive_cache()
        await demo_load_balancing()
        await demo_resource_pooling()
        await demo_performance_optimization()
        await demo_auto_scaling()
        await demo_integrated_scaling()
        
        print("🎉 Generation 3 Features Successfully Demonstrated!")
        print("=" * 50)
        print("✅ Adaptive Caching: Intelligent cache management with multiple policies")
        print("✅ Load Balancing: Performance-based traffic distribution")
        print("✅ Resource Pooling: Efficient connection and object pooling")
        print("✅ Performance Optimization: Real-time analysis and recommendations")
        print("✅ Auto-Scaling: Predictive and reactive scaling capabilities")
        print("✅ Integration: Seamless coordination between optimization components")
        print()
        print("Generation 3 implementation delivers enterprise-scale performance!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    import sys
    result = asyncio.run(main())
    sys.exit(result)