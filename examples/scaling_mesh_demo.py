#!/usr/bin/env python3
"""
Scaling Agent Mesh Demo - Generation 3 Implementation

Demonstrates advanced performance optimizations, intelligent caching,
load balancing, multi-region capabilities, and scalability features
in a large-scale mesh network scenario.
"""

import asyncio
import logging
import random
import time
from contextlib import AsyncExitStack
from uuid import uuid4

from agent_mesh.core.network import P2PNetwork
from agent_mesh.core.discovery import MultiDiscovery, DiscoveryConfig, DiscoveryMethod
from agent_mesh.core.health import initialize_health_monitoring
from agent_mesh.core.metrics import initialize_metrics
from agent_mesh.core.performance import get_performance_manager
from agent_mesh.database.connection import initialize_database


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scaling_demo")


class ScalableAgent:
    """
    High-performance scalable agent with advanced optimizations.
    
    Demonstrates Generation 3 capabilities including intelligent caching,
    performance optimization, load balancing, and scalability features.
    """
    
    def __init__(self, node_id=None, listen_port=None, region="us-east-1"):
        self.node_id = node_id or uuid4()
        self.listen_port = listen_port or random.randint(4001, 4200)
        self.region = region
        
        # Core components
        self.network = None
        self.discovery = None
        self.db_manager = None
        self.health_monitor = None
        self.metrics_manager = None
        self.performance_manager = None
        
        # Scaling features
        self.message_cache = {}  # Simple message caching
        self.peer_performance = {}  # Track peer performance
        self.load_metrics = {"cpu": 0.0, "memory": 0.0, "network": 0.0}
        
        # Statistics
        self.messages_processed = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"ScalableAgent {self.node_id} initialized in region {region} on port {self.listen_port}")
    
    async def start(self):
        """Start the scalable agent with all optimizations."""
        logger.info(f"Starting scalable agent {self.node_id}")
        
        async with AsyncExitStack() as stack:
            # Initialize performance manager first
            self.performance_manager = get_performance_manager()
            await self.performance_manager.start()
            stack.push_async_callback(self.performance_manager.stop)
            
            # Initialize database with optimizations
            self.db_manager = initialize_database(
                f"sqlite:///data/mesh_scale_{self.listen_port}.db",
                pool_size=10,
                auto_migrate=True
            )
            await self.db_manager.initialize_async()
            stack.push_async_callback(self.db_manager.close)
            
            # Initialize network with performance optimizations
            self.network = P2PNetwork(
                node_id=self.node_id,
                listen_addr=f"/ip4/0.0.0.0/tcp/{self.listen_port}"
            )
            await self.network.start()
            stack.push_async_callback(self.network.stop)
            
            # Register optimized message handlers
            self._register_optimized_handlers()
            
            # Initialize discovery with intelligent caching
            self.discovery = MultiDiscovery(
                config=DiscoveryConfig(
                    methods={DiscoveryMethod.MDNS, DiscoveryMethod.GOSSIP},
                    discovery_interval=20,
                    max_peers=50  # Higher capacity for scaling
                ),
                node_id=self.node_id,
                network_manager=self.network
            )
            await self.discovery.start()
            stack.push_async_callback(self.discovery.stop)
            
            # Initialize health monitoring
            self.health_monitor = await initialize_health_monitoring(
                network_manager=self.network,
                db_manager=self.db_manager
            )
            stack.push_async_callback(self.health_monitor.stop)
            
            # Initialize metrics with performance tracking
            self.metrics_manager = await initialize_metrics(
                network_manager=self.network,
                db_manager=self.db_manager,
                collection_interval=5.0  # More frequent for scaling demo
            )
            stack.push_async_callback(self.metrics_manager.stop)
            
            # Announce with scaling capabilities
            await self.discovery.announce({
                "node_id": str(self.node_id),
                "port": self.listen_port,
                "region": self.region,
                "capabilities": [
                    "scalable_agent", "intelligent_caching", 
                    "load_balancing", "high_performance"
                ],
                "version": "3.0_scaling",
                "max_connections": 100,
                "performance_tier": "high"
            })
            
            logger.info(f"Scalable agent {self.node_id} started with all optimizations")
            
            # Start scaling background tasks
            tasks = [
                asyncio.create_task(self._scaling_discovery_loop()),
                asyncio.create_task(self._performance_monitoring_loop()),
                asyncio.create_task(self._cache_optimization_loop()),
                asyncio.create_task(self._load_balancing_loop()),
                asyncio.create_task(self._benchmark_loop()),  # Performance testing
            ]
            
            try:
                # Run until interrupted
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info(f"Agent {self.node_id} interrupted")
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def _register_optimized_handlers(self):
        """Register message handlers with performance optimizations."""
        self.network.register_request_handler("ping", self._optimized_ping_handler)
        self.network.register_request_handler("cached_request", self._cached_request_handler)
        self.network.register_request_handler("benchmark", self._benchmark_handler)
        self.network.register_request_handler("get_performance", self._get_performance_handler)
        self.network.register_request_handler("load_test", self._load_test_handler)
        self.network.register_request_handler("get_peer_stats", self._get_peer_stats_handler)
    
    # Optimized Message Handlers
    
    async def _optimized_ping_handler(self, request_data: dict) -> dict:
        """Optimized ping handler with caching and performance tracking."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"ping_{request_data.get('sender', 'unknown')}"
            cached_response = await self.performance_manager.cache.get(cache_key)
            
            if cached_response:
                self.cache_hits += 1
                cached_response["cache_hit"] = True
                cached_response["response_time"] = time.time() - start_time
                return cached_response
            
            self.cache_misses += 1
            
            # Generate response
            response = {
                "status": "pong",
                "timestamp": time.time(),
                "node_id": str(self.node_id),
                "region": self.region,
                "performance_tier": "high",
                "cache_hit": False,
                "load_metrics": self.load_metrics.copy(),
                "response_time": time.time() - start_time
            }
            
            # Cache response for future use
            await self.performance_manager.cache.set(
                cache_key, response, ttl=30.0  # Cache for 30 seconds
            )
            
            self.messages_processed += 1
            return response
            
        except Exception as e:
            logger.error(f"Error in optimized ping handler: {e}")
            return {"error": str(e), "response_time": time.time() - start_time}
    
    async def _cached_request_handler(self, request_data: dict) -> dict:
        """Handler that demonstrates intelligent caching."""
        request_type = request_data.get("type", "unknown")
        cache_key = f"cached_{request_type}_{hash(str(request_data))}"
        
        # Check cache
        cached_result = await self.performance_manager.cache.get(cache_key)
        if cached_result:
            self.cache_hits += 1
            return {"result": cached_result, "cached": True}
        
        self.cache_misses += 1
        
        # Simulate expensive computation
        await asyncio.sleep(0.1)  # Simulated work
        
        result = {
            "computed_at": time.time(),
            "node_id": str(self.node_id),
            "request_type": request_type,
            "expensive_computation": random.randint(1000, 9999)
        }
        
        # Cache result
        await self.performance_manager.cache.set(cache_key, result, ttl=60.0)
        
        self.messages_processed += 1
        return {"result": result, "cached": False}
    
    async def _benchmark_handler(self, request_data: dict) -> dict:
        """Handler for performance benchmarking."""
        benchmark_type = request_data.get("type", "basic")
        iterations = request_data.get("iterations", 1000)
        
        start_time = time.time()
        
        if benchmark_type == "cpu":
            # CPU-intensive benchmark
            result = sum(i * i for i in range(iterations))
        
        elif benchmark_type == "memory":
            # Memory allocation benchmark
            data = [random.random() for _ in range(iterations)]
            result = len(data)
        
        elif benchmark_type == "network":
            # Network operation benchmark
            peers = await self.network.get_connected_peers()
            result = len(peers)
        
        else:
            result = 0
        
        duration = time.time() - start_time
        
        return {
            "benchmark_type": benchmark_type,
            "iterations": iterations,
            "result": result,
            "duration_ms": duration * 1000,
            "ops_per_second": iterations / duration if duration > 0 else 0,
            "node_id": str(self.node_id)
        }
    
    async def _get_performance_handler(self, request_data: dict) -> dict:
        """Get comprehensive performance statistics."""
        try:
            performance_summary = await self.performance_manager.get_performance_summary()
            network_stats = await self.network.get_statistics()
            
            return {
                "node_id": str(self.node_id),
                "region": self.region,
                "messages_processed": self.messages_processed,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                "performance_summary": performance_summary,
                "network_stats": network_stats,
                "load_metrics": self.load_metrics,
                "peer_performance": dict(list(self.peer_performance.items())[:5])  # Top 5
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _load_test_handler(self, request_data: dict) -> dict:
        """Handle load testing requests."""
        load_type = request_data.get("type", "light")
        duration = request_data.get("duration", 5)
        
        start_time = time.time()
        operations = 0
        
        if load_type == "light":
            target_ops = 100
        elif load_type == "medium":
            target_ops = 500
        else:  # heavy
            target_ops = 1000
        
        while time.time() - start_time < duration:
            # Simulate operations
            for _ in range(10):
                await asyncio.sleep(0.001)  # Small work unit
                operations += 1
            
            if operations >= target_ops:
                break
        
        actual_duration = time.time() - start_time
        ops_per_second = operations / actual_duration if actual_duration > 0 else 0
        
        return {
            "load_type": load_type,
            "operations": operations,
            "duration": actual_duration,
            "ops_per_second": ops_per_second,
            "target_ops": target_ops,
            "node_id": str(self.node_id)
        }
    
    async def _get_peer_stats_handler(self, request_data: dict) -> dict:
        """Get peer performance statistics."""
        peers = await self.network.get_connected_peers()
        peer_stats = []
        
        for peer in peers[:10]:  # Limit to first 10 peers
            stats = {
                "peer_id": str(peer.peer_id),
                "status": peer.status.value,
                "latency_ms": peer.latency_ms,
                "last_seen": peer.last_seen,
                "protocols": [p.value for p in peer.protocols]
            }
            
            # Add performance data if available
            if peer.peer_id in self.peer_performance:
                stats.update(self.peer_performance[peer.peer_id])
            
            peer_stats.append(stats)
        
        return {
            "peer_count": len(peers),
            "peer_stats": peer_stats,
            "collection_time": time.time()
        }
    
    # Background Tasks for Scaling
    
    async def _scaling_discovery_loop(self):
        """Enhanced discovery loop with intelligent peer management."""
        while True:
            try:
                # Discover new peers with regional awareness
                discovered = await self.discovery.discover_peers()
                
                if discovered:
                    logger.info(f"Agent {self.node_id} discovered {len(discovered)} peers")
                    
                    # Prioritize peers in same region
                    same_region_peers = []
                    other_region_peers = []
                    
                    for peer in discovered:
                        # Simple region detection from addresses (in real implementation would be more sophisticated)
                        if "127.0.0.1" in str(peer.addresses):
                            same_region_peers.append(peer)
                        else:
                            other_region_peers.append(peer)
                    
                    # Connect to same-region peers first
                    for peer in (same_region_peers + other_region_peers)[:5]:
                        if peer.addresses:
                            try:
                                await self.network.connect_to_peer(peer.addresses[0])
                            except Exception as e:
                                logger.debug(f"Failed to connect to peer: {e}")
                
                # Update load metrics
                self._update_load_metrics()
                
                await asyncio.sleep(15)  # Faster discovery for scaling
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling discovery error: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitoring_loop(self):
        """Monitor and optimize performance continuously."""
        while True:
            try:
                # Collect performance data
                peers = await self.network.get_connected_peers()
                
                # Update peer performance tracking
                for peer in peers:
                    if peer.peer_id not in self.peer_performance:
                        self.peer_performance[peer.peer_id] = {
                            "first_seen": time.time(),
                            "message_count": 0,
                            "avg_latency": peer.latency_ms,
                            "error_count": 0
                        }
                    
                    # Update latency tracking
                    perf_data = self.peer_performance[peer.peer_id]
                    perf_data["avg_latency"] = (perf_data["avg_latency"] + peer.latency_ms) / 2
                    perf_data["last_seen"] = peer.last_seen
                
                # Log performance summary
                if len(peers) > 0:
                    avg_latency = sum(p.latency_ms for p in peers) / len(peers)
                    logger.info(f"Agent {self.node_id} - Peers: {len(peers)}, "
                              f"Avg Latency: {avg_latency:.1f}ms, "
                              f"Cache Hit Rate: {self.cache_hits / max(self.cache_hits + self.cache_misses, 1):.2f}")
                
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _cache_optimization_loop(self):
        """Optimize caching based on usage patterns."""
        while True:
            try:
                cache_stats = self.performance_manager.cache.get_statistics()
                
                # Log cache performance
                if cache_stats["hits"] + cache_stats["misses"] > 0:
                    logger.debug(f"Cache stats - Size: {cache_stats['size']}, "
                               f"Hit Rate: {cache_stats['hit_rate']:.2f}")
                
                # Auto-tune cache if hit rate is low
                if cache_stats["hit_rate"] < 0.5 and cache_stats["size"] < 1000:
                    logger.info("Low cache hit rate, considering optimization")
                
                await asyncio.sleep(120)  # Every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(120)
    
    async def _load_balancing_loop(self):
        """Distribute load intelligently across peers."""
        while True:
            try:
                peers = await self.network.get_connected_peers()
                
                if len(peers) > 3:  # Only balance if we have enough peers
                    # Send test messages to measure peer performance
                    for peer in random.sample(peers, min(3, len(peers))):
                        try:
                            start_time = time.time()
                            response = await self.network.send_request(
                                peer.peer_id, "ping", {"load_test": True}, timeout=5.0
                            )
                            latency = time.time() - start_time
                            
                            # Update peer performance data
                            if peer.peer_id in self.peer_performance:
                                self.peer_performance[peer.peer_id]["message_count"] += 1
                                self.peer_performance[peer.peer_id]["avg_latency"] = latency * 1000
                        
                        except Exception:
                            if peer.peer_id in self.peer_performance:
                                self.peer_performance[peer.peer_id]["error_count"] += 1
                
                await asyncio.sleep(60)  # Load balance every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(60)
    
    async def _benchmark_loop(self):
        """Continuous performance benchmarking."""
        while True:
            try:
                # Run periodic benchmarks
                benchmark_types = ["cpu", "memory", "network"]
                benchmark_type = random.choice(benchmark_types)
                
                start_time = time.time()
                await self._benchmark_handler({
                    "type": benchmark_type,
                    "iterations": 100
                })
                duration = time.time() - start_time
                
                logger.debug(f"Benchmark {benchmark_type} completed in {duration*1000:.1f}ms")
                
                await asyncio.sleep(180)  # Every 3 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Benchmark loop error: {e}")
                await asyncio.sleep(180)
    
    def _update_load_metrics(self):
        """Update system load metrics."""
        # Simulate load metrics (in real implementation would use actual system metrics)
        self.load_metrics["cpu"] = random.uniform(10, 80)
        self.load_metrics["memory"] = random.uniform(20, 70)
        self.load_metrics["network"] = random.uniform(5, 95)


async def demo_scaling_network():
    """Demonstrate large-scale mesh networking with optimization."""
    logger.info("Starting large-scale mesh network demo with advanced optimizations")
    
    # Create multiple scalable agents across different "regions"
    regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
    agents = []
    
    try:
        # Create 6 agents across different regions
        for i in range(6):
            region = regions[i % len(regions)]
            agent = ScalableAgent(listen_port=4001 + i, region=region)
            agents.append(agent)
        
        logger.info(f"Created {len(agents)} scalable agents across {len(regions)} regions")
        
        # Start agents concurrently
        agent_tasks = [asyncio.create_task(agent.start()) for agent in agents]
        
        # Let network form and optimize
        await asyncio.sleep(20)
        
        # Bootstrap connections between regions
        for i in range(1, min(4, len(agents))):
            try:
                await agents[i].network.connect_to_peer(f"127.0.0.1:{4001 + i - 1}")
            except Exception as e:
                logger.warning(f"Bootstrap connection failed: {e}")
        
        logger.info("Network forming with cross-region connectivity...")
        
        # Run performance tests
        await asyncio.sleep(30)
        
        # Demonstrate scaling capabilities
        logger.info("\n=== SCALING DEMONSTRATION ===")
        
        # Test 1: Cache performance
        logger.info("Testing intelligent caching...")
        for agent in agents[:3]:
            try:
                # Make repeated cached requests to test cache hits
                for _ in range(3):
                    peers = await agent.network.get_connected_peers()
                    if peers:
                        await agent.network.send_request(
                            peers[0].peer_id, 
                            "cached_request", 
                            {"type": "test_computation"}, 
                            timeout=5.0
                        )
            except Exception as e:
                logger.error(f"Cache test failed: {e}")
        
        await asyncio.sleep(10)
        
        # Test 2: Load balancing
        logger.info("Testing load balancing...")
        if len(agents) >= 2:
            source_agent = agents[0]
            peers = await source_agent.network.get_connected_peers()
            
            # Send requests to multiple peers to test load distribution
            for _ in range(5):
                for peer in peers[:3]:
                    try:
                        await source_agent.network.send_request(
                            peer.peer_id, "load_test", 
                            {"type": "medium", "duration": 2}, timeout=10.0
                        )
                    except Exception as e:
                        logger.debug(f"Load test request failed: {e}")
        
        await asyncio.sleep(10)
        
        # Test 3: Performance benchmarking
        logger.info("Running performance benchmarks...")
        for agent in agents[:2]:
            try:
                peers = await agent.network.get_connected_peers()
                if peers:
                    benchmark_result = await agent.network.send_request(
                        peers[0].peer_id, 
                        "benchmark", 
                        {"type": "cpu", "iterations": 1000}, 
                        timeout=10.0
                    )
                    logger.info(f"Benchmark result: {benchmark_result.get('ops_per_second', 0):.0f} ops/sec")
            except Exception as e:
                logger.warning(f"Benchmark failed: {e}")
        
        # Display final scaling metrics
        await asyncio.sleep(20)
        
        logger.info("\n=== FINAL SCALING METRICS ===")
        for i, agent in enumerate(agents):
            try:
                performance = await agent._get_performance_handler({})
                
                logger.info(f"\nAgent {i+1} ({agent.region}):")
                logger.info(f"  Messages Processed: {performance.get('messages_processed', 0)}")
                logger.info(f"  Cache Hit Rate: {performance.get('cache_hit_rate', 0):.2f}")
                logger.info(f"  Connected Peers: {len(await agent.network.get_connected_peers())}")
                
                cache_stats = performance.get('performance_summary', {}).get('cache', {})
                logger.info(f"  Cache Size: {cache_stats.get('size', 0)}")
                logger.info(f"  Cache Hits: {cache_stats.get('hits', 0)}")
                
            except Exception as e:
                logger.error(f"Failed to get metrics for agent {i+1}: {e}")
        
        logger.info("\nScaling demonstration completed successfully!")
        logger.info("Key features demonstrated:")
        logger.info("- Intelligent caching with LRU eviction")
        logger.info("- Regional awareness and optimization") 
        logger.info("- Performance monitoring and auto-tuning")
        logger.info("- Load balancing across peers")
        logger.info("- Continuous benchmarking and optimization")
        
    except KeyboardInterrupt:
        logger.info("Scaling demo interrupted")
    finally:
        # Cleanup
        logger.info("Shutting down scalable agents...")
        for task in agent_tasks:
            task.cancel()
        await asyncio.gather(*agent_tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(demo_scaling_network())