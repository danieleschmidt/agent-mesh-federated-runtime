#!/usr/bin/env python3
"""
Robust Agent Mesh Demo - Generation 2 Implementation

Demonstrates enhanced error handling, health monitoring, circuit breakers,
metrics collection, and self-healing capabilities in the mesh network.
"""

import asyncio
import logging
import random
import signal
import time
from contextlib import AsyncExitStack
from uuid import uuid4

from agent_mesh.core.network import P2PNetwork
from agent_mesh.core.discovery import MultiDiscovery, DiscoveryConfig, DiscoveryMethod
from agent_mesh.core.health import initialize_health_monitoring, get_health_monitor
from agent_mesh.core.metrics import initialize_metrics, get_metrics_manager
from agent_mesh.database.connection import initialize_database


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("robust_mesh_demo")


class RobustAgent:
    """
    Robust agent with comprehensive error handling and monitoring.
    
    Demonstrates Generation 2 capabilities including health monitoring,
    circuit breakers, metrics collection, and graceful failure handling.
    """
    
    def __init__(self, node_id=None, listen_port=None):
        self.node_id = node_id or uuid4()
        self.listen_port = listen_port or random.randint(4001, 4100)
        
        # Core components
        self.network = None
        self.discovery = None
        self.db_manager = None
        self.health_monitor = None
        self.metrics_manager = None
        
        # Agent state
        self.running = False
        self.start_time = 0
        self.message_count = 0
        self.error_count = 0
        
        # Graceful shutdown
        self._shutdown_event = asyncio.Event()
        self._tasks = []
        
        logger.info(f"RobustAgent {self.node_id} initialized on port {self.listen_port}")
    
    async def start(self):
        """Start the robust agent with full monitoring."""
        logger.info(f"Starting robust agent {self.node_id}")
        self.start_time = time.time()
        
        try:
            async with AsyncExitStack() as stack:
                # Initialize database with connection pooling
                self.db_manager = initialize_database(
                    f"sqlite:///data/mesh_robust_{self.listen_port}.db",
                    pool_size=5,
                    auto_migrate=True
                )
                await self.db_manager.initialize_async()
                stack.push_async_callback(self.db_manager.close)
                
                # Initialize network with enhanced error handling
                self.network = P2PNetwork(
                    node_id=self.node_id,
                    listen_addr=f"/ip4/0.0.0.0/tcp/{self.listen_port}"
                )
                await self.network.start()
                stack.push_async_callback(self.network.stop)
                
                # Register message handlers with error handling
                self._register_message_handlers()
                
                # Initialize peer discovery
                self.discovery = MultiDiscovery(
                    config=DiscoveryConfig(
                        methods={DiscoveryMethod.MDNS, DiscoveryMethod.GOSSIP},
                        discovery_interval=15,
                        max_peers=20
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
                
                # Initialize metrics collection
                self.metrics_manager = await initialize_metrics(
                    network_manager=self.network,
                    db_manager=self.db_manager,
                    collection_interval=10.0
                )
                stack.push_async_callback(self.metrics_manager.stop)
                
                # Add health monitoring callbacks
                self.health_monitor.add_health_callback(self._on_health_change)
                
                # Announce ourselves
                await self.discovery.announce({
                    "node_id": str(self.node_id),
                    "port": self.listen_port,
                    "capabilities": ["robust_agent", "health_monitoring", "metrics"],
                    "version": "2.0"
                })
                
                self.running = True
                logger.info(f"Robust agent {self.node_id} started successfully")
                
                # Start background tasks
                self._start_background_tasks()
                
                # Setup signal handlers for graceful shutdown
                self._setup_signal_handlers()
                
                # Wait for shutdown signal
                await self._shutdown_event.wait()
                
        except Exception as e:
            logger.error(f"Failed to start robust agent: {e}", exc_info=True)
            self.error_count += 1
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the agent gracefully."""
        logger.info(f"Stopping robust agent {self.node_id}")
        self.running = False
        
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Log final statistics
        uptime = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Agent {self.node_id} stopped - Uptime: {uptime:.1f}s, "
                   f"Messages: {self.message_count}, Errors: {self.error_count}")
    
    def _register_message_handlers(self):
        """Register message handlers with error handling."""
        self.network.register_request_handler("ping", self._handle_ping)
        self.network.register_request_handler("get_peer_list", self._handle_get_peer_list)
        self.network.register_request_handler("get_status", self._handle_get_status)
        self.network.register_request_handler("get_health", self._handle_get_health)
        self.network.register_request_handler("get_metrics", self._handle_get_metrics)
        self.network.register_request_handler("simulate_error", self._handle_simulate_error)
    
    def _start_background_tasks(self):
        """Start background tasks."""
        self._tasks = [
            asyncio.create_task(self._discovery_loop()),
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._status_monitoring_loop()),
            asyncio.create_task(self._error_injection_loop()),  # For demo purposes
        ]
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler():
            logger.info(f"Received shutdown signal for agent {self.node_id}")
            self._shutdown_event.set()
        
        try:
            # Only works on Unix systems
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, lambda s, f: signal_handler())
        except AttributeError:
            # Windows doesn't support these signals
            pass
    
    # Message Handlers
    
    async def _handle_ping(self, request_data: dict) -> dict:
        """Handle ping with error simulation."""
        try:
            self.message_count += 1
            
            # Simulate occasional network errors
            if random.random() < 0.05:  # 5% error rate
                raise Exception("Simulated network error in ping handler")
            
            return {
                "status": "pong",
                "timestamp": time.time(),
                "node_id": str(self.node_id),
                "uptime": time.time() - self.start_time,
                "message_count": self.message_count
            }
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in ping handler: {e}")
            raise
    
    async def _handle_get_status(self, request_data: dict) -> dict:
        """Get comprehensive agent status."""
        try:
            network_stats = await self.network.get_statistics()
            peers = await self.network.get_connected_peers()
            
            return {
                "node_id": str(self.node_id),
                "uptime": time.time() - self.start_time,
                "connected_peers": len(peers),
                "message_count": self.message_count,
                "error_count": self.error_count,
                "network_stats": network_stats,
                "health_status": self.health_monitor.get_system_health().value if self.health_monitor else "unknown"
            }
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def _handle_get_health(self, request_data: dict) -> dict:
        """Get health monitoring information."""
        try:
            if self.health_monitor:
                return self.health_monitor.get_health_summary()
            else:
                return {"error": "Health monitoring not available"}
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error getting health: {e}")
            return {"error": str(e)}
    
    async def _handle_get_metrics(self, request_data: dict) -> dict:
        """Get metrics information."""
        try:
            if self.metrics_manager:
                latest_metrics = self.metrics_manager.get_latest_metrics()
                return {
                    "latest_metrics": {name: {
                        "value": metric.value,
                        "timestamp": metric.timestamp,
                        "labels": metric.labels
                    } for name, metric in latest_metrics.items()},
                    "prometheus_format": self.metrics_manager.get_prometheus_metrics()
                }
            else:
                return {"error": "Metrics not available"}
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    async def _handle_simulate_error(self, request_data: dict) -> dict:
        """Simulate various types of errors for testing robustness."""
        error_type = request_data.get("error_type", "generic")
        
        if error_type == "timeout":
            await asyncio.sleep(35)  # Longer than typical timeout
        elif error_type == "exception":
            raise ValueError("Simulated exception for testing")
        elif error_type == "blacklist_peer":
            # Simulate blacklisting a peer
            peers = await self.network.get_connected_peers()
            if peers:
                peer = random.choice(peers)
                self.network.blacklist_peer(peer.peer_id, "Simulated blacklisting")
        
        self.error_count += 1
        return {"status": "error_simulated", "type": error_type}
    
    async def _handle_get_peer_list(self, request_data: dict) -> dict:
        """Handle peer list request with circuit breaker protection."""
        try:
            # Use circuit breaker for this operation
            circuit_breaker = self.health_monitor.circuit_breakers.get("network_operations")
            
            async def get_peers():
                peers = await self.network.get_connected_peers()
                return {
                    "peers": [
                        {
                            "peer_id": str(peer.peer_id),
                            "addresses": peer.addresses,
                            "protocols": [p.value for p in peer.protocols],
                            "last_seen": peer.last_seen,
                            "status": peer.status.value
                        }
                        for peer in peers
                    ]
                }
            
            if circuit_breaker:
                return await circuit_breaker.call(get_peers)
            else:
                return await get_peers()
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error getting peer list: {e}")
            return {"error": str(e)}
    
    # Background Tasks
    
    async def _discovery_loop(self):
        """Enhanced peer discovery with error handling."""
        while self.running:
            try:
                # Discover new peers
                discovered = await self.discovery.discover_peers()
                
                if discovered:
                    logger.info(f"Agent {self.node_id} discovered {len(discovered)} peers")
                    
                    # Try connecting to discovered peers with retry logic
                    for peer in discovered[:2]:  # Limit concurrent connections
                        if peer.addresses:
                            try:
                                await self.network.connect_to_peer(peer.addresses[0])
                            except Exception as e:
                                logger.warning(f"Failed to connect to discovered peer: {e}")
                
                await asyncio.sleep(20)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(30)
    
    async def _heartbeat_loop(self):
        """Enhanced heartbeat with health information."""
        while self.running:
            try:
                # Prepare heartbeat with health information
                health_status = "unknown"
                if self.health_monitor:
                    health_status = self.health_monitor.get_system_health().value
                
                heartbeat_data = {
                    "timestamp": time.time(),
                    "uptime": time.time() - self.start_time,
                    "health_status": health_status,
                    "message_count": self.message_count,
                    "error_count": self.error_count,
                    "node_version": "2.0_robust"
                }
                
                # Broadcast heartbeat
                await self.network.broadcast_message("heartbeat", heartbeat_data)
                
                await asyncio.sleep(45)  # Less frequent heartbeats
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(60)
    
    async def _status_monitoring_loop(self):
        """Monitor and log system status periodically."""
        while self.running:
            try:
                # Log comprehensive status
                status = await self._handle_get_status({})
                logger.info(f"Agent {self.node_id} Status - "
                          f"Peers: {status.get('connected_peers', 0)}, "
                          f"Messages: {status.get('message_count', 0)}, "
                          f"Errors: {status.get('error_count', 0)}, "
                          f"Health: {status.get('health_status', 'unknown')}")
                
                # Log network error statistics if available
                if hasattr(self.network, 'get_error_statistics'):
                    error_stats = self.network.get_error_statistics()
                    if error_stats['error_counts']:
                        logger.info(f"Network errors: {error_stats['error_counts']}")
                
                await asyncio.sleep(60)  # Status every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Status monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _error_injection_loop(self):
        """Inject errors periodically for testing robustness (demo only)."""
        while self.running:
            try:
                await asyncio.sleep(random.randint(120, 300))  # 2-5 minutes
                
                # Randomly inject errors for testing
                error_types = ["timeout", "exception", "blacklist_peer"]
                error_type = random.choice(error_types)
                
                logger.info(f"Injecting test error: {error_type}")
                try:
                    await self._handle_simulate_error({"error_type": error_type})
                except Exception:
                    pass  # Expected for some error types
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error injection loop error: {e}")
                await asyncio.sleep(300)
    
    # Callbacks
    
    def _on_health_change(self, health_results: dict):
        """Callback for health status changes."""
        unhealthy_components = [
            name for name, health in health_results.items()
            if health.get_worst_status().value in ["unhealthy", "critical"]
        ]
        
        if unhealthy_components:
            logger.warning(f"Agent {self.node_id} - Unhealthy components: {unhealthy_components}")
    
    def shutdown(self):
        """Trigger graceful shutdown."""
        self._shutdown_event.set()


async def demo_robust_network():
    """Demonstrate robust mesh networking with monitoring."""
    logger.info("Starting robust mesh network demo with monitoring")
    
    # Create multiple robust agents
    agents = []
    
    try:
        # Create 3 robust agents
        for i in range(3):
            agent = RobustAgent(listen_port=4001 + i)
            agents.append(agent)
        
        # Start agents with staggered startup
        agent_tasks = []
        for i, agent in enumerate(agents):
            await asyncio.sleep(2)  # Stagger startup
            task = asyncio.create_task(agent.start())
            agent_tasks.append(task)
            logger.info(f"Started agent {i+1}")
        
        logger.info(f"Created {len(agents)} robust agents with full monitoring")
        
        # Let them discover and connect to each other
        await asyncio.sleep(10)
        
        # Manually bootstrap some connections
        if len(agents) >= 2:
            try:
                await agents[1].network.connect_to_peer("127.0.0.1:4001")
                await agents[2].network.connect_to_peer("127.0.0.1:4002")
            except Exception as e:
                logger.warning(f"Bootstrap connection failed: {e}")
        
        # Let the robust network run and demonstrate self-healing
        logger.info("Network running with health monitoring, metrics, and error injection...")
        logger.info("The system will demonstrate:")
        logger.info("- Automatic error detection and recovery")
        logger.info("- Circuit breaker protection")
        logger.info("- Health monitoring and reporting")
        logger.info("- Metrics collection and export")
        logger.info("- Graceful error handling")
        
        # Run for demonstration period
        await asyncio.sleep(180)  # 3 minutes
        
        # Display final statistics
        logger.info("\n=== FINAL NETWORK STATISTICS ===")
        for i, agent in enumerate(agents):
            if agent.running:
                try:
                    status = await agent._handle_get_status({})
                    health = await agent._handle_get_health({}) if agent.health_monitor else {}
                    
                    logger.info(f"\nAgent {i+1} ({agent.node_id}):")
                    logger.info(f"  Uptime: {status.get('uptime', 0):.1f}s")
                    logger.info(f"  Peers: {status.get('connected_peers', 0)}")
                    logger.info(f"  Messages: {status.get('message_count', 0)}")
                    logger.info(f"  Errors: {status.get('error_count', 0)}")
                    logger.info(f"  Health: {status.get('health_status', 'unknown')}")
                    
                    if 'system_status' in health:
                        logger.info(f"  System Status: {health['system_status']}")
                    
                    # Show some metrics
                    if agent.metrics_manager:
                        latest = agent.metrics_manager.get_latest_metrics()
                        if 'mesh_connections_active' in latest:
                            logger.info(f"  Active Connections: {latest['mesh_connections_active'].value}")
                        if 'system_cpu_usage_percent' in latest:
                            logger.info(f"  CPU Usage: {latest['system_cpu_usage_percent'].value:.1f}%")
                
                except Exception as e:
                    logger.error(f"Failed to get final stats for agent {i+1}: {e}")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Shutting down robust agents...")
        for agent in agents:
            try:
                agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent: {e}")
        
        # Wait for agent tasks to complete
        if agent_tasks:
            await asyncio.gather(*agent_tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(demo_robust_network())