#!/usr/bin/env python3
"""
Simple Agent Mesh Demo - Generation 1 Implementation

Demonstrates the real P2P networking, message serialization, and peer discovery
working together in a basic mesh network scenario.
"""

import asyncio
import logging
import random
import time
from uuid import uuid4

from agent_mesh.core.network import P2PNetwork
from agent_mesh.core.discovery import MultiDiscovery, DiscoveryConfig, DiscoveryMethod
from agent_mesh.core.serialization import MessageSerializer
from agent_mesh.database.connection import initialize_database


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mesh_demo")


class SimpleAgent:
    """
    Simple agent for mesh demonstration.
    
    Shows basic mesh networking capabilities including peer discovery,
    message passing, and simple coordination.
    """
    
    def __init__(self, node_id=None, listen_port=None):
        self.node_id = node_id or uuid4()
        self.listen_port = listen_port or random.randint(4001, 4100)
        
        # Initialize components
        self.network = P2PNetwork(
            node_id=self.node_id,
            listen_addr=f"/ip4/0.0.0.0/tcp/{self.listen_port}"
        )
        
        self.discovery = MultiDiscovery(
            config=DiscoveryConfig(
                methods={DiscoveryMethod.MDNS, DiscoveryMethod.GOSSIP},
                discovery_interval=10,
                max_peers=20
            ),
            node_id=self.node_id,
            network_manager=self.network
        )
        
        self.serializer = MessageSerializer()
        
        # Agent state
        self.running = False
        self.peers_connected = 0
        self.messages_sent = 0
        self.messages_received = 0
        
        logger.info(f"Agent {self.node_id} initialized on port {self.listen_port}")
    
    async def start(self):
        """Start the agent and join the mesh network."""
        logger.info(f"Starting agent {self.node_id}")
        
        try:
            # Initialize database
            db = initialize_database("sqlite:///data/mesh_demo.db")
            await db.initialize_async()
            
            # Start network layer
            await self.network.start()
            
            # Register message handlers
            self.network.register_request_handler("ping", self._handle_ping)
            self.network.register_request_handler("get_peer_list", self._handle_get_peer_list)
            
            # Start peer discovery
            await self.discovery.start()
            self.discovery.add_peer_callback(self._on_peer_discovered)
            
            # Announce ourselves
            await self.discovery.announce({
                "node_id": str(self.node_id),
                "port": self.listen_port,
                "capabilities": ["simple_agent", "demo"],
                "version": "1.0"
            })
            
            self.running = True
            logger.info(f"Agent {self.node_id} started successfully")
            
            # Start background tasks
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._heartbeat_loop())
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the agent and clean up resources."""
        logger.info(f"Stopping agent {self.node_id}")
        
        self.running = False
        
        # Stop components
        await self.discovery.stop()
        await self.network.stop()
        
        logger.info(f"Agent {self.node_id} stopped")
    
    async def connect_to_peer(self, peer_address: str):
        """Connect to a specific peer."""
        try:
            peer = await self.network.connect_to_peer(peer_address)
            logger.info(f"Connected to peer {peer.peer_id} at {peer_address}")
            self.peers_connected += 1
            return peer
        except Exception as e:
            logger.error(f"Failed to connect to {peer_address}: {e}")
            raise
    
    async def send_message(self, peer_id, message_type: str, data: dict):
        """Send a message to a peer."""
        try:
            await self.network.send_message(peer_id, message_type, data)
            self.messages_sent += 1
            logger.debug(f"Sent {message_type} to {peer_id}")
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
    
    async def broadcast_message(self, message_type: str, data: dict):
        """Broadcast a message to all connected peers."""
        try:
            await self.network.broadcast_message(message_type, data)
            peers = await self.network.get_connected_peers()
            self.messages_sent += len(peers)
            logger.info(f"Broadcasted {message_type} to {len(peers)} peers")
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    async def get_network_status(self) -> dict:
        """Get network status information."""
        peers = await self.network.get_connected_peers()
        stats = await self.network.get_statistics()
        
        return {
            "node_id": str(self.node_id),
            "listen_port": self.listen_port,
            "connected_peers": len(peers),
            "total_peers_discovered": self.discovery.get_peer_count(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "network_stats": stats,
            "uptime": time.time() - stats.get("uptime_seconds", 0)
        }
    
    # Message handlers
    
    async def _handle_ping(self, request_data: dict) -> dict:
        """Handle ping request."""
        self.messages_received += 1
        return {
            "status": "pong",
            "timestamp": time.time(),
            "node_id": str(self.node_id)
        }
    
    async def _handle_get_peer_list(self, request_data: dict) -> dict:
        """Handle peer list request."""
        peers = await self.network.get_connected_peers()
        peer_list = [
            {
                "peer_id": str(peer.peer_id),
                "addresses": peer.addresses,
                "protocols": [p.value for p in peer.protocols],
                "last_seen": peer.last_seen
            }
            for peer in peers
        ]
        
        return {"peers": peer_list}
    
    # Background tasks
    
    async def _discovery_loop(self):
        """Periodic peer discovery."""
        while self.running:
            try:
                # Discover new peers
                discovered = await self.discovery.discover_peers()
                
                if discovered:
                    logger.info(f"Discovered {len(discovered)} new peers")
                
                # Try connecting to some discovered peers
                for peer in discovered[:3]:  # Limit connections
                    if peer.addresses:
                        try:
                            await self.connect_to_peer(peer.addresses[0])
                        except Exception:
                            continue  # Skip failed connections
                
                await asyncio.sleep(30)  # Discovery interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(30)
    
    async def _heartbeat_loop(self):
        """Periodic heartbeat and status updates."""
        while self.running:
            try:
                # Send heartbeat to connected peers
                heartbeat_data = {
                    "timestamp": time.time(),
                    "status": "healthy",
                    "peers_connected": self.peers_connected
                }
                
                await self.broadcast_message("heartbeat", heartbeat_data)
                
                # Log status
                status = await self.get_network_status()
                logger.info(f"Agent {self.node_id} - Peers: {status['connected_peers']}, "
                          f"Messages sent/received: {status['messages_sent']}/{status['messages_received']}")
                
                await asyncio.sleep(60)  # Heartbeat interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(60)
    
    def _on_peer_discovered(self, peer):
        """Callback for when a new peer is discovered."""
        logger.info(f"New peer discovered: {peer.peer_id}")


async def demo_simple_network():
    """Demonstrate simple mesh networking."""
    logger.info("Starting simple mesh network demo")
    
    # Create multiple agents
    agents = []
    
    try:
        # Create 3 agents
        for i in range(3):
            agent = SimpleAgent(listen_port=4001 + i)
            agents.append(agent)
            await agent.start()
            await asyncio.sleep(1)  # Stagger startup
        
        logger.info(f"Created {len(agents)} agents")
        
        # Let them discover each other
        await asyncio.sleep(10)
        
        # Connect agents manually to bootstrap the network
        if len(agents) >= 2:
            await agents[1].connect_to_peer("127.0.0.1:4001")
            await agents[2].connect_to_peer("127.0.0.1:4002")
        
        # Let the network stabilize
        await asyncio.sleep(5)
        
        # Test message passing
        logger.info("Testing message passing...")
        
        for agent in agents:
            await agent.broadcast_message("test_message", {
                "sender": str(agent.node_id),
                "message": f"Hello from agent {agent.node_id}!",
                "timestamp": time.time()
            })
            
            await asyncio.sleep(2)
        
        # Display network status
        logger.info("\nNetwork Status:")
        for i, agent in enumerate(agents):
            status = await agent.get_network_status()
            logger.info(f"Agent {i+1}: {status['connected_peers']} peers, "
                       f"{status['messages_sent']} sent, {status['messages_received']} received")
        
        # Run for a bit to see the network in action
        logger.info("Letting network run for 30 seconds...")
        await asyncio.sleep(30)
        
    finally:
        # Clean up
        logger.info("Stopping agents...")
        for agent in agents:
            try:
                await agent.stop()
            except Exception as e:
                logger.error(f"Error stopping agent: {e}")


async def demo_peer_discovery():
    """Demonstrate peer discovery mechanisms."""
    logger.info("Starting peer discovery demo")
    
    agent = SimpleAgent()
    
    try:
        await agent.start()
        
        # Run discovery for a while
        for i in range(6):  # 1 minute total
            discovered = await agent.discovery.discover_peers()
            logger.info(f"Discovery round {i+1}: found {len(discovered)} peers")
            
            # Show all known peers
            all_peers = agent.discovery.get_all_peers()
            logger.info(f"Total known peers: {len(all_peers)}")
            
            await asyncio.sleep(10)
    
    finally:
        await agent.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "discovery":
        asyncio.run(demo_peer_discovery())
    else:
        asyncio.run(demo_simple_network())