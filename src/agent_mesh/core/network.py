"""P2P Network layer implementation.

This module provides the networking foundation for the Agent Mesh system,
supporting multiple transport protocols including libp2p, gRPC, and WebRTC
for maximum connectivity and performance.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field


class TransportProtocol(Enum):
    """Supported transport protocols."""
    
    LIBP2P = "libp2p"
    GRPC = "grpc"
    WEBRTC = "webrtc"
    TCP = "tcp"


class PeerStatus(Enum):
    """Peer connection status."""
    
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class PeerInfo:
    """Information about a peer in the network."""
    
    peer_id: UUID
    addresses: List[str] = field(default_factory=list)
    protocols: Set[TransportProtocol] = field(default_factory=set)
    status: PeerStatus = PeerStatus.UNKNOWN
    last_seen: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    capabilities: Optional[Dict[str, Any]] = None
    reputation_score: float = 1.0


@dataclass
class NetworkMessage:
    """Network message structure."""
    
    message_id: UUID = field(default_factory=uuid4)
    sender_id: UUID = field(default=UUID(int=0))
    recipient_id: Optional[UUID] = None  # None for broadcast
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time to live (hops)
    signature: Optional[str] = None


@dataclass
class NetworkStatistics:
    """Network performance statistics."""
    
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    connections_active: int = 0
    connections_total: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


class MessageHandler(ABC):
    """Abstract base class for message handlers."""
    
    @abstractmethod
    async def handle_message(self, message: NetworkMessage, sender: PeerInfo) -> Optional[Dict[str, Any]]:
        """Handle incoming network message."""
        pass


class P2PTransport(ABC):
    """Abstract base class for P2P transport implementations."""
    
    @abstractmethod
    async def start(self, listen_addr: str) -> None:
        """Start the transport layer."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport layer."""
        pass
    
    @abstractmethod
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to a peer."""
        pass
    
    @abstractmethod
    async def send_message(self, peer_id: UUID, message: NetworkMessage) -> None:
        """Send message to a specific peer."""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: NetworkMessage) -> None:
        """Broadcast message to all connected peers."""
        pass
    
    @abstractmethod
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        pass


class LibP2PTransport(P2PTransport):
    """libp2p transport implementation."""
    
    def __init__(self, node_id: UUID, security_manager):
        self.node_id = node_id
        self.security_manager = security_manager
        self.logger = structlog.get_logger("libp2p_transport", node_id=str(node_id))
        
        self._peers: Dict[UUID, PeerInfo] = {}
        self._connections: Dict[UUID, Any] = {}  # Transport-specific connections
        self._listen_addr = ""
        self._running = False
    
    async def start(self, listen_addr: str) -> None:
        """Start the libp2p transport."""
        self.logger.info("Starting libp2p transport", listen_addr=listen_addr)
        self._listen_addr = listen_addr
        self._running = True
        
        # In a real implementation, this would initialize libp2p
        # For now, we simulate the transport layer
        await self._simulate_libp2p_start()
    
    async def stop(self) -> None:
        """Stop the libp2p transport."""
        self.logger.info("Stopping libp2p transport")
        self._running = False
        
        # Close all connections
        for peer_id in list(self._connections.keys()):
            await self._disconnect_peer(peer_id)
        
        self._peers.clear()
        self._connections.clear()
    
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to a peer using libp2p."""
        self.logger.info("Connecting to peer", peer_addr=peer_addr)
        
        # Parse peer address (simplified)
        peer_id = self._parse_peer_address(peer_addr)
        
        if peer_id in self._peers:
            peer = self._peers[peer_id]
            if peer.status == PeerStatus.CONNECTED:
                return peer
        
        # Simulate connection process
        peer = await self._establish_connection(peer_id, peer_addr)
        self._peers[peer_id] = peer
        
        self.logger.info("Connected to peer", peer_id=str(peer_id))
        return peer
    
    async def send_message(self, peer_id: UUID, message: NetworkMessage) -> None:
        """Send message to a specific peer."""
        if peer_id not in self._peers:
            raise ValueError(f"Peer {peer_id} not connected")
        
        peer = self._peers[peer_id]
        if peer.status != PeerStatus.CONNECTED:
            raise ValueError(f"Peer {peer_id} not in connected state")
        
        # Simulate message sending
        await self._simulate_send_message(peer_id, message)
        self.logger.debug("Message sent", peer_id=str(peer_id), 
                         message_type=message.message_type)
    
    async def broadcast_message(self, message: NetworkMessage) -> None:
        """Broadcast message to all connected peers."""
        connected_peers = [p for p in self._peers.values() 
                          if p.status == PeerStatus.CONNECTED]
        
        for peer in connected_peers:
            try:
                await self.send_message(peer.peer_id, message)
            except Exception as e:
                self.logger.warning("Failed to send broadcast message",
                                  peer_id=str(peer.peer_id), error=str(e))
    
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return [p for p in self._peers.values() if p.status == PeerStatus.CONNECTED]
    
    # Private methods
    
    async def _simulate_libp2p_start(self) -> None:
        """Simulate libp2p initialization."""
        # In real implementation, this would:
        # 1. Initialize libp2p node
        # 2. Set up transport protocols
        # 3. Configure security
        # 4. Start listening
        await asyncio.sleep(0.1)  # Simulate startup time
    
    def _parse_peer_address(self, peer_addr: str) -> UUID:
        """Parse peer address to extract peer ID."""
        # Simplified parsing - in real implementation would parse multiaddr
        # For now, generate deterministic UUID from address
        import hashlib
        hash_obj = hashlib.md5(peer_addr.encode())
        return UUID(hash_obj.hexdigest())
    
    async def _establish_connection(self, peer_id: UUID, peer_addr: str) -> PeerInfo:
        """Establish connection to a peer."""
        # Simulate connection establishment
        await asyncio.sleep(0.1)  # Connection time
        
        peer = PeerInfo(
            peer_id=peer_id,
            addresses=[peer_addr],
            protocols={TransportProtocol.LIBP2P},
            status=PeerStatus.CONNECTED,
            last_seen=time.time(),
            latency_ms=50.0  # Simulated latency
        )
        
        # Store connection handle (simulated)
        self._connections[peer_id] = {"address": peer_addr, "connected_at": time.time()}
        
        return peer
    
    async def _disconnect_peer(self, peer_id: UUID) -> None:
        """Disconnect from a peer."""
        if peer_id in self._peers:
            self._peers[peer_id].status = PeerStatus.DISCONNECTED
        
        if peer_id in self._connections:
            del self._connections[peer_id]
    
    async def _simulate_send_message(self, peer_id: UUID, message: NetworkMessage) -> None:
        """Simulate sending a message."""
        # In real implementation, this would serialize and send over libp2p
        await asyncio.sleep(0.01)  # Simulate network delay


class GRPCTransport(P2PTransport):
    """gRPC transport implementation for high-performance RPC."""
    
    def __init__(self, node_id: UUID, security_manager):
        self.node_id = node_id
        self.security_manager = security_manager
        self.logger = structlog.get_logger("grpc_transport", node_id=str(node_id))
        
        self._peers: Dict[UUID, PeerInfo] = {}
        self._server = None
        self._running = False
    
    async def start(self, listen_addr: str) -> None:
        """Start the gRPC transport."""
        self.logger.info("Starting gRPC transport", listen_addr=listen_addr)
        self._running = True
        
        # In real implementation, would start gRPC server
        await self._simulate_grpc_start()
    
    async def stop(self) -> None:
        """Stop the gRPC transport."""
        self.logger.info("Stopping gRPC transport")
        self._running = False
        
        if self._server:
            await self._server.stop(grace=5.0)
    
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to peer via gRPC."""
        peer_id = uuid4()  # Simplified
        
        peer = PeerInfo(
            peer_id=peer_id,
            addresses=[peer_addr],
            protocols={TransportProtocol.GRPC},
            status=PeerStatus.CONNECTED,
            latency_ms=20.0  # gRPC typically has lower latency
        )
        
        self._peers[peer_id] = peer
        return peer
    
    async def send_message(self, peer_id: UUID, message: NetworkMessage) -> None:
        """Send message via gRPC."""
        if peer_id not in self._peers:
            raise ValueError(f"Peer {peer_id} not connected")
        
        # Simulate gRPC call
        await asyncio.sleep(0.005)  # Faster than libp2p
    
    async def broadcast_message(self, message: NetworkMessage) -> None:
        """Broadcast message via gRPC."""
        tasks = [self.send_message(peer_id, message) 
                for peer_id in self._peers.keys()]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get connected peers."""
        return list(self._peers.values())
    
    async def _simulate_grpc_start(self) -> None:
        """Simulate gRPC server startup."""
        await asyncio.sleep(0.05)


class P2PNetwork:
    """
    Multi-protocol P2P network manager.
    
    Manages multiple transport protocols and provides a unified interface
    for network operations in the Agent Mesh system.
    """
    
    def __init__(
        self, 
        node_id: UUID,
        listen_addr: str = "/ip4/0.0.0.0/tcp/0",
        security_manager = None,
        enabled_protocols: Optional[Set[TransportProtocol]] = None
    ):
        """
        Initialize P2P network manager.
        
        Args:
            node_id: Unique node identifier
            listen_addr: Network address to listen on
            security_manager: Security manager instance
            enabled_protocols: Set of enabled transport protocols
        """
        self.node_id = node_id
        self.listen_addr = listen_addr
        self.security_manager = security_manager
        self.enabled_protocols = enabled_protocols or {
            TransportProtocol.LIBP2P, 
            TransportProtocol.GRPC
        }
        
        self.logger = structlog.get_logger("p2p_network", node_id=str(node_id))
        
        # Transport implementations
        self._transports: Dict[TransportProtocol, P2PTransport] = {}
        self._primary_transport = TransportProtocol.LIBP2P
        
        # Message handling
        self._message_handlers: Dict[str, MessageHandler] = {}
        self._request_handlers: Dict[str, Callable] = {}
        
        # Network state
        self._all_peers: Dict[UUID, PeerInfo] = {}
        self._statistics = NetworkStatistics()
        self._running = False
        self._start_time = 0.0
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the P2P network layer."""
        self.logger.info("Starting P2P network", protocols=len(self.enabled_protocols))
        self._start_time = time.time()
        
        # Initialize transports
        if TransportProtocol.LIBP2P in self.enabled_protocols:
            self._transports[TransportProtocol.LIBP2P] = LibP2PTransport(
                self.node_id, self.security_manager
            )
        
        if TransportProtocol.GRPC in self.enabled_protocols:
            self._transports[TransportProtocol.GRPC] = GRPCTransport(
                self.node_id, self.security_manager
            )
        
        # Start all transports
        for protocol, transport in self._transports.items():
            try:
                await transport.start(self.listen_addr)
                self.logger.info("Transport started", protocol=protocol.value)
            except Exception as e:
                self.logger.error("Failed to start transport", 
                                protocol=protocol.value, error=str(e))
        
        # Start background tasks
        self._discovery_task = asyncio.create_task(self._peer_discovery_loop())
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        self._running = True
        self.logger.info("P2P network started successfully")
    
    async def stop(self) -> None:
        """Stop the P2P network layer."""
        self.logger.info("Stopping P2P network")
        self._running = False
        
        # Stop background tasks
        if self._discovery_task:
            self._discovery_task.cancel()
        if self._maintenance_task:
            self._maintenance_task.cancel()
        
        # Stop all transports
        for protocol, transport in self._transports.items():
            try:
                await transport.stop()
                self.logger.info("Transport stopped", protocol=protocol.value)
            except Exception as e:
                self.logger.error("Error stopping transport", 
                                protocol=protocol.value, error=str(e))
        
        self._transports.clear()
        self._all_peers.clear()
    
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to a peer using the best available transport."""
        transport = self._get_primary_transport()
        peer = await transport.connect_to_peer(peer_addr)
        
        # Update peer registry
        self._all_peers[peer.peer_id] = peer
        self._statistics.connections_total += 1
        self._statistics.connections_active += 1
        
        self.logger.info("Connected to peer", peer_id=str(peer.peer_id))
        return peer
    
    async def connect_to_peer_id(self, peer_id: UUID) -> PeerInfo:
        """Connect to a peer by ID (requires known address)."""
        if peer_id in self._all_peers:
            peer = self._all_peers[peer_id]
            if peer.status == PeerStatus.CONNECTED:
                return peer
            
            # Try to reconnect using known addresses
            for addr in peer.addresses:
                try:
                    return await self.connect_to_peer(addr)
                except Exception:
                    continue
        
        raise ValueError(f"Cannot connect to unknown peer {peer_id}")
    
    async def send_message(
        self, 
        peer_id: UUID, 
        message_type: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> None:
        """Send a message to a specific peer."""
        message = NetworkMessage(
            sender_id=self.node_id,
            recipient_id=peer_id,
            message_type=message_type,
            payload=payload
        )
        
        transport = self._get_primary_transport()
        await transport.send_message(peer_id, message)
        
        self._statistics.messages_sent += 1
    
    async def broadcast_message(
        self, 
        message_type: str, 
        payload: Dict[str, Any]
    ) -> None:
        """Broadcast a message to all connected peers."""
        message = NetworkMessage(
            sender_id=self.node_id,
            message_type=message_type,
            payload=payload
        )
        
        transport = self._get_primary_transport()
        await transport.broadcast_message(message)
        
        peer_count = len(await self.get_connected_peers())
        self._statistics.messages_sent += peer_count
    
    async def send_request(
        self, 
        peer_id: UUID, 
        request_type: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send a request and wait for response."""
        request_id = str(uuid4())
        
        message = NetworkMessage(
            sender_id=self.node_id,
            recipient_id=peer_id,
            message_type="request",
            payload={
                "request_id": request_id,
                "request_type": request_type,
                "data": payload
            }
        )
        
        # Set up response waiting
        response_future = asyncio.Future()
        self._pending_requests[request_id] = response_future
        
        try:
            transport = self._get_primary_transport()
            await transport.send_message(peer_id, message)
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        finally:
            self._pending_requests.pop(request_id, None)
    
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of all connected peers across all transports."""
        all_peers = []
        
        for transport in self._transports.values():
            peers = await transport.get_connected_peers()
            all_peers.extend(peers)
        
        # Deduplicate by peer_id
        unique_peers = {}
        for peer in all_peers:
            if peer.peer_id not in unique_peers:
                unique_peers[peer.peer_id] = peer
        
        return list(unique_peers.values())
    
    async def get_listen_address(self) -> str:
        """Get the actual listen address."""
        return self.listen_addr
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        self._statistics.uptime_seconds = time.time() - self._start_time
        self._statistics.connections_active = len(await self.get_connected_peers())
        
        return {
            "messages_sent": self._statistics.messages_sent,
            "messages_received": self._statistics.messages_received,
            "bytes_sent": self._statistics.bytes_sent,
            "bytes_received": self._statistics.bytes_received,
            "connections_active": self._statistics.connections_active,
            "connections_total": self._statistics.connections_total,
            "avg_latency_ms": self._statistics.avg_latency_ms,
            "uptime_seconds": self._statistics.uptime_seconds
        }
    
    async def is_peer_responsive(self, peer_id: UUID) -> bool:
        """Check if a peer is responsive."""
        try:
            # Send ping message
            response = await self.send_request(
                peer_id, "ping", {"timestamp": time.time()}, timeout=5.0
            )
            return response.get("status") == "pong"
        except Exception:
            return False
    
    async def request_peer_list(self, peer_id: UUID) -> List[UUID]:
        """Request peer list from a connected peer."""
        try:
            response = await self.send_request(
                peer_id, "get_peers", {}, timeout=10.0
            )
            return [UUID(pid) for pid in response.get("peer_ids", [])]
        except Exception as e:
            self.logger.warning("Failed to get peer list", 
                              peer_id=str(peer_id), error=str(e))
            return []
    
    def register_message_handler(self, message_type: str, handler: MessageHandler) -> None:
        """Register a message handler for specific message types."""
        self._message_handlers[message_type] = handler
        self.logger.info("Message handler registered", message_type=message_type)
    
    def register_request_handler(self, request_type: str, handler: Callable) -> None:
        """Register a request handler for specific request types."""
        self._request_handlers[request_type] = handler
        self.logger.info("Request handler registered", request_type=request_type)
    
    # Private methods
    
    def _get_primary_transport(self) -> P2PTransport:
        """Get the primary transport for operations."""
        if self._primary_transport not in self._transports:
            # Fallback to any available transport
            if self._transports:
                return next(iter(self._transports.values()))
            else:
                raise RuntimeError("No transports available")
        
        return self._transports[self._primary_transport]
    
    async def _peer_discovery_loop(self) -> None:
        """Background peer discovery loop."""
        while self._running:
            try:
                # Periodic peer discovery logic
                peers = await self.get_connected_peers()
                
                # Request peer lists from connected peers
                for peer in peers:
                    try:
                        discovered_peers = await self.request_peer_list(peer.peer_id)
                        # Process discovered peers...
                    except Exception:
                        pass  # Ignore discovery failures
                
                await asyncio.sleep(60)  # Discovery interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Peer discovery error", error=str(e))
                await asyncio.sleep(60)
    
    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self._running:
            try:
                # Health check connected peers
                peers = await self.get_connected_peers()
                
                for peer in peers:
                    if not await self.is_peer_responsive(peer.peer_id):
                        self.logger.warning("Peer unresponsive", 
                                          peer_id=str(peer.peer_id))
                        peer.status = PeerStatus.FAILED
                
                # Update statistics
                await self.get_statistics()
                
                await asyncio.sleep(30)  # Maintenance interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Maintenance loop error", error=str(e))
                await asyncio.sleep(30)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_requests: Dict[str, asyncio.Future] = {}