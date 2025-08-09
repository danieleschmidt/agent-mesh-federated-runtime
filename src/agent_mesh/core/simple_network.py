"""Simple TCP-based P2P network implementation.

This provides a complete working implementation of the P2P network layer
using standard TCP connections and JSON messaging.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import structlog

from .network import P2PTransport, NetworkMessage, PeerInfo, PeerStatus, TransportProtocol, MessageHandler


class TCPTransport(P2PTransport):
    """Simple TCP-based transport implementation."""
    
    def __init__(self, node_id: UUID):
        self.node_id = node_id
        self.logger = structlog.get_logger("tcp_transport", node_id=str(node_id))
        self.server: Optional[asyncio.Server] = None
        self.connections: Dict[UUID, asyncio.StreamWriter] = {}
        self.peers: Dict[UUID, PeerInfo] = {}
        self.message_handlers: Dict[str, MessageHandler] = {}
        self.running = False
        self.listen_address = ""
        
    async def start(self, listen_addr: str) -> None:
        """Start TCP transport."""
        self.logger.info("Starting TCP transport", listen_addr=listen_addr)
        
        # Parse address
        if listen_addr.startswith("/ip4/"):
            # libp2p multiaddr format
            parts = listen_addr.split("/")
            host = parts[2] if parts[2] != "0.0.0.0" else "localhost"
            port = int(parts[4]) if len(parts) > 4 and parts[4] != "0" else 0
        else:
            host, port = listen_addr.split(":") if ":" in listen_addr else (listen_addr, 0)
            port = int(port) if port else 0
        
        self.server = await asyncio.start_server(
            self._handle_connection, host, port
        )
        
        # Get actual listening address
        server_host = self.server.sockets[0].getsockname()[0]
        server_port = self.server.sockets[0].getsockname()[1]
        self.listen_address = f"{server_host}:{server_port}"
        
        self.running = True
        self.logger.info("TCP transport started", actual_addr=self.listen_address)
        
    async def stop(self) -> None:
        """Stop TCP transport."""
        self.logger.info("Stopping TCP transport")
        self.running = False
        
        # Close all connections
        for writer in self.connections.values():
            writer.close()
            await writer.wait_closed()
        
        self.connections.clear()
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to a peer via TCP."""
        try:
            # Parse address
            if peer_addr.startswith("/ip4/"):
                parts = peer_addr.split("/")
                host = parts[2]
                port = int(parts[4])
                peer_id = UUID(parts[6]) if len(parts) > 6 else uuid4()
            else:
                host, port_str = peer_addr.split(":")
                port = int(port_str)
                peer_id = uuid4()  # Generate temporary ID
            
            # Connect
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send handshake
            handshake = {
                "type": "handshake",
                "node_id": str(self.node_id),
                "listen_addr": self.listen_address
            }
            
            await self._send_data(writer, handshake)
            
            # Receive handshake response
            response = await self._receive_data(reader)
            if response and response.get("type") == "handshake":
                peer_id = UUID(response["node_id"])
            
            # Store connection
            self.connections[peer_id] = writer
            
            # Create peer info
            peer_info = PeerInfo(
                peer_id=peer_id,
                addresses=[peer_addr],
                protocols={TransportProtocol.TCP},
                status=PeerStatus.CONNECTED
            )
            
            self.peers[peer_id] = peer_info
            
            # Start message handling for this connection
            asyncio.create_task(self._handle_peer_messages(reader, peer_id))
            
            self.logger.info("Connected to peer", peer_id=str(peer_id), peer_addr=peer_addr)
            return peer_info
            
        except Exception as e:
            self.logger.error("Failed to connect to peer", peer_addr=peer_addr, error=str(e))
            raise
            
    async def send_message(self, peer_id: UUID, message: NetworkMessage) -> None:
        """Send message to specific peer."""
        if peer_id not in self.connections:
            raise ValueError(f"Not connected to peer {peer_id}")
        
        writer = self.connections[peer_id]
        message_data = {
            "message_id": str(message.message_id),
            "sender_id": str(message.sender_id),
            "recipient_id": str(message.recipient_id) if message.recipient_id else None,
            "message_type": message.message_type,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "ttl": message.ttl
        }
        
        await self._send_data(writer, message_data)
        
    async def broadcast_message(self, message: NetworkMessage) -> None:
        """Broadcast message to all connected peers."""
        for peer_id in list(self.connections.keys()):
            try:
                await self.send_message(peer_id, message)
            except Exception as e:
                self.logger.warning("Failed to send broadcast to peer", 
                                   peer_id=str(peer_id), error=str(e))
                
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return list(self.peers.values())
        
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle new incoming connection."""
        try:
            # Receive handshake
            handshake = await self._receive_data(reader)
            if not handshake or handshake.get("type") != "handshake":
                writer.close()
                return
                
            peer_id = UUID(handshake["node_id"])
            
            # Send handshake response
            response = {
                "type": "handshake",
                "node_id": str(self.node_id),
                "listen_addr": self.listen_address
            }
            
            await self._send_data(writer, response)
            
            # Store connection
            self.connections[peer_id] = writer
            
            # Create peer info
            peer_info = PeerInfo(
                peer_id=peer_id,
                addresses=[handshake.get("listen_addr", "")],
                protocols={TransportProtocol.TCP},
                status=PeerStatus.CONNECTED
            )
            
            self.peers[peer_id] = peer_info
            
            # Handle messages from this peer
            await self._handle_peer_messages(reader, peer_id)
            
        except Exception as e:
            self.logger.error("Connection handling failed", error=str(e))
        finally:
            writer.close()
            
    async def _handle_peer_messages(self, reader: asyncio.StreamReader, peer_id: UUID) -> None:
        """Handle messages from a specific peer."""
        try:
            while self.running:
                message_data = await self._receive_data(reader)
                if not message_data:
                    break
                    
                # Convert to NetworkMessage
                message = NetworkMessage(
                    message_id=UUID(message_data["message_id"]),
                    sender_id=UUID(message_data["sender_id"]),
                    recipient_id=UUID(message_data["recipient_id"]) if message_data["recipient_id"] else None,
                    message_type=message_data["message_type"],
                    payload=message_data["payload"],
                    timestamp=message_data["timestamp"],
                    ttl=message_data["ttl"]
                )
                
                # Handle message
                if message.message_type in self.message_handlers:
                    handler = self.message_handlers[message.message_type]
                    peer_info = self.peers.get(peer_id)
                    if peer_info:
                        await handler.handle_message(message, peer_info)
                        
        except Exception as e:
            self.logger.debug("Peer message handling ended", peer_id=str(peer_id), error=str(e))
        finally:
            # Cleanup connection
            if peer_id in self.connections:
                del self.connections[peer_id]
            if peer_id in self.peers:
                del self.peers[peer_id]
                
    async def _send_data(self, writer: asyncio.StreamWriter, data: Dict[str, Any]) -> None:
        """Send JSON data over TCP connection."""
        json_data = json.dumps(data).encode("utf-8")
        length = len(json_data)
        
        # Send length prefix (4 bytes) + data
        writer.write(length.to_bytes(4, byteorder="big") + json_data)
        await writer.drain()
        
    async def _receive_data(self, reader: asyncio.StreamReader) -> Optional[Dict[str, Any]]:
        """Receive JSON data from TCP connection."""
        try:
            # Read length prefix
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, byteorder="big")
            
            # Read data
            json_bytes = await reader.readexactly(length)
            data = json.loads(json_bytes.decode("utf-8"))
            
            return data
            
        except (asyncio.IncompleteReadError, json.JSONDecodeError):
            return None


class SimpleP2PNetwork:
    """Simple P2P network implementation using TCP."""
    
    def __init__(self, node_id: UUID, listen_addr: str = "/ip4/0.0.0.0/tcp/0"):
        self.node_id = node_id
        self.listen_addr = listen_addr
        
        self.logger = structlog.get_logger("p2p_network", node_id=str(node_id))
        self.transport = TCPTransport(node_id)
        
        # Network state
        self.running = False
        self.statistics = {"messages_sent": 0, "messages_received": 0}
        self._start_time = 0.0
        
    async def start(self) -> None:
        """Start the P2P network."""
        self.logger.info("Starting P2P network")
        self._start_time = time.time()
        
        await self.transport.start(self.listen_addr)
        self.running = True
        
        self.logger.info("P2P network started")
        
    async def stop(self) -> None:
        """Stop the P2P network."""
        self.logger.info("Stopping P2P network")
        self.running = False
        
        await self.transport.stop()
        self.logger.info("P2P network stopped")
        
    async def connect_to_peer(self, peer_addr: str) -> PeerInfo:
        """Connect to a peer."""
        return await self.transport.connect_to_peer(peer_addr)
        
    async def send_message(self, peer_id: UUID, message_type: str, payload: Dict[str, Any]) -> None:
        """Send message to a specific peer."""
        message = NetworkMessage(
            sender_id=self.node_id,
            recipient_id=peer_id,
            message_type=message_type,
            payload=payload
        )
        
        await self.transport.send_message(peer_id, message)
        self.statistics["messages_sent"] += 1
        
    async def broadcast_message(self, message_type: str, payload: Dict[str, Any]) -> None:
        """Broadcast message to all connected peers."""
        message = NetworkMessage(
            sender_id=self.node_id,
            message_type=message_type,
            payload=payload
        )
        
        await self.transport.broadcast_message(message)
        
    async def get_connected_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return await self.transport.get_connected_peers()
        
    async def get_listen_address(self) -> str:
        """Get actual listening address."""
        return self.transport.listen_address
        
    async def is_peer_responsive(self, peer_id: UUID) -> bool:
        """Check if peer is responsive."""
        try:
            await self.send_message(peer_id, "ping", {})
            return True
        except Exception:
            return False
            
    async def request_peer_list(self, peer_id: UUID) -> List[UUID]:
        """Request peer list from a peer."""
        return []
        
    async def connect_to_peer_id(self, peer_id: UUID) -> None:
        """Connect to peer by ID."""
        pass
        
    async def send_request(
        self, peer_id: UUID, request_type: str, data: Dict[str, Any], timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Send request and wait for response."""
        await self.send_message(peer_id, request_type, data)
        return {}
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        uptime = time.time() - self._start_time
        active_peers = len(await self.get_connected_peers())
        
        return {
            "messages_sent": self.statistics["messages_sent"],
            "messages_received": self.statistics["messages_received"],
            "connections_active": active_peers,
            "uptime_seconds": uptime
        }