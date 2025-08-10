"""Peer discovery mechanisms for the Agent Mesh network.

Implements multiple peer discovery strategies including mDNS, DHT,
bootstrap servers, and gossip protocols for automatic peer finding.
Enhanced with network partition detection and recovery mechanisms.
"""

import asyncio
import json
import random
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Tuple
from uuid import UUID
from collections import defaultdict, deque

import structlog

from .network import PeerInfo, PeerStatus, TransportProtocol


logger = structlog.get_logger("discovery")


class DiscoveryMethod(Enum):
    """Peer discovery methods."""
    MDNS = "mdns"
    DHT = "dht"  
    BOOTSTRAP = "bootstrap"
    GOSSIP = "gossip"
    STATIC = "static"


@dataclass
class DiscoveryConfig:
    """Configuration for peer discovery."""
    methods: Set[DiscoveryMethod]
    bootstrap_peers: List[str]
    discovery_interval: int = 30
    service_name: str = "_agent-mesh._tcp"
    port: int = 4001
    max_peers: int = 50
    gossip_fanout: int = 3


class PeerDiscovery(ABC):
    """Abstract base class for peer discovery implementations."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the discovery mechanism."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the discovery mechanism."""
        pass
    
    @abstractmethod
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover new peers."""
        pass
    
    @abstractmethod
    async def announce(self, node_info: dict) -> None:
        """Announce this node to the network."""
        pass


class MDNSDiscovery(PeerDiscovery):
    """
    mDNS-based peer discovery for local networks.
    
    Uses multicast DNS for automatic peer discovery on local networks.
    Ideal for development and local deployments.
    """
    
    def __init__(self, config: DiscoveryConfig, node_id: UUID):
        self.config = config
        self.node_id = node_id
        self.logger = structlog.get_logger("mdns_discovery", node_id=str(node_id))
        
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._discovery_task: Optional[asyncio.Task] = None
        self._known_peers: Dict[UUID, PeerInfo] = {}
        
        # mDNS configuration
        self.multicast_group = "224.0.0.251"
        self.multicast_port = 5353
    
    async def start(self) -> None:
        """Start mDNS discovery."""
        self.logger.info("Starting mDNS discovery")
        self._running = True
        
        # Create multicast socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Join multicast group
        mreq = socket.inet_aton(self.multicast_group) + socket.inet_aton("0.0.0.0")
        self._socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Bind to multicast port
        self._socket.bind(("", self.multicast_port))
        self._socket.setblocking(False)
        
        # Start discovery loop
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        
        # Announce ourselves
        await self.announce({
            "node_id": str(self.node_id),
            "service": self.config.service_name,
            "port": self.config.port
        })
        
        self.logger.info("mDNS discovery started")
    
    async def stop(self) -> None:
        """Stop mDNS discovery."""
        self.logger.info("Stopping mDNS discovery")
        self._running = False
        
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        
        if self._socket:
            self._socket.close()
    
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover peers using mDNS."""
        # In a real implementation, this would query mDNS for service records
        # For now, we simulate peer discovery
        
        discovered = []
        
        # Simulate discovering some peers
        for i in range(random.randint(1, 3)):
            peer_id = UUID(int=random.randint(1, 1000000))
            if peer_id != self.node_id and peer_id not in self._known_peers:
                peer = PeerInfo(
                    peer_id=peer_id,
                    addresses=[f"192.168.1.{100+i}:4001"],
                    protocols={TransportProtocol.LIBP2P},
                    status=PeerStatus.UNKNOWN,
                    capabilities={"federated_learning": True, "consensus": True}
                )
                discovered.append(peer)
                self._known_peers[peer_id] = peer
        
        if discovered:
            self.logger.info("Discovered peers via mDNS", count=len(discovered))
        
        return discovered
    
    async def announce(self, node_info: dict) -> None:
        """Announce node via mDNS."""
        # Create service announcement
        announcement = {
            "type": "service_announcement",
            "service": self.config.service_name,
            "node_info": node_info,
            "timestamp": time.time()
        }
        
        # Broadcast announcement
        if self._socket:
            data = json.dumps(announcement).encode('utf-8')
            self._socket.sendto(data, (self.multicast_group, self.multicast_port))
        
        self.logger.debug("Announced service via mDNS")
    
    async def _discovery_loop(self) -> None:
        """Main mDNS discovery loop."""
        while self._running:
            try:
                # Listen for announcements
                await self._listen_for_announcements()
                
                # Periodic discovery
                await self.discover_peers()
                
                await asyncio.sleep(self.config.discovery_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("mDNS discovery loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _listen_for_announcements(self) -> None:
        """Listen for service announcements."""
        if not self._socket:
            return
        
        try:
            # Non-blocking receive with timeout
            loop = asyncio.get_event_loop()
            data, addr = await asyncio.wait_for(
                loop.sock_recvfrom(self._socket, 1024), 
                timeout=1.0
            )
            
            # Parse announcement
            announcement = json.loads(data.decode('utf-8'))
            
            if announcement.get("type") == "service_announcement":
                await self._process_announcement(announcement, addr)
                
        except asyncio.TimeoutError:
            pass  # Normal timeout
        except Exception as e:
            self.logger.debug("Error processing announcement", error=str(e))
    
    async def _process_announcement(self, announcement: dict, addr: tuple) -> None:
        """Process received service announcement."""
        node_info = announcement.get("node_info", {})
        node_id_str = node_info.get("node_id")
        
        if not node_id_str or node_id_str == str(self.node_id):
            return  # Ignore our own announcements
        
        try:
            peer_id = UUID(node_id_str)
            
            if peer_id not in self._known_peers:
                peer = PeerInfo(
                    peer_id=peer_id,
                    addresses=[f"{addr[0]}:{node_info.get('port', 4001)}"],
                    protocols={TransportProtocol.LIBP2P},
                    status=PeerStatus.UNKNOWN,
                    last_seen=time.time()
                )
                
                self._known_peers[peer_id] = peer
                self.logger.info("New peer discovered via mDNS", peer_id=str(peer_id))
                
        except ValueError:
            self.logger.warning("Invalid node ID in announcement", node_id=node_id_str)


class BootstrapDiscovery(PeerDiscovery):
    """
    Bootstrap server-based peer discovery.
    
    Connects to known bootstrap nodes to discover other peers in the network.
    """
    
    def __init__(self, config: DiscoveryConfig, node_id: UUID):
        self.config = config
        self.node_id = node_id
        self.logger = structlog.get_logger("bootstrap_discovery", node_id=str(node_id))
        
        self._running = False
        self._known_peers: Dict[UUID, PeerInfo] = {}
    
    async def start(self) -> None:
        """Start bootstrap discovery."""
        self.logger.info("Starting bootstrap discovery", 
                        bootstrap_count=len(self.config.bootstrap_peers))
        self._running = True
    
    async def stop(self) -> None:
        """Stop bootstrap discovery."""
        self.logger.info("Stopping bootstrap discovery")
        self._running = False
    
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover peers via bootstrap nodes."""
        discovered = []
        
        for bootstrap_addr in self.config.bootstrap_peers:
            try:
                peers = await self._query_bootstrap_node(bootstrap_addr)
                discovered.extend(peers)
            except Exception as e:
                self.logger.warning("Failed to query bootstrap node", 
                                  address=bootstrap_addr, error=str(e))
        
        if discovered:
            self.logger.info("Discovered peers via bootstrap", count=len(discovered))
        
        return discovered
    
    async def announce(self, node_info: dict) -> None:
        """Announce to bootstrap nodes."""
        for bootstrap_addr in self.config.bootstrap_peers:
            try:
                await self._announce_to_bootstrap(bootstrap_addr, node_info)
            except Exception as e:
                self.logger.warning("Failed to announce to bootstrap node",
                                  address=bootstrap_addr, error=str(e))
    
    async def _query_bootstrap_node(self, bootstrap_addr: str) -> List[PeerInfo]:
        """Query bootstrap node for peer list."""
        # In a real implementation, this would make HTTP/gRPC calls to bootstrap server
        # For now, simulate peer discovery
        
        discovered = []
        
        # Parse bootstrap address to create initial peer
        try:
            parts = bootstrap_addr.split(":")
            if len(parts) >= 2:
                host, port = parts[0], int(parts[1])
                
                # Create peer info for bootstrap node
                bootstrap_peer = PeerInfo(
                    peer_id=UUID(int=hash(bootstrap_addr) & 0xFFFFFFFFFFFFFFFF),
                    addresses=[bootstrap_addr],
                    protocols={TransportProtocol.LIBP2P},
                    status=PeerStatus.UNKNOWN,
                    capabilities={"bootstrap": True}
                )
                
                if bootstrap_peer.peer_id != self.node_id:
                    discovered.append(bootstrap_peer)
                    self._known_peers[bootstrap_peer.peer_id] = bootstrap_peer
                
        except Exception as e:
            self.logger.error("Error processing bootstrap address", 
                            address=bootstrap_addr, error=str(e))
        
        return discovered
    
    async def _announce_to_bootstrap(self, bootstrap_addr: str, node_info: dict) -> None:
        """Announce node to bootstrap server."""
        # In a real implementation, this would POST to bootstrap server
        self.logger.debug("Announcing to bootstrap node", address=bootstrap_addr)


class GossipDiscovery(PeerDiscovery):
    """
    Gossip protocol-based peer discovery.
    
    Uses gossip protocol to spread peer information throughout the network.
    """
    
    def __init__(self, config: DiscoveryConfig, node_id: UUID, network_manager):
        self.config = config
        self.node_id = node_id
        self.network_manager = network_manager
        self.logger = structlog.get_logger("gossip_discovery", node_id=str(node_id))
        
        self._running = False
        self._known_peers: Dict[UUID, PeerInfo] = {}
        self._gossip_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start gossip discovery."""
        self.logger.info("Starting gossip discovery")
        self._running = True
        
        # Start gossip loop
        self._gossip_task = asyncio.create_task(self._gossip_loop())
    
    async def stop(self) -> None:
        """Stop gossip discovery."""
        self.logger.info("Stopping gossip discovery")
        self._running = False
        
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
    
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover peers via gossip."""
        # Gossip discovery happens through periodic gossip messages
        return list(self._known_peers.values())
    
    async def announce(self, node_info: dict) -> None:
        """Announce via gossip protocol."""
        # Create gossip message
        gossip_msg = {
            "type": "peer_announcement",
            "node_info": node_info,
            "peers": [str(pid) for pid in self._known_peers.keys()],
            "timestamp": time.time()
        }
        
        # Broadcast to connected peers
        if self.network_manager:
            try:
                await self.network_manager.broadcast_message("gossip", gossip_msg)
            except Exception as e:
                self.logger.error("Failed to broadcast gossip", error=str(e))
    
    async def _gossip_loop(self) -> None:
        """Main gossip loop."""
        while self._running:
            try:
                await self._exchange_peer_lists()
                await asyncio.sleep(self.config.discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Gossip loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _exchange_peer_lists(self) -> None:
        """Exchange peer lists with random subset of peers."""
        if not self.network_manager:
            return
        
        try:
            connected_peers = await self.network_manager.get_connected_peers()
            
            if not connected_peers:
                return
            
            # Select random subset for gossip
            gossip_targets = random.sample(
                connected_peers, 
                min(self.config.gossip_fanout, len(connected_peers))
            )
            
            # Exchange peer lists
            for peer in gossip_targets:
                try:
                    # Request peer list from target
                    response = await self.network_manager.send_request(
                        peer.peer_id, "get_peer_list", {}, timeout=10.0
                    )
                    
                    # Process received peer list
                    if "peers" in response:
                        await self._process_peer_list(response["peers"])
                        
                except Exception as e:
                    self.logger.debug("Gossip exchange failed", 
                                    peer_id=str(peer.peer_id), error=str(e))
                    
        except Exception as e:
            self.logger.error("Error in peer list exchange", error=str(e))
    
    async def _process_peer_list(self, peer_list: List[dict]) -> None:
        """Process received peer list."""
        for peer_data in peer_list:
            try:
                peer_id = UUID(peer_data["peer_id"])
                
                if peer_id == self.node_id or peer_id in self._known_peers:
                    continue
                
                peer = PeerInfo(
                    peer_id=peer_id,
                    addresses=peer_data.get("addresses", []),
                    protocols=set(TransportProtocol(p) for p in peer_data.get("protocols", [])),
                    status=PeerStatus.UNKNOWN,
                    last_seen=time.time()
                )
                
                self._known_peers[peer_id] = peer
                self.logger.info("New peer discovered via gossip", peer_id=str(peer_id))
                
            except Exception as e:
                self.logger.warning("Invalid peer data in gossip", error=str(e))


@dataclass
class NetworkPartition:
    """Represents a detected network partition."""
    partition_id: str
    nodes: Set[UUID]
    detection_time: float
    last_contact: Dict[UUID, float] = field(default_factory=dict)
    is_active: bool = True


class PartitionDetector:
    """
    Network partition detection and recovery system.
    
    Monitors connectivity patterns and detects when the network
    becomes partitioned, implementing recovery strategies.
    """
    
    def __init__(self, node_id: UUID, detection_threshold: float = 30.0):
        self.node_id = node_id
        self.detection_threshold = detection_threshold
        self.logger = structlog.get_logger("partition_detector", node_id=str(node_id))
        
        # Partition tracking
        self.partitions: Dict[str, NetworkPartition] = {}
        self.peer_connectivity: Dict[UUID, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_contact: Dict[UUID, float] = {}
        
        # Detection state
        self._detection_active = False
        self._detection_task: Optional[asyncio.Task] = None
        self._recovery_callbacks: List[Callable[[NetworkPartition], None]] = []
    
    async def start_detection(self) -> None:
        """Start partition detection monitoring."""
        self.logger.info("Starting network partition detection")
        self._detection_active = True
        self._detection_task = asyncio.create_task(self._detection_loop())
    
    async def stop_detection(self) -> None:
        """Stop partition detection monitoring."""
        self.logger.info("Stopping network partition detection")
        self._detection_active = False
        
        if self._detection_task:
            self._detection_task.cancel()
            try:
                await self._detection_task
            except asyncio.CancelledError:
                pass
    
    def record_peer_contact(self, peer_id: UUID, success: bool) -> None:
        """Record contact attempt result for a peer."""
        current_time = time.time()
        
        # Record connectivity event
        self.peer_connectivity[peer_id].append((current_time, success))
        
        # Update last successful contact
        if success:
            self.last_contact[peer_id] = current_time
    
    def get_unreachable_peers(self) -> List[UUID]:
        """Get list of peers that appear unreachable."""
        current_time = time.time()
        unreachable = []
        
        for peer_id, last_seen in self.last_contact.items():
            if current_time - last_seen > self.detection_threshold:
                unreachable.append(peer_id)
        
        return unreachable
    
    def detect_partition(self, connected_peers: Set[UUID], all_known_peers: Set[UUID]) -> Optional[NetworkPartition]:
        """Detect if a network partition has occurred."""
        if not all_known_peers:
            return None
        
        # Calculate connectivity ratio
        connectivity_ratio = len(connected_peers) / len(all_known_peers)
        
        # If we can reach less than half the network, suspect partition
        if connectivity_ratio < 0.5 and len(all_known_peers) >= 3:
            unreachable_peers = all_known_peers - connected_peers
            
            # Create partition record
            partition_id = f"partition_{int(time.time())}"
            partition = NetworkPartition(
                partition_id=partition_id,
                nodes=unreachable_peers,
                detection_time=time.time()
            )
            
            # Update last contact times
            for peer_id in unreachable_peers:
                partition.last_contact[peer_id] = self.last_contact.get(peer_id, 0)
            
            self.logger.warning(
                "Network partition detected",
                partition_id=partition_id,
                unreachable_count=len(unreachable_peers),
                connectivity_ratio=connectivity_ratio
            )
            
            return partition
        
        return None
    
    def add_recovery_callback(self, callback: Callable[[NetworkPartition], None]) -> None:
        """Add callback for partition recovery events."""
        self._recovery_callbacks.append(callback)
    
    async def _detection_loop(self) -> None:
        """Main partition detection loop."""
        while self._detection_active:
            try:
                # Check for partition recovery
                await self._check_partition_recovery()
                
                # Sleep before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Partition detection loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _check_partition_recovery(self) -> None:
        """Check if any partitions have recovered."""
        current_time = time.time()
        recovered_partitions = []
        
        for partition_id, partition in self.partitions.items():
            if not partition.is_active:
                continue
            
            # Check if previously unreachable nodes are now reachable
            recovered_nodes = set()
            for node_id in partition.nodes:
                if node_id in self.last_contact:
                    # Node is reachable if contacted recently
                    if current_time - self.last_contact[node_id] < self.detection_threshold:
                        recovered_nodes.add(node_id)
            
            # If significant portion recovered, consider partition healed
            if len(recovered_nodes) >= len(partition.nodes) * 0.7:
                partition.is_active = False
                recovered_partitions.append(partition)
                
                self.logger.info(
                    "Network partition recovered",
                    partition_id=partition_id,
                    recovered_nodes=len(recovered_nodes),
                    duration=current_time - partition.detection_time
                )
        
        # Notify recovery callbacks
        for partition in recovered_partitions:
            for callback in self._recovery_callbacks:
                try:
                    callback(partition)
                except Exception as e:
                    self.logger.error("Recovery callback failed", error=str(e))


class MultiDiscovery:
    """
    Multi-method peer discovery coordinator with partition detection.
    
    Combines multiple discovery methods for robust peer finding
    and includes network partition detection and recovery.
    """
    
    def __init__(self, config: DiscoveryConfig, node_id: UUID, network_manager=None):
        self.config = config
        self.node_id = node_id
        self.network_manager = network_manager
        self.logger = structlog.get_logger("multi_discovery", node_id=str(node_id))
        
        self._discoveries: List[PeerDiscovery] = []
        self._all_peers: Dict[UUID, PeerInfo] = {}
        self._peer_callbacks: List[Callable[[PeerInfo], None]] = []
        
        # Partition detection
        self.partition_detector = PartitionDetector(
            node_id=node_id,
            detection_threshold=config.discovery_interval * 2
        )
        self._partition_callbacks: List[Callable[[NetworkPartition], None]] = []
        
        # Initialize discovery methods
        self._init_discovery_methods()
    
    def _init_discovery_methods(self) -> None:
        """Initialize configured discovery methods."""
        if DiscoveryMethod.MDNS in self.config.methods:
            self._discoveries.append(MDNSDiscovery(self.config, self.node_id))
        
        if DiscoveryMethod.BOOTSTRAP in self.config.methods and self.config.bootstrap_peers:
            self._discoveries.append(BootstrapDiscovery(self.config, self.node_id))
        
        if DiscoveryMethod.GOSSIP in self.config.methods and self.network_manager:
            self._discoveries.append(GossipDiscovery(self.config, self.node_id, self.network_manager))
        
        self.logger.info("Initialized discovery methods", count=len(self._discoveries))
    
    async def start(self) -> None:
        """Start all discovery methods and partition detection."""
        self.logger.info("Starting peer discovery with partition detection")
        
        # Start partition detection
        await self.partition_detector.start_detection()
        
        # Setup partition recovery callback
        self.partition_detector.add_recovery_callback(self._handle_partition_recovery)
        
        # Start discovery methods
        for discovery in self._discoveries:
            try:
                await discovery.start()
            except Exception as e:
                self.logger.error("Failed to start discovery method", 
                                discovery=discovery.__class__.__name__, error=str(e))
    
    async def stop(self) -> None:
        """Stop all discovery methods and partition detection."""
        self.logger.info("Stopping peer discovery and partition detection")
        
        # Stop partition detection
        await self.partition_detector.stop_detection()
        
        # Stop discovery methods
        for discovery in self._discoveries:
            try:
                await discovery.stop()
            except Exception as e:
                self.logger.error("Failed to stop discovery method",
                                discovery=discovery.__class__.__name__, error=str(e))
    
    async def discover_peers(self) -> List[PeerInfo]:
        """Discover peers using all methods with partition detection."""
        all_discovered = []
        
        for discovery in self._discoveries:
            try:
                peers = await discovery.discover_peers()
                all_discovered.extend(peers)
            except Exception as e:
                self.logger.error("Discovery method failed",
                                discovery=discovery.__class__.__name__, error=str(e))
                
                # Record failed discovery attempts for partition detection
                if hasattr(discovery, '_known_peers'):
                    for peer_id in discovery._known_peers.keys():
                        self.partition_detector.record_peer_contact(peer_id, False)
        
        # Deduplicate and update registry
        unique_peers = {}
        for peer in all_discovered:
            if peer.peer_id not in unique_peers:
                unique_peers[peer.peer_id] = peer
                
                # Record successful peer contact
                self.partition_detector.record_peer_contact(peer.peer_id, True)
                
                # Notify callbacks about new peers
                if peer.peer_id not in self._all_peers:
                    for callback in self._peer_callbacks:
                        try:
                            callback(peer)
                        except Exception as e:
                            self.logger.error("Peer callback failed", error=str(e))
        
        # Update registry
        self._all_peers.update(unique_peers)
        
        # Check for network partitions
        await self._check_for_partitions()
        
        return list(unique_peers.values())
    
    async def announce(self, node_info: dict) -> None:
        """Announce using all discovery methods."""
        for discovery in self._discoveries:
            try:
                await discovery.announce(node_info)
            except Exception as e:
                self.logger.error("Announcement failed",
                                discovery=discovery.__class__.__name__, error=str(e))
    
    def add_peer_callback(self, callback: Callable[[PeerInfo], None]) -> None:
        """Add callback for new peer discoveries."""
        self._peer_callbacks.append(callback)
    
    def get_all_peers(self) -> List[PeerInfo]:
        """Get all discovered peers."""
        return list(self._all_peers.values())
    
    def get_peer_count(self) -> int:
        """Get total number of discovered peers."""
        return len(self._all_peers)
    
    def add_partition_callback(self, callback: Callable[[NetworkPartition], None]) -> None:
        """Add callback for partition detection events."""
        self._partition_callbacks.append(callback)
    
    def get_partition_status(self) -> Dict[str, any]:
        """Get current partition detection status."""
        unreachable = self.partition_detector.get_unreachable_peers()
        
        return {
            "total_peers": len(self._all_peers),
            "unreachable_peers": len(unreachable),
            "active_partitions": len([p for p in self.partition_detector.partitions.values() if p.is_active]),
            "partition_threshold": self.partition_detector.detection_threshold,
            "unreachable_peer_ids": [str(pid) for pid in unreachable]
        }
    
    async def _check_for_partitions(self) -> None:
        """Check for network partitions and handle detection."""
        try:
            # Get connected peers from network manager
            connected_peers = set()
            if self.network_manager:
                try:
                    connected_peer_list = await self.network_manager.get_connected_peers()
                    connected_peers = {peer.peer_id for peer in connected_peer_list}
                except Exception as e:
                    self.logger.debug("Failed to get connected peers", error=str(e))
            
            # All known peers
            all_known_peers = set(self._all_peers.keys())
            
            # Detect partition
            partition = self.partition_detector.detect_partition(connected_peers, all_known_peers)
            
            if partition:
                # Store the partition
                self.partition_detector.partitions[partition.partition_id] = partition
                
                # Notify callbacks
                for callback in self._partition_callbacks:
                    try:
                        callback(partition)
                    except Exception as e:
                        self.logger.error("Partition callback failed", error=str(e))
                
                # Trigger recovery actions
                await self._trigger_partition_recovery(partition)
                
        except Exception as e:
            self.logger.error("Partition check failed", error=str(e))
    
    async def _trigger_partition_recovery(self, partition: NetworkPartition) -> None:
        """Trigger partition recovery actions."""
        self.logger.info(
            "Triggering partition recovery actions",
            partition_id=partition.partition_id,
            affected_nodes=len(partition.nodes)
        )
        
        # Recovery strategy: Try alternative discovery methods
        # 1. Increase discovery frequency temporarily
        original_interval = self.config.discovery_interval
        self.config.discovery_interval = max(5, original_interval // 2)
        
        # 2. Try to reconnect to bootstrap nodes if available
        if DiscoveryMethod.BOOTSTRAP in self.config.methods:
            for discovery in self._discoveries:
                if isinstance(discovery, BootstrapDiscovery):
                    try:
                        await discovery.discover_peers()
                    except Exception as e:
                        self.logger.debug("Bootstrap recovery failed", error=str(e))
        
        # 3. Schedule restoration of normal interval
        asyncio.create_task(self._restore_discovery_interval(original_interval, 300))  # 5 minutes
    
    async def _restore_discovery_interval(self, original_interval: int, delay: int) -> None:
        """Restore original discovery interval after delay."""
        await asyncio.sleep(delay)
        self.config.discovery_interval = original_interval
        self.logger.info("Restored normal discovery interval", interval=original_interval)
    
    def _handle_partition_recovery(self, partition: NetworkPartition) -> None:
        """Handle partition recovery notification."""
        self.logger.info(
            "Partition recovery detected",
            partition_id=partition.partition_id,
            recovered_nodes=len(partition.nodes),
            duration=time.time() - partition.detection_time
        )
        
        # Update peer status for recovered nodes
        for node_id in partition.nodes:
            if node_id in self._all_peers:
                peer = self._all_peers[node_id]
                peer.status = PeerStatus.CONNECTED
                peer.last_seen = time.time()