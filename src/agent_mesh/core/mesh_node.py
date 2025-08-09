"""Core MeshNode implementation.

The MeshNode is the fundamental building block of the Agent Mesh network.
It handles P2P connectivity, role negotiation, consensus participation,
and task execution.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from uuid import uuid4, UUID

import structlog
from pydantic import BaseModel, Field

from .simple_network import SimpleP2PNetwork as P2PNetwork, PeerInfo
from .consensus import ConsensusEngine, ConsensusResult
from .security import SecurityManager, NodeIdentity


class NodeRole(Enum):
    """Available node roles in the mesh network."""
    
    TRAINER = "trainer"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"


class NodeStatus(Enum):
    """Node lifecycle status."""
    
    INITIALIZING = "initializing"
    CONNECTING = "connecting"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


@dataclass
class NodeCapabilities:
    """Node computational and functional capabilities."""
    
    cpu_cores: int = 1
    memory_gb: float = 1.0
    storage_gb: float = 10.0
    gpu_available: bool = False
    bandwidth_mbps: float = 10.0
    skills: Set[str] = field(default_factory=set)
    supported_protocols: Set[str] = field(default_factory=lambda: {"libp2p", "grpc"})


@dataclass
class NodeMetrics:
    """Runtime metrics for node performance monitoring."""
    
    uptime_seconds: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    consensus_rounds_participated: int = 0
    tasks_completed: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    last_update: float = field(default_factory=time.time)


class TaskRequest(BaseModel):
    """Request for task execution."""
    
    task_id: UUID = Field(default_factory=uuid4)
    task_type: str
    payload: Dict[str, Any]
    requester_id: UUID
    priority: int = Field(default=1, ge=1, le=10)
    timeout_seconds: float = 300.0
    required_skills: List[str] = Field(default_factory=list)


class TaskResult(BaseModel):
    """Result of task execution."""
    
    task_id: UUID
    executor_id: UUID
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: float
    timestamp: float = Field(default_factory=time.time)


class MeshNode:
    """
    Core mesh network node implementation.
    
    Handles all aspects of mesh networking including:
    - P2P connectivity and peer management
    - Dynamic role negotiation and lifecycle
    - Consensus protocol participation
    - Task execution and coordination
    - Security and identity management
    """
    
    def __init__(
        self,
        node_id: Optional[UUID] = None,
        listen_addr: str = "/ip4/0.0.0.0/tcp/0",
        bootstrap_peers: Optional[List[str]] = None,
        capabilities: Optional[NodeCapabilities] = None,
        auto_role: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new mesh node.
        
        Args:
            node_id: Unique node identifier (generated if None)
            listen_addr: Network address to listen on
            bootstrap_peers: List of bootstrap peer addresses
            capabilities: Node computational capabilities
            auto_role: Whether to automatically negotiate role
            config: Additional configuration parameters
        """
        self.node_id = node_id or uuid4()
        self.listen_addr = listen_addr
        self.bootstrap_peers = bootstrap_peers or []
        self.capabilities = capabilities or NodeCapabilities()
        self.auto_role = auto_role
        self.config = config or {}
        
        # Node state
        self.status = NodeStatus.INITIALIZING
        self.role = NodeRole.OBSERVER
        self.metrics = NodeMetrics()
        
        # Component initialization
        self.logger = structlog.get_logger("mesh_node", node_id=str(self.node_id))
        self.security = SecurityManager()
        self.network = P2PNetwork(
            node_id=self.node_id,
            listen_addr=listen_addr,
            security_manager=self.security
        )
        self.consensus = ConsensusEngine(
            node_id=self.node_id,
            network=self.network
        )
        
        # Task management
        self._task_handlers: Dict[str, Callable] = {}
        self._active_tasks: Dict[UUID, asyncio.Task] = {}
        self._peer_capabilities: Dict[UUID, NodeCapabilities] = {}
        
        # Runtime state
        self._running = False
        self._start_time = 0.0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._role_negotiation_lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the mesh node and join the network."""
        try:
            self.logger.info("Starting mesh node")
            self._start_time = time.time()
            self.status = NodeStatus.CONNECTING
            
            # Initialize security and identity
            await self.security.initialize()
            identity = await self.security.get_node_identity()
            self.logger.info("Node identity established", 
                           public_key=identity.public_key_hex[:16] + "...")
            
            # Start network layer
            await self.network.start()
            actual_addr = await self.network.get_listen_address()
            self.logger.info("Network layer started", listen_addr=actual_addr)
            
            # Connect to bootstrap peers
            if self.bootstrap_peers:
                await self._connect_to_bootstrap_peers()
            
            # Start consensus engine
            await self.consensus.start()
            
            # Begin role negotiation if enabled
            if self.auto_role:
                await self._negotiate_initial_role()
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            self.status = NodeStatus.ACTIVE
            self._running = True
            
            self.logger.info("Mesh node started successfully", 
                           role=self.role.value, status=self.status.value)
            
        except Exception as e:
            self.logger.error("Failed to start mesh node", error=str(e))
            self.status = NodeStatus.FAILED
            raise
    
    async def stop(self) -> None:
        """Stop the mesh node and cleanup resources."""
        self.logger.info("Stopping mesh node")
        self._running = False
        
        # Stop background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active tasks
        for task in self._active_tasks.values():
            task.cancel()
        
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)
        
        # Stop components
        await self.consensus.stop()
        await self.network.stop()
        await self.security.cleanup()
        
        self.status = NodeStatus.DISCONNECTED
        self.logger.info("Mesh node stopped")
    
    async def join_network(self, bootstrap_peers: Optional[List[str]] = None) -> None:
        """Join the mesh network using bootstrap peers."""
        peers = bootstrap_peers or self.bootstrap_peers
        if not peers:
            self.logger.warning("No bootstrap peers provided")
            return
        
        self.logger.info("Joining network", bootstrap_peers=len(peers))
        
        connected_count = 0
        for peer_addr in peers:
            try:
                peer_info = await self.network.connect_to_peer(peer_addr)
                self.logger.info("Connected to bootstrap peer", 
                               peer_id=str(peer_info.peer_id))
                connected_count += 1
            except Exception as e:
                self.logger.warning("Failed to connect to bootstrap peer",
                                  peer_addr=peer_addr, error=str(e))
        
        if connected_count == 0:
            raise RuntimeError("Failed to connect to any bootstrap peers")
        
        # Discover additional peers
        await self._discover_peers()
    
    async def attach_task(self, task_type: str, handler: Callable) -> None:
        """Attach a task handler for specific task types."""
        self._task_handlers[task_type] = handler
        self.logger.info("Task handler attached", task_type=task_type)
    
    async def submit_task(self, task: TaskRequest) -> TaskResult:
        """Submit a task for execution on the mesh network."""
        self.logger.info("Submitting task", task_id=str(task.task_id), 
                        task_type=task.task_type)
        
        # Find suitable executor
        executor_peer = await self._find_task_executor(task)
        
        if executor_peer == self.node_id:
            # Execute locally
            return await self._execute_task_locally(task)
        else:
            # Delegate to remote peer
            return await self._delegate_task(task, executor_peer)
    
    async def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return await self.network.get_connected_peers()
    
    async def get_node_metrics(self) -> NodeMetrics:
        """Get current node performance metrics."""
        # Update runtime metrics
        self.metrics.uptime_seconds = time.time() - self._start_time
        self.metrics.last_update = time.time()
        
        # Get network metrics
        network_stats = await self.network.get_statistics()
        self.metrics.messages_sent = network_stats.get("messages_sent", 0)
        self.metrics.messages_received = network_stats.get("messages_received", 0)
        
        return self.metrics
    
    async def negotiate_role(self, preferred_roles: Optional[List[NodeRole]] = None) -> NodeRole:
        """Negotiate node role based on capabilities and network needs."""
        async with self._role_negotiation_lock:
            self.logger.info("Starting role negotiation", 
                           current_role=self.role.value,
                           preferred_roles=[r.value for r in (preferred_roles or [])])
            
            # Get network state and peer capabilities
            peers = await self.get_peers()
            network_state = await self._analyze_network_state(peers)
            
            # Determine optimal role
            new_role = await self._calculate_optimal_role(
                network_state, preferred_roles
            )
            
            if new_role != self.role:
                # Propose role change through consensus
                role_proposal = {
                    "type": "role_change",
                    "node_id": str(self.node_id),
                    "old_role": self.role.value,
                    "new_role": new_role.value,
                    "capabilities": self.capabilities.__dict__,
                    "timestamp": time.time()
                }
                
                result = await self.consensus.propose(role_proposal)
                
                if result.accepted:
                    old_role = self.role
                    self.role = new_role
                    self.logger.info("Role change accepted", 
                                   old_role=old_role.value, 
                                   new_role=new_role.value)
                else:
                    self.logger.warning("Role change rejected", 
                                      proposed_role=new_role.value,
                                      reason=result.reason)
            
            return self.role
    
    # Private methods
    
    async def _connect_to_bootstrap_peers(self) -> None:
        """Connect to initial bootstrap peers."""
        self.logger.info("Connecting to bootstrap peers", count=len(self.bootstrap_peers))
        
        connection_tasks = [
            self.network.connect_to_peer(peer_addr)
            for peer_addr in self.bootstrap_peers
        ]
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning("Bootstrap connection failed",
                                  peer_addr=self.bootstrap_peers[i],
                                  error=str(result))
            else:
                successful_connections += 1
                self.logger.info("Connected to bootstrap peer",
                               peer_id=str(result.peer_id))
        
        if successful_connections == 0:
            raise RuntimeError("Failed to connect to any bootstrap peers")
    
    async def _negotiate_initial_role(self) -> None:
        """Perform initial role negotiation based on network state."""
        # Wait a moment for network discovery
        await asyncio.sleep(2.0)
        
        await self.negotiate_role()
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat and health monitoring loop."""
        while self._running:
            try:
                # Update metrics
                await self.get_node_metrics()
                
                # Check peer health
                peers = await self.get_peers()
                for peer in peers:
                    if not await self.network.is_peer_responsive(peer.peer_id):
                        self.logger.warning("Peer unresponsive", 
                                          peer_id=str(peer.peer_id))
                
                # Periodic role rebalancing
                if self.auto_role and len(peers) > 0:
                    if time.time() % 300 < 30:  # Every 5 minutes
                        await self.negotiate_role()
                
                await asyncio.sleep(30)  # Heartbeat interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Heartbeat loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _discover_peers(self) -> None:
        """Discover additional peers through the network."""
        current_peers = await self.get_peers()
        
        for peer in current_peers:
            try:
                # Request peer list from connected peers
                peer_list = await self.network.request_peer_list(peer.peer_id)
                
                for discovered_peer in peer_list:
                    if discovered_peer != self.node_id:
                        try:
                            await self.network.connect_to_peer_id(discovered_peer)
                        except Exception as e:
                            self.logger.debug("Failed to connect to discovered peer",
                                            peer_id=str(discovered_peer), error=str(e))
                            
            except Exception as e:
                self.logger.debug("Peer discovery failed", 
                                peer_id=str(peer.peer_id), error=str(e))
    
    async def _find_task_executor(self, task: TaskRequest) -> UUID:
        """Find the best peer to execute a task."""
        peers = await self.get_peers()
        
        # Check if we can execute locally
        if self._can_execute_task(task):
            return self.node_id
        
        # Find suitable remote executor
        best_peer = None
        best_score = -1
        
        for peer in peers:
            capabilities = self._peer_capabilities.get(peer.peer_id)
            if capabilities and self._peer_can_execute_task(task, capabilities):
                score = self._calculate_task_suitability_score(task, capabilities)
                if score > best_score:
                    best_score = score
                    best_peer = peer.peer_id
        
        if best_peer is None:
            raise RuntimeError(f"No suitable executor found for task {task.task_type}")
        
        return best_peer
    
    def _can_execute_task(self, task: TaskRequest) -> bool:
        """Check if this node can execute the given task."""
        if task.task_type not in self._task_handlers:
            return False
        
        # Check skill requirements
        for skill in task.required_skills:
            if skill not in self.capabilities.skills:
                return False
        
        return True
    
    def _peer_can_execute_task(self, task: TaskRequest, capabilities: NodeCapabilities) -> bool:
        """Check if a peer can execute the given task."""
        # Check skill requirements
        for skill in task.required_skills:
            if skill not in capabilities.skills:
                return False
        
        return True
    
    def _calculate_task_suitability_score(self, task: TaskRequest, capabilities: NodeCapabilities) -> float:
        """Calculate how suitable a peer is for executing a task."""
        score = 0.0
        
        # Base score from capabilities
        score += capabilities.cpu_cores * 0.3
        score += capabilities.memory_gb * 0.2
        score += capabilities.bandwidth_mbps * 0.1
        
        if capabilities.gpu_available:
            score += 10.0
        
        # Bonus for exact skill matches
        matching_skills = len(set(task.required_skills) & capabilities.skills)
        score += matching_skills * 5.0
        
        return score
    
    async def _execute_task_locally(self, task: TaskRequest) -> TaskResult:
        """Execute a task locally."""
        start_time = time.time()
        
        try:
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            self.logger.info("Executing task locally", 
                           task_id=str(task.task_id), task_type=task.task_type)
            
            # Execute with timeout
            result_data = await asyncio.wait_for(
                handler(task.payload),
                timeout=task.timeout_seconds
            )
            
            execution_time = time.time() - start_time
            self.metrics.tasks_completed += 1
            
            return TaskResult(
                task_id=task.task_id,
                executor_id=self.node_id,
                success=True,
                result=result_data,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("Task execution failed", 
                            task_id=str(task.task_id), error=str(e))
            
            return TaskResult(
                task_id=task.task_id,
                executor_id=self.node_id,
                success=False,
                error_message=str(e),
                execution_time_seconds=execution_time
            )
    
    async def _delegate_task(self, task: TaskRequest, executor_peer: UUID) -> TaskResult:
        """Delegate task execution to a remote peer."""
        self.logger.info("Delegating task to peer", 
                        task_id=str(task.task_id), 
                        executor_peer=str(executor_peer))
        
        # Send task to remote executor
        response = await self.network.send_request(
            executor_peer,
            "execute_task",
            task.dict(),
            timeout=task.timeout_seconds + 30
        )
        
        return TaskResult(**response)
    
    async def _analyze_network_state(self, peers: List[PeerInfo]) -> Dict[str, Any]:
        """Analyze current network state for role negotiation."""
        role_counts = {}
        total_capabilities = NodeCapabilities()
        
        for peer in peers:
            # This would be populated from peer discovery messages
            capabilities = self._peer_capabilities.get(peer.peer_id, NodeCapabilities())
            
            total_capabilities.cpu_cores += capabilities.cpu_cores
            total_capabilities.memory_gb += capabilities.memory_gb
            total_capabilities.storage_gb += capabilities.storage_gb
        
        return {
            "peer_count": len(peers),
            "role_distribution": role_counts,
            "total_capabilities": total_capabilities,
            "network_load": self.metrics.cpu_usage_percent
        }
    
    async def _calculate_optimal_role(
        self, 
        network_state: Dict[str, Any], 
        preferred_roles: Optional[List[NodeRole]] = None
    ) -> NodeRole:
        """Calculate the optimal role for this node."""
        # Simple role assignment logic based on capabilities
        if self.capabilities.cpu_cores >= 4 and self.capabilities.memory_gb >= 8:
            if NodeRole.AGGREGATOR in (preferred_roles or []):
                return NodeRole.AGGREGATOR
            return NodeRole.COORDINATOR
        elif self.capabilities.gpu_available:
            return NodeRole.TRAINER
        else:
            return NodeRole.VALIDATOR