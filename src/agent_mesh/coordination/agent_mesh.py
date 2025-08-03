"""Multi-agent coordination and mesh management.

This module implements the AgentMesh class for coordinating multiple
autonomous agents in complex collaborative tasks.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable, Union
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field

from ..core.mesh_node import MeshNode, NodeCapabilities, TaskRequest, TaskResult
from .task_scheduler import TaskScheduler, Task, TaskStatus


class CoordinationProtocol(Enum):
    """Coordination protocols for multi-agent systems."""
    
    CONTRACT_NET = "contract_net"
    BLACKBOARD = "blackboard"
    STIGMERGY = "stigmergy"
    MARKET_BASED = "market_based"
    HIERARCHICAL = "hierarchical"


class AgentRole(Enum):
    """Roles in multi-agent coordination."""
    
    COORDINATOR = "coordinator"
    INITIATOR = "initiator"
    PARTICIPANT = "participant"
    CONTRACTOR = "contractor"
    MANAGER = "manager"
    WORKER = "worker"
    MONITOR = "monitor"


class CollaborationStrategy(Enum):
    """Strategies for agent collaboration."""
    
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    HYBRID = "hybrid"
    EMERGENT = "emergent"


@dataclass
class AgentProfile:
    """Profile of an agent in the mesh."""
    
    agent_id: UUID
    capabilities: NodeCapabilities
    roles: Set[AgentRole] = field(default_factory=set)
    reputation: float = 1.0
    trust_score: float = 1.0
    availability: float = 1.0
    last_activity: float = field(default_factory=time.time)
    
    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    quality_score: float = 1.0
    
    # Behavioral characteristics
    cooperation_tendency: float = 0.5
    reliability_score: float = 1.0
    communication_preference: str = "direct"


@dataclass
class CollaborativeTask:
    """Task requiring multi-agent collaboration."""
    
    task_id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # Requirements
    min_agents: int = 1
    max_agents: int = 10
    required_skills: Set[str] = field(default_factory=set)
    coordination_protocol: CoordinationProtocol = CoordinationProtocol.CONTRACT_NET
    
    # Task structure
    subtasks: List[Task] = field(default_factory=list)
    dependencies: Dict[UUID, List[UUID]] = field(default_factory=dict)  # task_id -> dependencies
    
    # Constraints
    deadline: Optional[float] = None
    budget: Optional[float] = None
    quality_threshold: float = 0.8
    
    # Coordination state
    assigned_agents: Set[UUID] = field(default_factory=set)
    task_allocation: Dict[UUID, UUID] = field(default_factory=dict)  # subtask -> agent
    completion_status: Dict[UUID, TaskStatus] = field(default_factory=dict)
    
    # Results
    results: Dict[UUID, Any] = field(default_factory=dict)
    overall_quality: float = 0.0
    completion_time: Optional[float] = None


class ContractNetProtocol:
    """Implementation of Contract Net Protocol for task allocation."""
    
    def __init__(self, mesh: 'AgentMesh'):
        self.mesh = mesh
        self.logger = structlog.get_logger("contract_net")
        
        # Active negotiations
        self.active_negotiations: Dict[UUID, Dict[str, Any]] = {}
        self.bid_timeout = 30.0  # seconds
    
    async def initiate_task_announcement(
        self, 
        task: Task, 
        potential_contractors: List[UUID]
    ) -> List[Dict[str, Any]]:
        """Announce task and collect bids from potential contractors."""
        negotiation_id = uuid4()
        
        announcement = {
            "negotiation_id": str(negotiation_id),
            "task": task.dict(),
            "deadline": time.time() + self.bid_timeout,
            "evaluation_criteria": ["cost", "time", "quality"]
        }
        
        self.active_negotiations[negotiation_id] = {
            "task": task,
            "contractors": potential_contractors,
            "bids": [],
            "deadline": announcement["deadline"]
        }
        
        self.logger.info("Task announcement initiated",
                        negotiation_id=str(negotiation_id),
                        task_id=str(task.task_id),
                        contractors=len(potential_contractors))
        
        # Send announcements to potential contractors
        for contractor_id in potential_contractors:
            try:
                await self.mesh.send_message(
                    contractor_id,
                    "contract_net_announcement",
                    announcement
                )
            except Exception as e:
                self.logger.warning("Failed to send announcement",
                                  contractor_id=str(contractor_id),
                                  error=str(e))
        
        # Wait for bids
        await asyncio.sleep(self.bid_timeout)
        
        # Collect and return bids
        negotiation = self.active_negotiations.get(negotiation_id, {})
        bids = negotiation.get("bids", [])
        
        # Cleanup
        if negotiation_id in self.active_negotiations:
            del self.active_negotiations[negotiation_id]
        
        return bids
    
    async def submit_bid(
        self, 
        negotiation_id: UUID, 
        contractor_id: UUID, 
        bid: Dict[str, Any]
    ) -> bool:
        """Submit a bid for a task."""
        if negotiation_id not in self.active_negotiations:
            return False
        
        negotiation = self.active_negotiations[negotiation_id]
        
        # Check if still within deadline
        if time.time() > negotiation["deadline"]:
            return False
        
        # Check if contractor is eligible
        if contractor_id not in negotiation["contractors"]:
            return False
        
        # Add bid
        bid_entry = {
            "contractor_id": contractor_id,
            "bid": bid,
            "timestamp": time.time()
        }
        
        negotiation["bids"].append(bid_entry)
        
        self.logger.info("Bid submitted",
                        negotiation_id=str(negotiation_id),
                        contractor_id=str(contractor_id),
                        bid_cost=bid.get("cost"))
        
        return True
    
    def evaluate_bids(
        self, 
        bids: List[Dict[str, Any]], 
        evaluation_criteria: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate bids and select the best contractor."""
        if not bids:
            return None
        
        # Simple scoring based on criteria
        scored_bids = []
        
        for bid_entry in bids:
            bid = bid_entry["bid"]
            score = 0.0
            
            # Cost (lower is better)
            if "cost" in evaluation_criteria and "cost" in bid:
                max_cost = max(b["bid"].get("cost", 0) for b in bids)
                if max_cost > 0:
                    cost_score = 1.0 - (bid["cost"] / max_cost)
                    score += cost_score * 0.4
            
            # Time (lower is better)
            if "time" in evaluation_criteria and "estimated_time" in bid:
                max_time = max(b["bid"].get("estimated_time", 0) for b in bids)
                if max_time > 0:
                    time_score = 1.0 - (bid["estimated_time"] / max_time)
                    score += time_score * 0.3
            
            # Quality (higher is better)
            if "quality" in evaluation_criteria and "quality_guarantee" in bid:
                quality_score = bid["quality_guarantee"]
                score += quality_score * 0.3
            
            scored_bids.append({
                "bid_entry": bid_entry,
                "score": score
            })
        
        # Select best bid
        best_bid = max(scored_bids, key=lambda x: x["score"])
        return best_bid["bid_entry"]


class StigmergyCoordination:
    """Stigmergy-based coordination using environmental traces."""
    
    def __init__(self, mesh: 'AgentMesh'):
        self.mesh = mesh
        self.logger = structlog.get_logger("stigmergy")
        
        # Pheromone-like traces
        self.traces: Dict[str, Dict[str, float]] = {}
        self.decay_rate = 0.1
        self.evaporation_interval = 60.0  # seconds
        
        # Start evaporation process
        self._evaporation_task = None
    
    async def start(self) -> None:
        """Start stigmergy coordination."""
        self._evaporation_task = asyncio.create_task(self._evaporation_loop())
    
    async def stop(self) -> None:
        """Stop stigmergy coordination."""
        if self._evaporation_task:
            self._evaporation_task.cancel()
            try:
                await self._evaporation_task
            except asyncio.CancelledError:
                pass
    
    async def deposit_trace(
        self, 
        location: str, 
        trace_type: str, 
        intensity: float,
        depositor_id: UUID
    ) -> None:
        """Deposit a trace at a location."""
        if location not in self.traces:
            self.traces[location] = {}
        
        current_intensity = self.traces[location].get(trace_type, 0.0)
        self.traces[location][trace_type] = min(1.0, current_intensity + intensity)
        
        self.logger.debug("Trace deposited",
                         location=location,
                         trace_type=trace_type,
                         intensity=intensity,
                         depositor=str(depositor_id))
    
    async def read_traces(self, location: str) -> Dict[str, float]:
        """Read traces at a location."""
        return self.traces.get(location, {}).copy()
    
    async def find_strongest_trace(
        self, 
        locations: List[str], 
        trace_type: str
    ) -> Optional[str]:
        """Find location with strongest trace of given type."""
        strongest_location = None
        strongest_intensity = 0.0
        
        for location in locations:
            intensity = self.traces.get(location, {}).get(trace_type, 0.0)
            if intensity > strongest_intensity:
                strongest_intensity = intensity
                strongest_location = location
        
        return strongest_location
    
    async def _evaporation_loop(self) -> None:
        """Background loop for trace evaporation."""
        while True:
            try:
                await asyncio.sleep(self.evaporation_interval)
                
                # Evaporate traces
                for location in self.traces:
                    for trace_type in list(self.traces[location].keys()):
                        current = self.traces[location][trace_type]
                        new_intensity = current * (1.0 - self.decay_rate)
                        
                        if new_intensity < 0.01:  # Remove very weak traces
                            del self.traces[location][trace_type]
                        else:
                            self.traces[location][trace_type] = new_intensity
                    
                    # Remove empty locations
                    if not self.traces[location]:
                        del self.traces[location]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Evaporation loop error", error=str(e))


class AgentMesh:
    """
    Multi-agent coordination system.
    
    Manages collaboration between multiple agents using various coordination
    protocols including Contract Net, blackboard systems, and emergent
    coordination through stigmergy.
    """
    
    def __init__(
        self,
        mesh_node: MeshNode,
        default_protocol: CoordinationProtocol = CoordinationProtocol.CONTRACT_NET,
        collaboration_strategy: CollaborationStrategy = CollaborationStrategy.COOPERATIVE
    ):
        """
        Initialize agent mesh coordination system.
        
        Args:
            mesh_node: Underlying mesh network node
            default_protocol: Default coordination protocol
            collaboration_strategy: Strategy for agent collaboration
        """
        self.mesh_node = mesh_node
        self.default_protocol = default_protocol
        self.collaboration_strategy = collaboration_strategy
        
        self.logger = structlog.get_logger("agent_mesh", 
                                         node_id=str(mesh_node.node_id))
        
        # Agent management
        self.agents: Dict[UUID, AgentProfile] = {}
        self.local_agent_id = mesh_node.node_id
        
        # Task and coordination
        self.task_scheduler = TaskScheduler(mesh_node.node_id)
        self.active_collaborations: Dict[UUID, CollaborativeTask] = {}
        
        # Coordination protocols
        self.contract_net = ContractNetProtocol(self)
        self.stigmergy = StigmergyCoordination(self)
        
        # Reputation and trust system
        self.reputation_history: Dict[UUID, List[float]] = {}
        self.trust_network: Dict[UUID, Dict[UUID, float]] = {}
        
        # Performance tracking
        self.collaboration_metrics: Dict[str, Any] = {}
        
        # Message handlers
        self._setup_message_handlers()
    
    async def start(self) -> None:
        """Start the agent mesh coordination system."""
        self.logger.info("Starting agent mesh coordination")
        
        # Start coordination protocols
        await self.stigmergy.start()
        
        # Register self as an agent
        self_profile = AgentProfile(
            agent_id=self.local_agent_id,
            capabilities=self.mesh_node.capabilities,
            roles={AgentRole.PARTICIPANT}
        )
        self.agents[self.local_agent_id] = self_profile
        
        # Start task scheduler
        await self.task_scheduler.start()
        
        self.logger.info("Agent mesh coordination started")
    
    async def stop(self) -> None:
        """Stop the agent mesh coordination system."""
        self.logger.info("Stopping agent mesh coordination")
        
        # Stop coordination protocols
        await self.stigmergy.stop()
        
        # Stop task scheduler
        await self.task_scheduler.stop()
        
        self.logger.info("Agent mesh coordination stopped")
    
    async def register_agent(self, agent_profile: AgentProfile) -> None:
        """Register a new agent in the mesh."""
        self.agents[agent_profile.agent_id] = agent_profile
        
        # Initialize reputation history
        if agent_profile.agent_id not in self.reputation_history:
            self.reputation_history[agent_profile.agent_id] = [agent_profile.reputation]
        
        # Initialize trust relationships
        if agent_profile.agent_id not in self.trust_network:
            self.trust_network[agent_profile.agent_id] = {}
        
        self.logger.info("Agent registered",
                        agent_id=str(agent_profile.agent_id),
                        roles=[r.value for r in agent_profile.roles])
    
    async def unregister_agent(self, agent_id: UUID) -> None:
        """Unregister an agent from the mesh."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info("Agent unregistered", agent_id=str(agent_id))
    
    async def submit_collaborative_task(
        self, 
        task: CollaborativeTask
    ) -> UUID:
        """Submit a task requiring multi-agent collaboration."""
        self.logger.info("Collaborative task submitted",
                        task_id=str(task.task_id),
                        min_agents=task.min_agents,
                        max_agents=task.max_agents,
                        protocol=task.coordination_protocol.value)
        
        # Store task
        self.active_collaborations[task.task_id] = task
        
        # Start coordination process
        coordination_task = asyncio.create_task(
            self._coordinate_collaborative_task(task)
        )
        
        return task.task_id
    
    async def get_collaboration_status(self, task_id: UUID) -> Optional[CollaborativeTask]:
        """Get status of a collaborative task."""
        return self.active_collaborations.get(task_id)
    
    async def find_suitable_agents(
        self, 
        required_skills: Set[str],
        min_reputation: float = 0.5,
        max_agents: int = 10
    ) -> List[UUID]:
        """Find agents suitable for a task based on skills and reputation."""
        suitable_agents = []
        
        for agent_id, profile in self.agents.items():
            # Skip self
            if agent_id == self.local_agent_id:
                continue
            
            # Check reputation
            if profile.reputation < min_reputation:
                continue
            
            # Check availability
            if profile.availability < 0.1:
                continue
            
            # Check skills
            agent_skills = profile.capabilities.skills
            if required_skills and not required_skills.issubset(agent_skills):
                continue
            
            suitable_agents.append(agent_id)
        
        # Sort by reputation and capability
        suitable_agents.sort(
            key=lambda aid: (
                self.agents[aid].reputation,
                len(self.agents[aid].capabilities.skills),
                self.agents[aid].availability
            ),
            reverse=True
        )
        
        return suitable_agents[:max_agents]
    
    async def update_agent_reputation(
        self, 
        agent_id: UUID, 
        performance_score: float,
        task_completion: bool = True
    ) -> None:
        """Update agent reputation based on performance."""
        if agent_id not in self.agents:
            return
        
        profile = self.agents[agent_id]
        
        # Update basic metrics
        if task_completion:
            profile.tasks_completed += 1
        else:
            profile.tasks_failed += 1
        
        # Calculate new reputation using exponential smoothing
        alpha = 0.1  # Learning rate
        new_reputation = alpha * performance_score + (1 - alpha) * profile.reputation
        profile.reputation = max(0.0, min(1.0, new_reputation))
        
        # Update reputation history
        if agent_id not in self.reputation_history:
            self.reputation_history[agent_id] = []
        
        self.reputation_history[agent_id].append(profile.reputation)
        
        # Keep only recent history
        if len(self.reputation_history[agent_id]) > 100:
            self.reputation_history[agent_id] = self.reputation_history[agent_id][-50:]
        
        self.logger.debug("Agent reputation updated",
                         agent_id=str(agent_id),
                         new_reputation=profile.reputation,
                         performance_score=performance_score)
    
    async def send_message(
        self, 
        recipient_id: UUID, 
        message_type: str, 
        content: Dict[str, Any]
    ) -> None:
        """Send message to another agent."""
        await self.mesh_node.network.send_message(
            recipient_id, message_type, content
        )
    
    async def broadcast_message(
        self, 
        message_type: str, 
        content: Dict[str, Any]
    ) -> None:
        """Broadcast message to all agents."""
        await self.mesh_node.network.broadcast_message(message_type, content)
    
    def get_collaboration_metrics(self) -> Dict[str, Any]:
        """Get collaboration performance metrics."""
        total_agents = len(self.agents)
        active_collaborations = len(self.active_collaborations)
        
        if self.agents:
            avg_reputation = sum(a.reputation for a in self.agents.values()) / total_agents
            avg_availability = sum(a.availability for a in self.agents.values()) / total_agents
        else:
            avg_reputation = 0.0
            avg_availability = 0.0
        
        return {
            "total_agents": total_agents,
            "active_collaborations": active_collaborations,
            "average_reputation": avg_reputation,
            "average_availability": avg_availability,
            "coordination_protocols": [p.value for p in CoordinationProtocol],
            "collaboration_strategy": self.collaboration_strategy.value
        }
    
    # Private methods
    
    async def _coordinate_collaborative_task(self, task: CollaborativeTask) -> None:
        """Coordinate execution of a collaborative task."""
        try:
            self.logger.info("Starting task coordination",
                           task_id=str(task.task_id),
                           protocol=task.coordination_protocol.value)
            
            # Select coordination protocol
            if task.coordination_protocol == CoordinationProtocol.CONTRACT_NET:
                await self._contract_net_coordination(task)
            elif task.coordination_protocol == CoordinationProtocol.STIGMERGY:
                await self._stigmergy_coordination(task)
            else:
                await self._basic_coordination(task)
            
        except Exception as e:
            self.logger.error("Task coordination failed",
                            task_id=str(task.task_id),
                            error=str(e))
    
    async def _contract_net_coordination(self, task: CollaborativeTask) -> None:
        """Coordinate task using Contract Net Protocol."""
        # Find suitable contractors for each subtask
        for subtask in task.subtasks:
            suitable_agents = await self.find_suitable_agents(
                subtask.required_skills,
                max_agents=min(10, task.max_agents)
            )
            
            if not suitable_agents:
                self.logger.warning("No suitable agents found for subtask",
                                  subtask_id=str(subtask.task_id))
                continue
            
            # Initiate bidding process
            bids = await self.contract_net.initiate_task_announcement(
                subtask, suitable_agents
            )
            
            # Select best bid
            if bids:
                best_bid = self.contract_net.evaluate_bids(
                    bids, ["cost", "time", "quality"]
                )
                
                if best_bid:
                    contractor_id = best_bid["contractor_id"]
                    task.task_allocation[subtask.task_id] = contractor_id
                    task.assigned_agents.add(contractor_id)
                    
                    # Award contract
                    await self._award_contract(subtask, contractor_id, best_bid)
    
    async def _stigmergy_coordination(self, task: CollaborativeTask) -> None:
        """Coordinate task using stigmergy principles."""
        # Decompose task and deposit traces
        for subtask in task.subtasks:
            # Deposit task traces in the environment
            await self.stigmergy.deposit_trace(
                location=f"task_{subtask.task_id}",
                trace_type="work_needed",
                intensity=subtask.priority / 10.0,
                depositor_id=self.local_agent_id
            )
        
        # Let agents discover and self-assign to tasks
        # This would be implemented through periodic trace reading and decision making
    
    async def _basic_coordination(self, task: CollaborativeTask) -> None:
        """Basic task coordination without specific protocol."""
        # Simple assignment based on capabilities
        suitable_agents = await self.find_suitable_agents(
            task.required_skills,
            max_agents=task.max_agents
        )
        
        if len(suitable_agents) < task.min_agents:
            self.logger.warning("Insufficient agents for task",
                              task_id=str(task.task_id),
                              required=task.min_agents,
                              available=len(suitable_agents))
            return
        
        # Assign subtasks to agents
        for i, subtask in enumerate(task.subtasks):
            if i < len(suitable_agents):
                agent_id = suitable_agents[i]
                task.task_allocation[subtask.task_id] = agent_id
                task.assigned_agents.add(agent_id)
                
                # Send task to agent
                await self._assign_task_to_agent(subtask, agent_id)
    
    async def _award_contract(
        self, 
        task: Task, 
        contractor_id: UUID, 
        bid: Dict[str, Any]
    ) -> None:
        """Award contract to winning bidder."""
        contract = {
            "task": task.dict(),
            "contract_terms": bid["bid"],
            "deadline": task.deadline,
            "payment": bid["bid"].get("cost", 0)
        }
        
        await self.send_message(
            contractor_id,
            "contract_award",
            contract
        )
        
        self.logger.info("Contract awarded",
                        task_id=str(task.task_id),
                        contractor_id=str(contractor_id),
                        cost=bid["bid"].get("cost"))
    
    async def _assign_task_to_agent(self, task: Task, agent_id: UUID) -> None:
        """Assign task directly to agent."""
        assignment = {
            "task": task.dict(),
            "assignment_type": "direct",
            "coordinator_id": str(self.local_agent_id)
        }
        
        await self.send_message(agent_id, "task_assignment", assignment)
    
    def _setup_message_handlers(self) -> None:
        """Setup message handlers for coordination protocols."""
        # This would register handlers with the mesh node
        # For now, we'll just log that handlers would be set up
        self.logger.debug("Message handlers configured for coordination protocols")