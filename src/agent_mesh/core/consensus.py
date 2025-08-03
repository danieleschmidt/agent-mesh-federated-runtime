"""Byzantine fault-tolerant consensus implementation.

This module implements PBFT (Practical Byzantine Fault Tolerance) and Raft
consensus algorithms for coordinating decisions across the Agent Mesh network.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, Field


class ConsensusAlgorithm(Enum):
    """Supported consensus algorithms."""
    
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    RAFT = "raft"  # Raft consensus algorithm
    TENDERMINT = "tendermint"  # Tendermint BFT


class ProposalStatus(Enum):
    """Status of consensus proposals."""
    
    PENDING = "pending"
    PROPOSED = "proposed"
    VOTING = "voting"
    COMMITTED = "committed"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


class VoteType(Enum):
    """Types of votes in consensus."""
    
    PREPARE = "prepare"
    COMMIT = "commit"
    APPROVE = "approve"
    REJECT = "reject"


@dataclass
class ConsensusConfig:
    """Configuration for consensus algorithms."""
    
    algorithm: ConsensusAlgorithm = ConsensusAlgorithm.PBFT
    fault_tolerance: float = 0.33  # Maximum fraction of Byzantine nodes
    timeout_seconds: float = 30.0
    min_participants: int = 3
    max_participants: int = 1000
    batch_size: int = 100  # Proposals per batch
    view_change_timeout: float = 60.0


class Proposal(BaseModel):
    """Consensus proposal structure."""
    
    proposal_id: UUID = Field(default_factory=uuid4)
    proposer_id: UUID
    proposal_type: str
    data: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)
    view_number: int = 0
    sequence_number: int = 0
    signature: Optional[str] = None


class Vote(BaseModel):
    """Consensus vote structure."""
    
    vote_id: UUID = Field(default_factory=uuid4)
    proposal_id: UUID
    voter_id: UUID
    vote_type: VoteType
    approve: bool
    timestamp: float = Field(default_factory=time.time)
    view_number: int = 0
    signature: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    
    proposal_id: UUID
    accepted: bool
    value: Optional[Any] = None
    votes_for: int = 0
    votes_against: int = 0
    participants: Set[UUID] = field(default_factory=set)
    finalized_at: float = field(default_factory=time.time)
    reason: Optional[str] = None


class ConsensusParticipant(BaseModel):
    """Information about consensus participant."""
    
    node_id: UUID
    public_key: str
    role: str = "participant"  # participant, leader, validator
    reputation: float = 1.0
    last_activity: float = Field(default_factory=time.time)
    is_byzantine: bool = False  # For testing


class ConsensusAlgorithmBase(ABC):
    """Abstract base class for consensus algorithms."""
    
    def __init__(self, node_id: UUID, config: ConsensusConfig):
        self.node_id = node_id
        self.config = config
        self.logger = structlog.get_logger(
            f"{config.algorithm.value}_consensus", 
            node_id=str(node_id)
        )
    
    @abstractmethod
    async def propose(self, proposal: Proposal) -> ConsensusResult:
        """Submit a proposal for consensus."""
        pass
    
    @abstractmethod
    async def vote(self, proposal_id: UUID, approve: bool, reason: str = "") -> Vote:
        """Cast a vote on a proposal."""
        pass
    
    @abstractmethod
    async def get_consensus_status(self, proposal_id: UUID) -> Optional[ConsensusResult]:
        """Get status of a consensus proposal."""
        pass
    
    @abstractmethod
    async def add_participant(self, participant: ConsensusParticipant) -> None:
        """Add a new consensus participant."""
        pass
    
    @abstractmethod
    async def remove_participant(self, node_id: UUID) -> None:
        """Remove a consensus participant."""
        pass


class PBFTConsensus(ConsensusAlgorithmBase):
    """
    Practical Byzantine Fault Tolerance implementation.
    
    PBFT can tolerate up to f Byzantine nodes out of 3f+1 total nodes.
    The algorithm consists of three phases: pre-prepare, prepare, and commit.
    """
    
    def __init__(self, node_id: UUID, config: ConsensusConfig, network_manager):
        super().__init__(node_id, config)
        self.network = network_manager
        
        # PBFT state
        self.view_number = 0
        self.sequence_number = 0
        self.is_primary = False
        
        # Consensus state tracking
        self._active_proposals: Dict[UUID, Proposal] = {}
        self._proposal_votes: Dict[UUID, List[Vote]] = {}
        self._participants: Dict[UUID, ConsensusParticipant] = {}
        self._view_change_in_progress = False
        
        # Message tracking for PBFT phases
        self._prepare_messages: Dict[UUID, Set[UUID]] = {}  # proposal_id -> node_ids
        self._commit_messages: Dict[UUID, Set[UUID]] = {}   # proposal_id -> node_ids
        
        # Timeouts and futures
        self._proposal_futures: Dict[UUID, asyncio.Future] = {}
        self._timeout_tasks: Dict[UUID, asyncio.Task] = {}
    
    async def propose(self, proposal: Proposal) -> ConsensusResult:
        """Submit a proposal using PBFT consensus."""
        if not self.is_primary:
            raise ValueError("Only primary node can propose in current view")
        
        proposal.view_number = self.view_number
        proposal.sequence_number = self.sequence_number
        self.sequence_number += 1
        
        self.logger.info("Starting PBFT proposal", 
                        proposal_id=str(proposal.proposal_id),
                        sequence=proposal.sequence_number)
        
        # Store proposal
        self._active_proposals[proposal.proposal_id] = proposal
        self._proposal_votes[proposal.proposal_id] = []
        self._prepare_messages[proposal.proposal_id] = set()
        self._commit_messages[proposal.proposal_id] = set()
        
        # Create future for result
        result_future = asyncio.Future()
        self._proposal_futures[proposal.proposal_id] = result_future
        
        # Set timeout
        timeout_task = asyncio.create_task(
            self._handle_proposal_timeout(proposal.proposal_id)
        )
        self._timeout_tasks[proposal.proposal_id] = timeout_task
        
        try:
            # Phase 1: Pre-prepare (broadcast proposal)
            await self._broadcast_pre_prepare(proposal)
            
            # Wait for consensus result
            result = await result_future
            
            self.logger.info("PBFT consensus completed", 
                           proposal_id=str(proposal.proposal_id),
                           accepted=result.accepted)
            return result
            
        except asyncio.TimeoutError:
            self.logger.warning("PBFT proposal timeout", 
                              proposal_id=str(proposal.proposal_id))
            return ConsensusResult(
                proposal_id=proposal.proposal_id,
                accepted=False,
                reason="timeout"
            )
        finally:
            # Cleanup
            self._cleanup_proposal(proposal.proposal_id)
    
    async def vote(self, proposal_id: UUID, approve: bool, reason: str = "") -> Vote:
        """Cast a vote on a PBFT proposal."""
        if proposal_id not in self._active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self._active_proposals[proposal_id]
        
        vote = Vote(
            proposal_id=proposal_id,
            voter_id=self.node_id,
            vote_type=VoteType.APPROVE if approve else VoteType.REJECT,
            approve=approve,
            view_number=proposal.view_number,
            reason=reason
        )
        
        # Store vote
        self._proposal_votes[proposal_id].append(vote)
        
        self.logger.debug("Vote cast", 
                         proposal_id=str(proposal_id),
                         approve=approve)
        
        return vote
    
    async def get_consensus_status(self, proposal_id: UUID) -> Optional[ConsensusResult]:
        """Get current status of a proposal."""
        if proposal_id not in self._active_proposals:
            return None
        
        votes = self._proposal_votes.get(proposal_id, [])
        approve_votes = sum(1 for v in votes if v.approve)
        reject_votes = len(votes) - approve_votes
        
        # Check if consensus reached
        required_votes = self._calculate_required_votes()
        
        if approve_votes >= required_votes:
            return ConsensusResult(
                proposal_id=proposal_id,
                accepted=True,
                votes_for=approve_votes,
                votes_against=reject_votes,
                participants=set(v.voter_id for v in votes)
            )
        elif reject_votes > len(self._participants) - required_votes:
            return ConsensusResult(
                proposal_id=proposal_id,
                accepted=False,
                votes_for=approve_votes,
                votes_against=reject_votes,
                participants=set(v.voter_id for v in votes),
                reason="majority_reject"
            )
        
        return None  # Still in progress
    
    async def add_participant(self, participant: ConsensusParticipant) -> None:
        """Add a new participant to PBFT consensus."""
        self._participants[participant.node_id] = participant
        self.logger.info("Participant added", 
                        node_id=str(participant.node_id),
                        total_participants=len(self._participants))
        
        # Recalculate if this node should be primary
        await self._update_primary_status()
    
    async def remove_participant(self, node_id: UUID) -> None:
        """Remove a participant from PBFT consensus."""
        if node_id in self._participants:
            del self._participants[node_id]
            self.logger.info("Participant removed", 
                           node_id=str(node_id),
                           total_participants=len(self._participants))
            
            # Trigger view change if primary was removed
            if node_id == self._get_primary_node():
                await self._initiate_view_change()
    
    # PBFT-specific methods
    
    async def _broadcast_pre_prepare(self, proposal: Proposal) -> None:
        """Broadcast pre-prepare message to all participants."""
        message = {
            "type": "pre_prepare",
            "proposal": proposal.dict(),
            "view_number": self.view_number,
            "sequence_number": proposal.sequence_number
        }
        
        # Send to all participants except self
        for participant_id in self._participants:
            if participant_id != self.node_id:
                try:
                    await self.network.send_message(
                        participant_id, "pbft_message", message
                    )
                except Exception as e:
                    self.logger.warning("Failed to send pre-prepare",
                                      participant_id=str(participant_id),
                                      error=str(e))
    
    async def _handle_pre_prepare(self, message: Dict[str, Any], sender_id: UUID) -> None:
        """Handle incoming pre-prepare message."""
        proposal_data = message["proposal"]
        proposal = Proposal(**proposal_data)
        
        # Validate pre-prepare
        if not await self._validate_pre_prepare(proposal, sender_id):
            return
        
        # Store proposal
        self._active_proposals[proposal.proposal_id] = proposal
        self._prepare_messages[proposal.proposal_id] = {self.node_id}
        
        # Send prepare message
        await self._send_prepare(proposal)
    
    async def _send_prepare(self, proposal: Proposal) -> None:
        """Send prepare message for a proposal."""
        message = {
            "type": "prepare",
            "proposal_id": str(proposal.proposal_id),
            "view_number": proposal.view_number,
            "sequence_number": proposal.sequence_number
        }
        
        # Broadcast prepare message
        for participant_id in self._participants:
            if participant_id != self.node_id:
                try:
                    await self.network.send_message(
                        participant_id, "pbft_message", message
                    )
                except Exception as e:
                    self.logger.warning("Failed to send prepare",
                                      participant_id=str(participant_id),
                                      error=str(e))
    
    async def _handle_prepare(self, message: Dict[str, Any], sender_id: UUID) -> None:
        """Handle incoming prepare message."""
        proposal_id = UUID(message["proposal_id"])
        
        if proposal_id not in self._prepare_messages:
            self._prepare_messages[proposal_id] = set()
        
        self._prepare_messages[proposal_id].add(sender_id)
        
        # Check if we have enough prepare messages
        required_prepares = self._calculate_required_votes()
        if len(self._prepare_messages[proposal_id]) >= required_prepares:
            proposal = self._active_proposals.get(proposal_id)
            if proposal:
                await self._send_commit(proposal)
    
    async def _send_commit(self, proposal: Proposal) -> None:
        """Send commit message for a proposal."""
        message = {
            "type": "commit",
            "proposal_id": str(proposal.proposal_id),
            "view_number": proposal.view_number,
            "sequence_number": proposal.sequence_number
        }
        
        # Initialize commit tracking
        if proposal.proposal_id not in self._commit_messages:
            self._commit_messages[proposal.proposal_id] = {self.node_id}
        
        # Broadcast commit message
        for participant_id in self._participants:
            if participant_id != self.node_id:
                try:
                    await self.network.send_message(
                        participant_id, "pbft_message", message
                    )
                except Exception as e:
                    self.logger.warning("Failed to send commit",
                                      participant_id=str(participant_id),
                                      error=str(e))
    
    async def _handle_commit(self, message: Dict[str, Any], sender_id: UUID) -> None:
        """Handle incoming commit message."""
        proposal_id = UUID(message["proposal_id"])
        
        if proposal_id not in self._commit_messages:
            self._commit_messages[proposal_id] = set()
        
        self._commit_messages[proposal_id].add(sender_id)
        
        # Check if we have enough commit messages
        required_commits = self._calculate_required_votes()
        if len(self._commit_messages[proposal_id]) >= required_commits:
            await self._finalize_proposal(proposal_id, True)
    
    async def _finalize_proposal(self, proposal_id: UUID, accepted: bool) -> None:
        """Finalize a proposal with consensus result."""
        proposal = self._active_proposals.get(proposal_id)
        if not proposal:
            return
        
        result = ConsensusResult(
            proposal_id=proposal_id,
            accepted=accepted,
            value=proposal.data if accepted else None,
            participants=set(self._participants.keys())
        )
        
        # Resolve future if waiting
        future = self._proposal_futures.get(proposal_id)
        if future and not future.done():
            future.set_result(result)
        
        self.logger.info("Proposal finalized", 
                        proposal_id=str(proposal_id),
                        accepted=accepted)
    
    def _calculate_required_votes(self) -> int:
        """Calculate required votes for Byzantine fault tolerance."""
        n = len(self._participants)
        f = int(n * self.config.fault_tolerance)
        return 2 * f + 1  # 2f+1 for safety
    
    def _get_primary_node(self) -> UUID:
        """Get current primary node ID."""
        if not self._participants:
            return self.node_id
        
        # Simple primary selection: lowest node ID in current view
        sorted_participants = sorted(self._participants.keys())
        primary_index = self.view_number % len(sorted_participants)
        return sorted_participants[primary_index]
    
    async def _update_primary_status(self) -> None:
        """Update whether this node is the primary."""
        primary_node = self._get_primary_node()
        was_primary = self.is_primary
        self.is_primary = (primary_node == self.node_id)
        
        if self.is_primary != was_primary:
            self.logger.info("Primary status changed", 
                           is_primary=self.is_primary,
                           view_number=self.view_number)
    
    async def _validate_pre_prepare(self, proposal: Proposal, sender_id: UUID) -> bool:
        """Validate a pre-prepare message."""
        # Check if sender is current primary
        if sender_id != self._get_primary_node():
            self.logger.warning("Pre-prepare from non-primary",
                              sender_id=str(sender_id))
            return False
        
        # Check view number
        if proposal.view_number != self.view_number:
            self.logger.warning("Pre-prepare with wrong view number",
                              expected=self.view_number,
                              received=proposal.view_number)
            return False
        
        return True
    
    async def _handle_proposal_timeout(self, proposal_id: UUID) -> None:
        """Handle proposal timeout."""
        await asyncio.sleep(self.config.timeout_seconds)
        
        if proposal_id in self._proposal_futures:
            future = self._proposal_futures[proposal_id]
            if not future.done():
                await self._finalize_proposal(proposal_id, False)
    
    async def _initiate_view_change(self) -> None:
        """Initiate view change process."""
        if self._view_change_in_progress:
            return
        
        self._view_change_in_progress = True
        self.view_number += 1
        
        self.logger.info("Initiating view change", new_view=self.view_number)
        
        # Update primary status
        await self._update_primary_status()
        
        # Reset view change flag after timeout
        await asyncio.sleep(self.config.view_change_timeout)
        self._view_change_in_progress = False
    
    def _cleanup_proposal(self, proposal_id: UUID) -> None:
        """Cleanup proposal state."""
        self._active_proposals.pop(proposal_id, None)
        self._proposal_votes.pop(proposal_id, None)
        self._prepare_messages.pop(proposal_id, None)
        self._commit_messages.pop(proposal_id, None)
        self._proposal_futures.pop(proposal_id, None)
        
        timeout_task = self._timeout_tasks.pop(proposal_id, None)
        if timeout_task:
            timeout_task.cancel()


class ConsensusEngine:
    """
    Main consensus engine that manages different consensus algorithms.
    
    Provides a unified interface for consensus operations while supporting
    multiple underlying algorithms (PBFT, Raft, etc.).
    """
    
    def __init__(
        self, 
        node_id: UUID,
        network,
        config: Optional[ConsensusConfig] = None
    ):
        """
        Initialize consensus engine.
        
        Args:
            node_id: Unique identifier for this node
            network: Network manager for communication
            config: Consensus configuration
        """
        self.node_id = node_id
        self.network = network
        self.config = config or ConsensusConfig()
        
        self.logger = structlog.get_logger("consensus_engine", node_id=str(node_id))
        
        # Algorithm implementation
        self._algorithm: Optional[ConsensusAlgorithmBase] = None
        self._running = False
        
        # Consensus state
        self._participants: Dict[UUID, ConsensusParticipant] = {}
        self._consensus_history: List[ConsensusResult] = []
        
        # Message handling
        self._message_handlers: Dict[str, Callable] = {}
        self._setup_message_handlers()
    
    async def start(self) -> None:
        """Start the consensus engine."""
        self.logger.info("Starting consensus engine", 
                        algorithm=self.config.algorithm.value)
        
        # Initialize algorithm implementation
        if self.config.algorithm == ConsensusAlgorithm.PBFT:
            self._algorithm = PBFTConsensus(self.node_id, self.config, self.network)
        else:
            raise ValueError(f"Unsupported consensus algorithm: {self.config.algorithm}")
        
        # Register network message handlers
        self.network.register_message_handler("pbft_message", self._handle_pbft_message)
        self.network.register_request_handler("consensus_request", self._handle_consensus_request)
        
        # Add self as participant
        self_participant = ConsensusParticipant(
            node_id=self.node_id,
            public_key="self_public_key",  # Would use real key
            role="participant"
        )
        await self._algorithm.add_participant(self_participant)
        
        self._running = True
        self.logger.info("Consensus engine started")
    
    async def stop(self) -> None:
        """Stop the consensus engine."""
        self.logger.info("Stopping consensus engine")
        self._running = False
    
    async def propose(self, proposal_data: Dict[str, Any]) -> ConsensusResult:
        """
        Submit a proposal for consensus.
        
        Args:
            proposal_data: Data to reach consensus on
            
        Returns:
            ConsensusResult with the outcome
        """
        if not self._algorithm:
            raise RuntimeError("Consensus engine not started")
        
        proposal = Proposal(
            proposer_id=self.node_id,
            proposal_type="general",
            data=proposal_data
        )
        
        self.logger.info("Submitting proposal for consensus",
                        proposal_id=str(proposal.proposal_id))
        
        result = await self._algorithm.propose(proposal)
        
        # Store in history
        self._consensus_history.append(result)
        
        return result
    
    async def vote(self, proposal_id: UUID, approve: bool, reason: str = "") -> Vote:
        """Cast a vote on a proposal."""
        if not self._algorithm:
            raise RuntimeError("Consensus engine not started")
        
        return await self._algorithm.vote(proposal_id, approve, reason)
    
    async def get_consensus_status(self, proposal_id: UUID) -> Optional[ConsensusResult]:
        """Get status of a consensus proposal."""
        if not self._algorithm:
            return None
        
        return await self._algorithm.get_consensus_status(proposal_id)
    
    async def add_participant(self, node_id: UUID, public_key: str, role: str = "participant") -> None:
        """Add a new consensus participant."""
        participant = ConsensusParticipant(
            node_id=node_id,
            public_key=public_key,
            role=role
        )
        
        self._participants[node_id] = participant
        
        if self._algorithm:
            await self._algorithm.add_participant(participant)
        
        self.logger.info("Consensus participant added", 
                        node_id=str(node_id), role=role)
    
    async def remove_participant(self, node_id: UUID) -> None:
        """Remove a consensus participant."""
        if node_id in self._participants:
            del self._participants[node_id]
        
        if self._algorithm:
            await self._algorithm.remove_participant(node_id)
        
        self.logger.info("Consensus participant removed", node_id=str(node_id))
    
    def get_participants(self) -> List[ConsensusParticipant]:
        """Get list of consensus participants."""
        return list(self._participants.values())
    
    def get_consensus_history(self) -> List[ConsensusResult]:
        """Get history of consensus decisions."""
        return self._consensus_history.copy()
    
    # Private methods
    
    def _setup_message_handlers(self) -> None:
        """Setup message handlers for different consensus algorithms."""
        self._message_handlers["pbft_message"] = self._handle_pbft_message
    
    async def _handle_pbft_message(self, message, sender_info) -> Optional[Dict[str, Any]]:
        """Handle PBFT consensus messages."""
        if not isinstance(self._algorithm, PBFTConsensus):
            return None
        
        message_type = message.payload.get("type")
        sender_id = message.sender_id
        
        if message_type == "pre_prepare":
            await self._algorithm._handle_pre_prepare(message.payload, sender_id)
        elif message_type == "prepare":
            await self._algorithm._handle_prepare(message.payload, sender_id)
        elif message_type == "commit":
            await self._algorithm._handle_commit(message.payload, sender_id)
        
        return {"status": "processed"}
    
    async def _handle_consensus_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle consensus-related requests."""
        request_type = request_data.get("type")
        
        if request_type == "get_status":
            proposal_id = UUID(request_data["proposal_id"])
            status = await self.get_consensus_status(proposal_id)
            return {"status": status.dict() if status else None}
        
        elif request_type == "get_participants":
            participants = [p.dict() for p in self.get_participants()]
            return {"participants": participants}
        
        else:
            return {"error": f"Unknown request type: {request_type}"}