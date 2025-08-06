"""Federated Learning coordination and management.

This module implements the FederatedLearner class which coordinates
distributed machine learning across the Agent Mesh network.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID, uuid4

import numpy as np
import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

from .algorithms import FederatedAlgorithm, FedAvgAlgorithm
from .aggregator import SecureAggregator, ModelUpdate


class TrainingPhase(Enum):
    """Federated learning training phases."""
    
    INITIALIZATION = "initialization"
    TRAINING = "training"
    AGGREGATION = "aggregation"
    EVALUATION = "evaluation"
    COMPLETED = "completed"
    FAILED = "failed"


class ParticipantRole(Enum):
    """Roles for federated learning participants."""
    
    TRAINER = "trainer"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    COORDINATOR = "coordinator"


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    
    # Training parameters
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # Participation requirements
    min_participants: int = 3
    max_participants: int = 100
    participation_rate: float = 0.8  # Fraction of nodes that must participate
    
    # Aggregation settings
    aggregation_algorithm: str = "fedavg"
    secure_aggregation: bool = True
    
    # Privacy settings
    differential_privacy: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    
    # Convergence criteria
    convergence_threshold: float = 0.001
    patience: int = 10  # Rounds without improvement before stopping
    
    # Resource constraints
    max_model_size_mb: float = 100.0
    timeout_seconds: float = 600.0


class TrainingMetrics(BaseModel):
    """Metrics for federated training progress."""
    
    round_number: int
    loss: float
    accuracy: Optional[float] = None
    participants_count: int
    aggregation_time_seconds: float
    communication_cost_mb: float = 0.0
    convergence_score: float = 0.0
    timestamp: float = Field(default_factory=time.time)


@dataclass
class ParticipantInfo:
    """Information about federated learning participant."""
    
    node_id: UUID
    role: ParticipantRole
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_participation: float = field(default_factory=time.time)
    total_rounds_participated: int = 0
    average_contribution_quality: float = 1.0
    is_reliable: bool = True


class LocalTrainingResult(BaseModel):
    """Result of local training on a participant."""
    
    participant_id: UUID
    round_number: int
    model_update: Dict[str, Any]  # Serialized model parameters
    loss: float
    accuracy: Optional[float] = None
    samples_count: int
    training_time_seconds: float
    timestamp: float = Field(default_factory=time.time)


class FederatedLearner:
    """
    Federated learning coordinator and participant.
    
    Manages distributed machine learning across the Agent Mesh network,
    including training coordination, secure aggregation, and convergence
    monitoring.
    """
    
    def __init__(
        self,
        node_id: UUID,
        model_fn: Callable[[], nn.Module],
        dataset_fn: Callable[[], Any],
        config: Optional[FederatedConfig] = None,
        network_manager = None,
        consensus_engine = None
    ):
        """
        Initialize federated learner.
        
        Args:
            node_id: Unique node identifier
            model_fn: Function that returns a PyTorch model
            dataset_fn: Function that returns training dataset
            config: Federated learning configuration
            network_manager: Network manager for communication
            consensus_engine: Consensus engine for coordination
        """
        self.node_id = node_id
        self.model_fn = model_fn
        self.dataset_fn = dataset_fn
        self.config = config or FederatedConfig()
        self.network = network_manager
        self.consensus = consensus_engine
        
        self.logger = structlog.get_logger("federated_learner", node_id=str(node_id))
        
        # Training state
        self.current_round = 0
        self.phase = TrainingPhase.INITIALIZATION
        self.global_model: Optional[nn.Module] = None
        self.local_model: Optional[nn.Module] = None
        
        # Participant management
        self.participants: Dict[UUID, ParticipantInfo] = {}
        self.my_role = ParticipantRole.TRAINER
        
        # Aggregation
        self.aggregator = SecureAggregator(
            node_id=node_id,
            algorithm=self.config.aggregation_algorithm
        )
        self.algorithm: FederatedAlgorithm = FedAvgAlgorithm()
        
        # Training history
        self.training_history: List[TrainingMetrics] = []
        self.convergence_history: List[float] = []
        
        # Local training data
        self.train_dataset = None
        self.train_loader = None
        
        # Coordination state
        self._round_participants: set[UUID] = set()
        self._round_updates: Dict[UUID, LocalTrainingResult] = {}
        self._training_task: Optional[asyncio.Task] = None
        self._is_coordinator = False
        
        # Message handling
        self._pending_invitations: List[Dict[str, Any]] = []
        self._pending_model_updates: List[Dict[str, Any]] = []
    
    async def start_training(self) -> None:
        """Start federated learning process."""
        self.logger.info("Starting federated learning", 
                        config=self.config.__dict__)
        
        try:
            # Initialize model and data
            await self._initialize_training()
            
            # Determine role and coordination
            await self._negotiate_role()
            
            # Start training loop
            if self._is_coordinator:
                self._training_task = asyncio.create_task(self._coordinator_loop())
            else:
                self._training_task = asyncio.create_task(self._participant_loop())
            
            await self._training_task
            
        except Exception as e:
            self.logger.error("Federated training failed", error=str(e))
            self.phase = TrainingPhase.FAILED
            raise
    
    async def stop_training(self) -> None:
        """Stop federated learning process."""
        self.logger.info("Stopping federated learning")
        
        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass
        
        self.phase = TrainingPhase.COMPLETED
    
    async def join_training_round(self, round_number: int) -> Optional[LocalTrainingResult]:
        """Participate in a federated training round."""
        if not self._can_participate():
            return None
        
        self.logger.info("Joining training round", round_number=round_number)
        
        try:
            # Perform local training
            result = await self._perform_local_training(round_number)
            
            # Update participation tracking
            self.participants[self.node_id].total_rounds_participated += 1
            self.participants[self.node_id].last_participation = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error("Local training failed", 
                            round_number=round_number, error=str(e))
            return None
    
    async def coordinate_round(self, round_number: int) -> TrainingMetrics:
        """Coordinate a federated training round as coordinator."""
        if not self._is_coordinator:
            raise ValueError("Only coordinator can coordinate rounds")
        
        self.logger.info("Coordinating training round", round_number=round_number)
        
        start_time = time.time()
        self.current_round = round_number
        self.phase = TrainingPhase.TRAINING
        
        try:
            # Select participants for this round
            selected_participants = await self._select_round_participants()
            
            # Broadcast training invitation
            await self._broadcast_training_invitation(round_number, selected_participants)
            
            # Collect local training results
            round_updates = await self._collect_training_results(
                round_number, selected_participants
            )
            
            # Perform secure aggregation
            self.phase = TrainingPhase.AGGREGATION
            aggregated_model = await self._perform_aggregation(round_updates)
            
            # Update global model
            self.global_model = aggregated_model
            
            # Broadcast updated model
            await self._broadcast_global_model(aggregated_model)
            
            # Calculate metrics
            self.phase = TrainingPhase.EVALUATION
            metrics = await self._calculate_round_metrics(
                round_number, round_updates, time.time() - start_time
            )
            
            self.training_history.append(metrics)
            
            # Check convergence
            converged = await self._check_convergence(metrics)
            if converged:
                self.phase = TrainingPhase.COMPLETED
                self.logger.info("Training converged", round_number=round_number)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Round coordination failed", 
                            round_number=round_number, error=str(e))
            self.phase = TrainingPhase.FAILED
            raise
    
    def get_training_metrics(self) -> List[TrainingMetrics]:
        """Get training history and metrics."""
        return self.training_history.copy()
    
    def get_global_model(self) -> Optional[nn.Module]:
        """Get current global model."""
        return self.global_model
    
    def is_training_complete(self) -> bool:
        """Check if training is complete."""
        return self.phase in [TrainingPhase.COMPLETED, TrainingPhase.FAILED]
    
    # Private methods
    
    async def _initialize_training(self) -> None:
        """Initialize training components."""
        self.logger.info("Initializing federated training")
        
        # Initialize global model
        self.global_model = self.model_fn()
        self.local_model = self.model_fn()
        
        # Load local dataset
        self.train_dataset = self.dataset_fn()
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Initialize aggregation algorithm
        if self.config.aggregation_algorithm == "fedavg":
            self.algorithm = FedAvgAlgorithm()
        elif self.config.aggregation_algorithm == "scaffold":
            from .algorithms import ScaffoldAlgorithm
            self.algorithm = ScaffoldAlgorithm()
        
        # Add self as participant
        self.participants[self.node_id] = ParticipantInfo(
            node_id=self.node_id,
            role=ParticipantRole.TRAINER,
            capabilities={"model_size": self._get_model_size()}
        )
        
        self.phase = TrainingPhase.TRAINING
    
    async def _negotiate_role(self) -> None:
        """Negotiate role in federated learning."""
        # Simple role assignment based on capabilities
        if len(self.participants) == 1:  # First node becomes coordinator
            self._is_coordinator = True
            self.my_role = ParticipantRole.COORDINATOR
        else:
            # Use consensus to elect coordinator
            if self.consensus:
                proposal = {
                    "type": "coordinator_election",
                    "candidate": str(self.node_id),
                    "capabilities": self.participants[self.node_id].capabilities
                }
                
                result = await self.consensus.propose(proposal)
                self._is_coordinator = result.accepted
                
                if self._is_coordinator:
                    self.my_role = ParticipantRole.COORDINATOR
        
        self.logger.info("Role negotiated", 
                        role=self.my_role.value,
                        is_coordinator=self._is_coordinator)
    
    async def _coordinator_loop(self) -> None:
        """Main training loop for coordinator."""
        self.logger.info("Starting coordinator training loop")
        
        for round_num in range(1, self.config.rounds + 1):
            try:
                metrics = await self.coordinate_round(round_num)
                
                self.logger.info("Round completed",
                               round_number=round_num,
                               loss=metrics.loss,
                               participants=metrics.participants_count)
                
                if self.is_training_complete():
                    break
                    
            except Exception as e:
                self.logger.error("Coordinator round failed", 
                                round_number=round_num, error=str(e))
                break
        
        self.logger.info("Coordinator training loop completed")
    
    async def _participant_loop(self) -> None:
        """Main training loop for participants."""
        self.logger.info("Starting participant training loop")
        
        while not self.is_training_complete():
            try:
                # Wait for training invitation
                invitation = await self._wait_for_training_invitation()
                
                if invitation:
                    round_number = invitation["round_number"]
                    
                    # Participate in round
                    result = await self.join_training_round(round_number)
                    
                    if result:
                        # Send result to coordinator
                        await self._send_training_result(result)
                    
                    # Wait for global model update
                    await self._wait_for_global_model_update()
                
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Participant loop error", error=str(e))
                await asyncio.sleep(5)
        
        self.logger.info("Participant training loop completed")
    
    async def _perform_local_training(self, round_number: int) -> LocalTrainingResult:
        """Perform local training for one round."""
        self.logger.info("Starting local training", round_number=round_number)
        
        start_time = time.time()
        
        # Copy global model to local model
        if self.global_model:
            self.local_model.load_state_dict(self.global_model.state_dict())
        
        # Training setup
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        training_time = time.time() - start_time
        
        # Calculate model update (difference from global model)
        model_update = self._calculate_model_update()
        
        result = LocalTrainingResult(
            participant_id=self.node_id,
            round_number=round_number,
            model_update=model_update,
            loss=avg_loss,
            samples_count=total_samples,
            training_time_seconds=training_time
        )
        
        self.logger.info("Local training completed",
                        round_number=round_number,
                        loss=avg_loss,
                        training_time=training_time)
        
        return result
    
    def _calculate_model_update(self) -> Dict[str, Any]:
        """Calculate model update as difference from global model."""
        if not self.global_model or not self.local_model:
            return {}
        
        update = {}
        global_state = self.global_model.state_dict()
        local_state = self.local_model.state_dict()
        
        for name, global_param in global_state.items():
            if name in local_state:
                # Calculate difference
                diff = local_state[name] - global_param
                update[name] = diff.cpu().numpy().tolist()
        
        return update
    
    async def _select_round_participants(self) -> List[UUID]:
        """Select participants for training round."""
        available_participants = [
            pid for pid, info in self.participants.items()
            if info.is_reliable and pid != self.node_id
        ]
        
        # Apply participation rate
        num_participants = min(
            len(available_participants),
            int(len(self.participants) * self.config.participation_rate)
        )
        
        # Simple random selection (could be more sophisticated)
        import random
        selected = random.sample(available_participants, 
                                min(num_participants, len(available_participants)))
        
        # Always include self if coordinator
        if self._is_coordinator:
            selected.append(self.node_id)
        
        return selected
    
    async def _broadcast_training_invitation(
        self, 
        round_number: int, 
        participants: List[UUID]
    ) -> None:
        """Broadcast training invitation to selected participants."""
        invitation = {
            "type": "training_invitation",
            "round_number": round_number,
            "coordinator_id": str(self.node_id),
            "global_model": self._serialize_model(self.global_model),
            "config": self.config.__dict__
        }
        
        for participant_id in participants:
            if participant_id != self.node_id:
                try:
                    await self.network.send_message(
                        participant_id,
                        "federated_learning",
                        invitation
                    )
                except Exception as e:
                    self.logger.warning("Failed to send invitation",
                                      participant_id=str(participant_id),
                                      error=str(e))
    
    async def _collect_training_results(
        self,
        round_number: int,
        participants: List[UUID]
    ) -> Dict[UUID, LocalTrainingResult]:
        """Collect training results from participants."""
        self.logger.info("Collecting training results", 
                        round_number=round_number,
                        expected_participants=len(participants))
        
        results = {}
        timeout = self.config.timeout_seconds
        
        # Perform own training if coordinator participates
        if self.node_id in participants:
            own_result = await self.join_training_round(round_number)
            if own_result:
                results[self.node_id] = own_result
        
        # Wait for results from other participants
        end_time = time.time() + timeout
        
        while time.time() < end_time and len(results) < len(participants):
            # Check for new results (would be received via network messages)
            await asyncio.sleep(1)
        
        self.logger.info("Training results collected",
                        round_number=round_number,
                        received_results=len(results),
                        expected_results=len(participants))
        
        return results
    
    async def _perform_aggregation(
        self, 
        round_updates: Dict[UUID, LocalTrainingResult]
    ) -> nn.Module:
        """Perform secure aggregation of model updates."""
        self.logger.info("Starting model aggregation", 
                        num_updates=len(round_updates))
        
        # Convert results to ModelUpdate format
        model_updates = []
        for participant_id, result in round_updates.items():
            update = ModelUpdate(
                participant_id=participant_id,
                round_number=result.round_number,
                weights=result.model_update,
                metadata={
                    "loss": result.loss,
                    "samples_count": result.samples_count
                }
            )
            model_updates.append(update)
        
        # Perform aggregation
        aggregated_weights = await self.aggregator.aggregate_updates(model_updates)
        
        # Apply aggregated weights to global model
        new_global_model = self.model_fn()
        new_global_model.load_state_dict(self.global_model.state_dict())
        
        # Apply updates
        current_state = new_global_model.state_dict()
        for name, param_tensor in current_state.items():
            if name in aggregated_weights:
                # Convert back to tensor and add to current weights
                update_tensor = torch.tensor(aggregated_weights[name])
                current_state[name] = param_tensor + update_tensor
        
        new_global_model.load_state_dict(current_state)
        
        self.logger.info("Model aggregation completed")
        return new_global_model
    
    async def _broadcast_global_model(self, model: nn.Module) -> None:
        """Broadcast updated global model to all participants."""
        message = {
            "type": "global_model_update",
            "round_number": self.current_round,
            "model": self._serialize_model(model)
        }
        
        await self.network.broadcast_message("federated_learning", message)
    
    async def _calculate_round_metrics(
        self,
        round_number: int,
        round_updates: Dict[UUID, LocalTrainingResult],
        aggregation_time: float
    ) -> TrainingMetrics:
        """Calculate metrics for completed training round."""
        if not round_updates:
            return TrainingMetrics(
                round_number=round_number,
                loss=float('inf'),
                participants_count=0,
                aggregation_time_seconds=aggregation_time
            )
        
        # Calculate weighted average loss
        total_loss = 0.0
        total_samples = 0
        
        for result in round_updates.values():
            total_loss += result.loss * result.samples_count
            total_samples += result.samples_count
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        # Calculate communication cost (simplified)
        model_size_mb = self._get_model_size() / (1024 * 1024)
        communication_cost = model_size_mb * len(round_updates) * 2  # Send + receive
        
        return TrainingMetrics(
            round_number=round_number,
            loss=avg_loss,
            participants_count=len(round_updates),
            aggregation_time_seconds=aggregation_time,
            communication_cost_mb=communication_cost
        )
    
    async def _check_convergence(self, metrics: TrainingMetrics) -> bool:
        """Check if training has converged."""
        self.convergence_history.append(metrics.loss)
        
        # Need at least 2 rounds to check convergence
        if len(self.convergence_history) < 2:
            return False
        
        # Check if improvement is below threshold
        recent_losses = self.convergence_history[-self.config.patience:]
        if len(recent_losses) >= self.config.patience:
            improvement = max(recent_losses) - min(recent_losses)
            if improvement < self.config.convergence_threshold:
                return True
        
        return False
    
    def _can_participate(self) -> bool:
        """Check if this node can participate in training."""
        return (
            self.phase == TrainingPhase.TRAINING and
            self.local_model is not None and
            self.train_loader is not None
        )
    
    def _get_model_size(self) -> int:
        """Get model size in bytes."""
        if not self.global_model:
            return 0
        
        param_size = 0
        for param in self.global_model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.global_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size
    
    def _serialize_model(self, model: nn.Module) -> Dict[str, Any]:
        """Serialize model for network transmission."""
        if not model:
            return {}
        
        state_dict = model.state_dict()
        serialized = {}
        
        for name, tensor in state_dict.items():
            serialized[name] = tensor.cpu().numpy().tolist()
        
        return serialized
    
    def _deserialize_model(self, serialized_model: Dict[str, Any]) -> nn.Module:
        """Deserialize model from network transmission."""
        model = self.model_fn()
        
        if serialized_model:
            state_dict = {}
            for name, tensor_list in serialized_model.items():
                state_dict[name] = torch.tensor(tensor_list)
            
            model.load_state_dict(state_dict)
        
        return model
    
    async def _wait_for_training_invitation(self) -> Optional[Dict[str, Any]]:
        """Wait for training invitation from coordinator."""
        # Wait for federated learning message from coordinator
        if hasattr(self, '_pending_invitations'):
            if self._pending_invitations:
                return self._pending_invitations.pop(0)
        
        # Poll for invitations with timeout
        for _ in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            if hasattr(self, '_pending_invitations') and self._pending_invitations:
                return self._pending_invitations.pop(0)
        
        return None
    
    async def _send_training_result(self, result: LocalTrainingResult) -> None:
        """Send training result to coordinator."""
        if not self.network:
            return
        
        # Find coordinator ID from participants
        coordinator_id = None
        for pid, info in self.participants.items():
            if info.role == ParticipantRole.COORDINATOR:
                coordinator_id = pid
                break
        
        if coordinator_id:
            message = {
                "type": "training_result",
                "result": result.dict()
            }
            
            try:
                await self.network.send_message(
                    coordinator_id,
                    "federated_learning",
                    message
                )
            except Exception as e:
                self.logger.error("Failed to send training result", error=str(e))
    
    async def _wait_for_global_model_update(self) -> None:
        """Wait for global model update from coordinator."""
        # Wait for global model update message
        if hasattr(self, '_pending_model_updates'):
            if self._pending_model_updates:
                update = self._pending_model_updates.pop(0)
                self.global_model = self._deserialize_model(update.get('model', {}))
                return
        
        # Poll for model updates with timeout
        for _ in range(60):  # Wait up to 60 seconds
            await asyncio.sleep(1)
            if hasattr(self, '_pending_model_updates') and self._pending_model_updates:
                update = self._pending_model_updates.pop(0)
                self.global_model = self._deserialize_model(update.get('model', {}))
                return