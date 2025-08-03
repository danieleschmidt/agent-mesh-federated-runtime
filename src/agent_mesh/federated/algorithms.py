"""Federated learning algorithms implementation.

This module contains various federated learning algorithms including
FedAvg, SCAFFOLD, FedProx, and other advanced techniques.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

import numpy as np
import structlog
import torch
import torch.nn as nn
from pydantic import BaseModel


@dataclass
class AlgorithmConfig:
    """Base configuration for federated learning algorithms."""
    
    learning_rate: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    clip_norm: Optional[float] = None
    
    # Algorithm-specific parameters can be added by subclasses


@dataclass
class FedAvgConfig(AlgorithmConfig):
    """Configuration for FedAvg algorithm."""
    
    # FedAvg uses base parameters only
    pass


@dataclass
class ScaffoldConfig(AlgorithmConfig):
    """Configuration for SCAFFOLD algorithm."""
    
    server_learning_rate: float = 1.0
    control_variate_decay: float = 0.9


@dataclass
class FedProxConfig(AlgorithmConfig):
    """Configuration for FedProx algorithm."""
    
    proximal_mu: float = 0.01  # Proximal term coefficient


class TrainingResult(BaseModel):
    """Result of local training."""
    
    participant_id: UUID
    loss: float
    accuracy: Optional[float] = None
    num_samples: int
    training_time: float
    model_update: Dict[str, Any]
    control_variates: Optional[Dict[str, Any]] = None  # For SCAFFOLD


class FederatedAlgorithm(ABC):
    """Abstract base class for federated learning algorithms."""
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.logger = structlog.get_logger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def local_update(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: str = "cpu"
    ) -> TrainingResult:
        """Perform local training update."""
        pass
    
    @abstractmethod
    async def aggregate_updates(
        self,
        global_model: nn.Module,
        training_results: List[TrainingResult]
    ) -> nn.Module:
        """Aggregate local updates into global model."""
        pass
    
    def _calculate_model_diff(
        self, 
        model_before: nn.Module, 
        model_after: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Calculate difference between two models."""
        diff = {}
        state_before = model_before.state_dict()
        state_after = model_after.state_dict()
        
        for name in state_before:
            if name in state_after:
                diff[name] = state_after[name] - state_before[name]
        
        return diff
    
    def _apply_model_diff(
        self,
        base_model: nn.Module,
        model_diff: Dict[str, torch.Tensor],
        scale: float = 1.0
    ) -> nn.Module:
        """Apply model difference to base model."""
        new_state = base_model.state_dict().copy()
        
        for name, diff in model_diff.items():
            if name in new_state:
                new_state[name] = new_state[name] + scale * diff
        
        base_model.load_state_dict(new_state)
        return base_model


class FedAvgAlgorithm(FederatedAlgorithm):
    """
    Federated Averaging (FedAvg) algorithm implementation.
    
    The original federated learning algorithm that performs local SGD
    and aggregates model parameters using weighted averaging.
    """
    
    def __init__(self, config: Optional[FedAvgConfig] = None):
        super().__init__(config or FedAvgConfig())
        self.config: FedAvgConfig = self.config
    
    async def local_update(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: str = "cpu"
    ) -> TrainingResult:
        """Perform FedAvg local training."""
        model.to(device)
        model.train()
        
        # Store initial model for calculating update
        initial_state = {name: param.clone() for name, param in model.state_dict().items()}
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Gradient clipping if specified
                if self.config.clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
                
                loss.backward()
                optimizer.step()
                
                # Accumulate statistics
                epoch_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
        
        training_time = time.time() - start_time
        
        # Calculate model update
        model_update = {}
        final_state = model.state_dict()
        
        for name, initial_param in initial_state.items():
            if name in final_state:
                update = final_state[name] - initial_param
                model_update[name] = update.cpu().numpy().tolist()
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return TrainingResult(
            participant_id=UUID(int=0),  # Will be set by caller
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total_samples,
            training_time=training_time,
            model_update=model_update
        )
    
    async def aggregate_updates(
        self,
        global_model: nn.Module,
        training_results: List[TrainingResult]
    ) -> nn.Module:
        """Aggregate updates using weighted averaging."""
        if not training_results:
            return global_model
        
        # Calculate total samples for weighting
        total_samples = sum(result.num_samples for result in training_results)
        
        if total_samples == 0:
            return global_model
        
        # Initialize aggregated update
        aggregated_update = {}
        global_state = global_model.state_dict()
        
        # Get parameter names from first result
        first_result = training_results[0]
        param_names = list(first_result.model_update.keys())
        
        for param_name in param_names:
            if param_name in global_state:
                # Initialize with zeros
                param_shape = global_state[param_name].shape
                aggregated_update[param_name] = torch.zeros(param_shape)
                
                # Weighted sum of updates
                for result in training_results:
                    if param_name in result.model_update:
                        weight = result.num_samples / total_samples
                        update_tensor = torch.tensor(result.model_update[param_name])
                        aggregated_update[param_name] += weight * update_tensor
        
        # Apply aggregated update to global model
        new_state = global_model.state_dict()
        for param_name, update in aggregated_update.items():
            new_state[param_name] = new_state[param_name] + update
        
        global_model.load_state_dict(new_state)
        
        self.logger.info("FedAvg aggregation completed",
                        num_participants=len(training_results),
                        total_samples=total_samples)
        
        return global_model


class ScaffoldAlgorithm(FederatedAlgorithm):
    """
    SCAFFOLD algorithm implementation.
    
    Uses control variates to reduce client drift and improve convergence
    in heterogeneous federated learning settings.
    """
    
    def __init__(self, config: Optional[ScaffoldConfig] = None):
        super().__init__(config or ScaffoldConfig())
        self.config: ScaffoldConfig = self.config
        
        # Server control variate
        self.server_control = {}
        
        # Client control variates
        self.client_controls: Dict[UUID, Dict[str, torch.Tensor]] = {}
    
    async def local_update(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: str = "cpu",
        client_id: Optional[UUID] = None
    ) -> TrainingResult:
        """Perform SCAFFOLD local training with control variates."""
        model.to(device)
        model.train()
        
        # Get or initialize client control variate
        if client_id and client_id not in self.client_controls:
            self._initialize_client_control(client_id, model)
        
        client_control = self.client_controls.get(client_id, {}) if client_id else {}
        
        # Store initial model
        initial_state = {name: param.clone() for name, param in model.state_dict().items()}
        
        # SCAFFOLD optimizer with control variate correction
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Apply control variate correction
                self._apply_control_variate_correction(
                    model, client_control, self.server_control
                )
                
                optimizer.step()
                
                # Accumulate statistics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        training_time = time.time() - start_time
        
        # Calculate model update
        model_update = {}
        final_state = model.state_dict()
        
        for name, initial_param in initial_state.items():
            if name in final_state:
                update = final_state[name] - initial_param
                model_update[name] = update.cpu().numpy().tolist()
        
        # Update client control variate
        new_client_control = self._update_client_control(
            client_control, initial_state, final_state, epochs, len(train_loader)
        )
        
        if client_id:
            self.client_controls[client_id] = new_client_control
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return TrainingResult(
            participant_id=client_id or UUID(int=0),
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total_samples,
            training_time=training_time,
            model_update=model_update,
            control_variates={name: cv.cpu().numpy().tolist() 
                            for name, cv in new_client_control.items()}
        )
    
    async def aggregate_updates(
        self,
        global_model: nn.Module,
        training_results: List[TrainingResult]
    ) -> nn.Module:
        """Aggregate updates using SCAFFOLD algorithm."""
        if not training_results:
            return global_model
        
        # Standard FedAvg aggregation
        aggregated_model = await super().aggregate_updates(global_model, training_results)
        
        # Update server control variate
        self._update_server_control(training_results)
        
        return aggregated_model
    
    def _initialize_client_control(self, client_id: UUID, model: nn.Module) -> None:
        """Initialize control variate for new client."""
        client_control = {}
        for name, param in model.state_dict().items():
            client_control[name] = torch.zeros_like(param)
        
        self.client_controls[client_id] = client_control
    
    def _apply_control_variate_correction(
        self,
        model: nn.Module,
        client_control: Dict[str, torch.Tensor],
        server_control: Dict[str, torch.Tensor]
    ) -> None:
        """Apply control variate correction to gradients."""
        for name, param in model.named_parameters():
            if param.grad is not None and name in client_control and name in server_control:
                # SCAFFOLD correction: g = g + server_c - client_c
                correction = server_control[name] - client_control[name]
                param.grad.data += correction
    
    def _update_client_control(
        self,
        old_control: Dict[str, torch.Tensor],
        initial_state: Dict[str, torch.Tensor],
        final_state: Dict[str, torch.Tensor],
        epochs: int,
        steps_per_epoch: int
    ) -> Dict[str, torch.Tensor]:
        """Update client control variate."""
        new_control = {}
        total_steps = epochs * steps_per_epoch
        
        for name in old_control:
            if name in initial_state and name in final_state:
                # Calculate update
                model_diff = final_state[name] - initial_state[name]
                
                # Update control variate
                new_control[name] = old_control[name] - self.server_control.get(name, torch.zeros_like(old_control[name]))
                new_control[name] += model_diff / (total_steps * self.config.learning_rate)
        
        return new_control
    
    def _update_server_control(self, training_results: List[TrainingResult]) -> None:
        """Update server control variate."""
        if not training_results:
            return
        
        # Calculate average of client control variates
        total_samples = sum(result.num_samples for result in training_results)
        
        if total_samples == 0:
            return
        
        # Initialize server control if needed
        if not self.server_control:
            first_result = training_results[0]
            if first_result.control_variates:
                for name, cv_data in first_result.control_variates.items():
                    self.server_control[name] = torch.zeros_like(torch.tensor(cv_data))
        
        # Update server control variate
        for name in self.server_control:
            weighted_sum = torch.zeros_like(self.server_control[name])
            
            for result in training_results:
                if result.control_variates and name in result.control_variates:
                    weight = result.num_samples / total_samples
                    cv_tensor = torch.tensor(result.control_variates[name])
                    weighted_sum += weight * cv_tensor
            
            # Apply decay and update
            self.server_control[name] = (
                self.config.control_variate_decay * self.server_control[name] +
                (1 - self.config.control_variate_decay) * weighted_sum
            )


class FedProxAlgorithm(FederatedAlgorithm):
    """
    FedProx algorithm implementation.
    
    Adds a proximal term to local objectives to handle system heterogeneity
    and improve stability in federated learning.
    """
    
    def __init__(self, config: Optional[FedProxConfig] = None):
        super().__init__(config or FedProxConfig())
        self.config: FedProxConfig = self.config
    
    async def local_update(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        device: str = "cpu",
        global_model: Optional[nn.Module] = None
    ) -> TrainingResult:
        """Perform FedProx local training with proximal term."""
        model.to(device)
        model.train()
        
        # Store global model parameters for proximal term
        global_params = {}
        if global_model:
            global_params = {name: param.clone().to(device) 
                           for name, param in global_model.state_dict().items()}
        
        # Store initial model
        initial_state = {name: param.clone() for name, param in model.state_dict().items()}
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Add proximal term
                if global_params:
                    proximal_loss = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            proximal_loss += torch.norm(param - global_params[name]) ** 2
                    
                    loss += (self.config.proximal_mu / 2) * proximal_loss
                
                loss.backward()
                
                if self.config.clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_norm)
                
                optimizer.step()
                
                # Accumulate statistics
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct_predictions += pred.eq(target.view_as(pred)).sum().item()
        
        training_time = time.time() - start_time
        
        # Calculate model update
        model_update = {}
        final_state = model.state_dict()
        
        for name, initial_param in initial_state.items():
            if name in final_state:
                update = final_state[name] - initial_param
                model_update[name] = update.cpu().numpy().tolist()
        
        # Calculate metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return TrainingResult(
            participant_id=UUID(int=0),
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total_samples,
            training_time=training_time,
            model_update=model_update
        )
    
    async def aggregate_updates(
        self,
        global_model: nn.Module,
        training_results: List[TrainingResult]
    ) -> nn.Module:
        """Aggregate updates using weighted averaging (same as FedAvg)."""
        # FedProx uses same aggregation as FedAvg
        return await FedAvgAlgorithm(self.config).aggregate_updates(global_model, training_results)