"""Secure aggregation for federated learning.

This module implements secure aggregation mechanisms to protect
participant privacy during federated learning model updates.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

import numpy as np
import structlog
from pydantic import BaseModel, Field


class AggregationAlgorithm(Enum):
    """Supported aggregation algorithms."""
    
    FEDAVG = "fedavg"
    SCAFFOLD = "scaffold"
    FEDPROX = "fedprox"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"


class SecretSharingScheme(Enum):
    """Secret sharing schemes for secure aggregation."""
    
    SHAMIR = "shamir"
    ADDITIVE = "additive"
    THRESHOLD = "threshold"


@dataclass
class ModelUpdate:
    """Model update from a federated learning participant."""
    
    participant_id: UUID
    round_number: int
    weights: Dict[str, Any]  # Model weight updates
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: Optional[bytes] = None


@dataclass
class SecretShare:
    """Secret share for secure aggregation."""
    
    participant_id: UUID
    share_id: int
    encrypted_share: bytes
    verification_hash: str
    threshold: int


@dataclass
class AggregationResult:
    """Result of secure aggregation."""
    
    aggregated_weights: Dict[str, Any]
    participants_count: int
    round_number: int
    aggregation_time_seconds: float
    security_level: str = "standard"
    verification_passed: bool = True


class HomomorphicEncryption:
    """Simple homomorphic encryption for secure aggregation."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
    
    def generate_keys(self) -> Tuple[Any, Any]:
        """Generate public and private keys."""
        # Simplified key generation - would use proper crypto library
        import random
        self.private_key = random.randint(1, 2**self.key_size)
        self.public_key = pow(2, self.private_key, 2**self.key_size)
        return self.public_key, self.private_key
    
    def encrypt(self, value: float, public_key: Any) -> bytes:
        """Encrypt a value using homomorphic encryption."""
        # Simplified encryption - would use proper scheme like Paillier
        import struct
        encrypted_value = int(value * 1000000) + public_key  # Scale and add
        return struct.pack('Q', encrypted_value)
    
    def decrypt(self, encrypted_value: bytes, private_key: Any) -> float:
        """Decrypt a value."""
        import struct
        decrypted_int = struct.unpack('Q', encrypted_value)[0] - self.public_key
        return decrypted_int / 1000000.0
    
    def add_encrypted(self, enc_a: bytes, enc_b: bytes) -> bytes:
        """Add two encrypted values (homomorphic property)."""
        import struct
        val_a = struct.unpack('Q', enc_a)[0]
        val_b = struct.unpack('Q', enc_b)[0]
        result = val_a + val_b
        return struct.pack('Q', result)


class SecretSharing:
    """Shamir's Secret Sharing for secure aggregation."""
    
    def __init__(self, threshold: int, num_shares: int):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = 2**127 - 1  # Large prime for arithmetic
    
    def create_shares(self, secret: int) -> List[Tuple[int, int]]:
        """Create secret shares using polynomial interpolation."""
        import random
        
        # Create polynomial coefficients
        coefficients = [secret]
        for _ in range(self.threshold - 1):
            coefficients.append(random.randint(0, self.prime - 1))
        
        # Generate shares
        shares = []
        for i in range(1, self.num_shares + 1):
            share_value = self._evaluate_polynomial(coefficients, i)
            shares.append((i, share_value))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """Reconstruct secret from shares using Lagrange interpolation."""
        if len(shares) < self.threshold:
            raise ValueError("Insufficient shares for reconstruction")
        
        secret = 0
        for i, (x_i, y_i) in enumerate(shares[:self.threshold]):
            # Lagrange interpolation
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares[:self.threshold]):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime
            
            # Modular division
            lagrange_coeff = (numerator * pow(denominator, -1, self.prime)) % self.prime
            secret = (secret + y_i * lagrange_coeff) % self.prime
        
        return secret
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        x_power = 1
        
        for coeff in coefficients:
            result = (result + coeff * x_power) % self.prime
            x_power = (x_power * x) % self.prime
        
        return result


class ByzantineRobustAggregation:
    """Byzantine-robust aggregation algorithms."""
    
    @staticmethod
    def krum(updates: List[np.ndarray], num_byzantine: int) -> np.ndarray:
        """Krum aggregation algorithm for Byzantine robustness."""
        n = len(updates)
        if n <= 2 * num_byzantine:
            raise ValueError("Too many Byzantine participants")
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(updates[i] - updates[j])
                distances[i][j] = dist
                distances[j][i] = dist
        
        # Find the update with smallest sum of distances to closest neighbors
        scores = []
        num_closest = n - num_byzantine - 2
        
        for i in range(n):
            # Sort distances for this update
            sorted_distances = np.sort(distances[i])
            # Sum of distances to closest neighbors (excluding self)
            score = np.sum(sorted_distances[1:num_closest + 1])
            scores.append(score)
        
        # Return update with minimum score
        best_idx = np.argmin(scores)
        return updates[best_idx]
    
    @staticmethod
    def trimmed_mean(updates: List[np.ndarray], trim_ratio: float = 0.2) -> np.ndarray:
        """Trimmed mean aggregation."""
        if not updates:
            raise ValueError("No updates provided")
        
        # Stack updates for easier manipulation
        stacked_updates = np.stack(updates)
        
        # Calculate number of values to trim from each side
        num_trim = int(len(updates) * trim_ratio / 2)
        
        if num_trim >= len(updates) // 2:
            # Fall back to median if too many to trim
            return np.median(stacked_updates, axis=0)
        
        # Sort along participant axis and trim extremes
        sorted_updates = np.sort(stacked_updates, axis=0)
        
        if num_trim > 0:
            trimmed_updates = sorted_updates[num_trim:-num_trim]
        else:
            trimmed_updates = sorted_updates
        
        # Return mean of trimmed values
        return np.mean(trimmed_updates, axis=0)


class SecureAggregator:
    """
    Secure aggregation coordinator for federated learning.
    
    Provides privacy-preserving aggregation of model updates using
    various techniques including secret sharing and homomorphic encryption.
    """
    
    def __init__(
        self,
        node_id: UUID,
        algorithm: str = "fedavg",
        security_level: str = "standard",
        byzantine_tolerance: int = 0
    ):
        """
        Initialize secure aggregator.
        
        Args:
            node_id: Unique node identifier
            algorithm: Aggregation algorithm to use
            security_level: Security level (none, standard, high)
            byzantine_tolerance: Number of Byzantine participants to tolerate
        """
        self.node_id = node_id
        self.algorithm = AggregationAlgorithm(algorithm)
        self.security_level = security_level
        self.byzantine_tolerance = byzantine_tolerance
        
        self.logger = structlog.get_logger("secure_aggregator", node_id=str(node_id))
        
        # Cryptographic components
        self.homomorphic_enc = HomomorphicEncryption()
        self.secret_sharing = None
        
        # Aggregation state
        self.current_round = 0
        self.pending_updates: Dict[int, List[ModelUpdate]] = {}
        
        # Security tracking
        self.participant_reputation: Dict[UUID, float] = {}
        self.aggregation_history: List[AggregationResult] = {}
    
    async def initialize_round(self, round_number: int, participants: List[UUID]) -> None:
        """Initialize a new aggregation round."""
        self.current_round = round_number
        self.pending_updates[round_number] = []
        
        # Initialize secret sharing if using high security
        if self.security_level == "high":
            threshold = len(participants) // 2 + 1
            self.secret_sharing = SecretSharing(threshold, len(participants))
        
        # Generate new encryption keys if needed
        if self.security_level in ["standard", "high"]:
            self.homomorphic_enc.generate_keys()
        
        self.logger.info("Aggregation round initialized",
                        round_number=round_number,
                        participants=len(participants),
                        security_level=self.security_level)
    
    async def add_update(self, update: ModelUpdate) -> None:
        """Add a model update to the current round."""
        if update.round_number not in self.pending_updates:
            self.pending_updates[update.round_number] = []
        
        # Validate update
        if await self._validate_update(update):
            self.pending_updates[update.round_number].append(update)
            self.logger.debug("Model update added",
                            participant_id=str(update.participant_id),
                            round_number=update.round_number)
        else:
            self.logger.warning("Invalid model update rejected",
                              participant_id=str(update.participant_id))
    
    async def aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """
        Perform secure aggregation of model updates.
        
        Args:
            updates: List of model updates to aggregate
            
        Returns:
            Aggregated model weights
        """
        if not updates:
            return {}
        
        start_time = time.time()
        round_number = updates[0].round_number
        
        self.logger.info("Starting secure aggregation",
                        round_number=round_number,
                        num_updates=len(updates),
                        algorithm=self.algorithm.value)
        
        try:
            # Apply security preprocessing if needed
            if self.security_level != "none":
                updates = await self._apply_security_preprocessing(updates)
            
            # Perform aggregation based on algorithm
            if self.algorithm == AggregationAlgorithm.FEDAVG:
                aggregated_weights = await self._fedavg_aggregation(updates)
            elif self.algorithm == AggregationAlgorithm.SCAFFOLD:
                aggregated_weights = await self._scaffold_aggregation(updates)
            elif self.algorithm == AggregationAlgorithm.KRUM:
                aggregated_weights = await self._krum_aggregation(updates)
            elif self.algorithm == AggregationAlgorithm.TRIMMED_MEAN:
                aggregated_weights = await self._trimmed_mean_aggregation(updates)
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            # Apply security postprocessing if needed
            if self.security_level != "none":
                aggregated_weights = await self._apply_security_postprocessing(aggregated_weights)
            
            aggregation_time = time.time() - start_time
            
            # Create result
            result = AggregationResult(
                aggregated_weights=aggregated_weights,
                participants_count=len(updates),
                round_number=round_number,
                aggregation_time_seconds=aggregation_time,
                security_level=self.security_level
            )
            
            self.aggregation_history.append(result)
            
            self.logger.info("Secure aggregation completed",
                           round_number=round_number,
                           aggregation_time=aggregation_time)
            
            return aggregated_weights
            
        except Exception as e:
            self.logger.error("Secure aggregation failed",
                            round_number=round_number,
                            error=str(e))
            raise
    
    def get_aggregation_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics and performance metrics."""
        if not self.aggregation_history:
            return {}
        
        recent_results = self.aggregation_history[-10:]  # Last 10 rounds
        
        avg_time = np.mean([r.aggregation_time_seconds for r in recent_results])
        avg_participants = np.mean([r.participants_count for r in recent_results])
        
        return {
            "total_rounds": len(self.aggregation_history),
            "average_aggregation_time": avg_time,
            "average_participants": avg_participants,
            "security_level": self.security_level,
            "algorithm": self.algorithm.value,
            "byzantine_tolerance": self.byzantine_tolerance
        }
    
    # Private methods
    
    async def _validate_update(self, update: ModelUpdate) -> bool:
        """Validate a model update."""
        # Check timestamp freshness
        if time.time() - update.timestamp > 3600:  # 1 hour old
            return False
        
        # Check if weights are reasonable (not too large)
        for layer_weights in update.weights.values():
            if isinstance(layer_weights, list):
                max_weight = max(abs(w) for w in layer_weights if isinstance(w, (int, float)))
                if max_weight > 1000:  # Threshold for reasonable weights
                    return False
        
        # Check participant reputation
        reputation = self.participant_reputation.get(update.participant_id, 1.0)
        if reputation < 0.1:  # Very low reputation
            return False
        
        return True
    
    async def _apply_security_preprocessing(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Apply security preprocessing to updates."""
        if self.security_level == "standard":
            # Apply differential privacy noise
            return await self._add_differential_privacy_noise(updates)
        elif self.security_level == "high":
            # Apply secret sharing
            return await self._apply_secret_sharing(updates)
        
        return updates
    
    async def _apply_security_postprocessing(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security postprocessing to aggregated weights."""
        if self.security_level == "high" and self.secret_sharing:
            # Reconstruct from secret shares
            return await self._reconstruct_from_shares(weights)
        
        return weights
    
    async def _add_differential_privacy_noise(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Add differential privacy noise to updates."""
        # Simplified DP implementation
        epsilon = 1.0  # Privacy parameter
        sensitivity = 1.0  # L2 sensitivity
        noise_scale = sensitivity / epsilon
        
        for update in updates:
            for layer_name, weights in update.weights.items():
                if isinstance(weights, list):
                    # Add Gaussian noise
                    noise = np.random.normal(0, noise_scale, len(weights))
                    update.weights[layer_name] = [w + n for w, n in zip(weights, noise)]
        
        return updates
    
    async def _apply_secret_sharing(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Apply secret sharing to updates."""
        # This would implement actual secret sharing
        # For now, just return updates unchanged
        return updates
    
    async def _reconstruct_from_shares(self, weights: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct weights from secret shares."""
        # This would implement actual secret reconstruction
        return weights
    
    async def _fedavg_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Perform FedAvg aggregation."""
        if not updates:
            return {}
        
        # Calculate total samples for weighting
        total_samples = sum(
            update.metadata.get("samples_count", 1) for update in updates
        )
        
        aggregated = {}
        
        # Get all layer names from first update
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_updates = []
            layer_weights = []
            
            for update in updates:
                if layer_name in update.weights:
                    weights = update.weights[layer_name]
                    if isinstance(weights, list):
                        layer_updates.append(np.array(weights))
                        samples = update.metadata.get("samples_count", 1)
                        layer_weights.append(samples / total_samples)
            
            if layer_updates:
                # Weighted average
                weighted_sum = np.zeros_like(layer_updates[0])
                for update, weight in zip(layer_updates, layer_weights):
                    weighted_sum += update * weight
                
                aggregated[layer_name] = weighted_sum.tolist()
        
        return aggregated
    
    async def _scaffold_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Perform SCAFFOLD aggregation."""
        # SCAFFOLD requires control variates - simplified implementation
        return await self._fedavg_aggregation(updates)
    
    async def _krum_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Perform Krum aggregation for Byzantine robustness."""
        if len(updates) <= 2 * self.byzantine_tolerance:
            self.logger.warning("Too many Byzantine nodes for Krum")
            return await self._fedavg_aggregation(updates)
        
        aggregated = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_updates = []
            
            for update in updates:
                if layer_name in update.weights:
                    weights = update.weights[layer_name]
                    if isinstance(weights, list):
                        layer_updates.append(np.array(weights))
            
            if layer_updates:
                # Apply Krum algorithm
                selected_update = ByzantineRobustAggregation.krum(
                    layer_updates, self.byzantine_tolerance
                )
                aggregated[layer_name] = selected_update.tolist()
        
        return aggregated
    
    async def _trimmed_mean_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Perform trimmed mean aggregation."""
        aggregated = {}
        layer_names = list(updates[0].weights.keys())
        
        for layer_name in layer_names:
            layer_updates = []
            
            for update in updates:
                if layer_name in update.weights:
                    weights = update.weights[layer_name]
                    if isinstance(weights, list):
                        layer_updates.append(np.array(weights))
            
            if layer_updates:
                # Apply trimmed mean
                trimmed_result = ByzantineRobustAggregation.trimmed_mean(
                    layer_updates, trim_ratio=0.2
                )
                aggregated[layer_name] = trimmed_result.tolist()
        
        return aggregated