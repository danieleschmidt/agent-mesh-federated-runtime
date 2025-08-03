"""Federated Learning Components.

This module contains implementations for federated learning algorithms
and secure aggregation mechanisms.
"""

from .learner import FederatedLearner
from .aggregator import SecureAggregator
from .algorithms import FedAvgAlgorithm, ScaffoldAlgorithm

__all__ = ["FederatedLearner", "SecureAggregator", "FedAvgAlgorithm", "ScaffoldAlgorithm"]