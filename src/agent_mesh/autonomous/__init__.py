"""Autonomous Intelligence Module for Agent Mesh.

This module provides advanced autonomous capabilities including self-healing,
adaptive optimization, intelligent routing, and autonomous decision making.
"""

from .self_healing import SelfHealingManager
from .adaptive_optimizer import AdaptiveOptimizer
from .intelligent_router import IntelligentRouter
from .decision_engine import AutonomousDecisionEngine
from .learning_coordinator import ContinuousLearningCoordinator

__all__ = [
    "SelfHealingManager",
    "AdaptiveOptimizer", 
    "IntelligentRouter",
    "AutonomousDecisionEngine",
    "ContinuousLearningCoordinator"
]