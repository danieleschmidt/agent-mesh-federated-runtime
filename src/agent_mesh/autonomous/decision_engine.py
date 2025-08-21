"""Autonomous Decision Engine with reinforcement learning capabilities."""

import asyncio
import logging
import time
import json
import statistics
import random
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of autonomous decisions."""
    RESOURCE_ALLOCATION = "resource_allocation"
    LOAD_BALANCING = "load_balancing"
    SCALING = "scaling"
    ROUTING = "routing"
    SECURITY = "security"
    MAINTENANCE = "maintenance"
    OPTIMIZATION = "optimization"

class ActionType(Enum):
    """Types of actions the system can take."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REALLOCATE_RESOURCES = "reallocate_resources"
    REDIRECT_TRAFFIC = "redirect_traffic"
    ENABLE_CACHE = "enable_cache"
    DISABLE_CACHE = "disable_cache"
    INCREASE_TIMEOUT = "increase_timeout"
    DECREASE_TIMEOUT = "decrease_timeout"
    START_MAINTENANCE = "start_maintenance"
    STOP_MAINTENANCE = "stop_maintenance"
    NO_ACTION = "no_action"

@dataclass
class SystemState:
    """Current system state representation."""
    timestamp: float = field(default_factory=time.time)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    queue_length: int = 0
    response_time: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class Decision:
    """Decision made by the autonomous system."""
    decision_id: str
    decision_type: DecisionType
    action: ActionType
    confidence: float
    reasoning: str
    state_snapshot: SystemState
    expected_impact: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

@dataclass
class DecisionOutcome:
    """Outcome of a decision for learning."""
    decision_id: str
    success: bool
    actual_impact: Dict[str, float]
    reward: float
    timestamp: float = field(default_factory=time.time)

class AutonomousDecisionEngine:
    """Reinforcement learning-based autonomous decision engine."""
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        exploration_rate: float = 0.1,
        discount_factor: float = 0.9,
        decision_interval: float = 30.0
    ):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.decision_interval = decision_interval
        
        # System state
        self.current_state = SystemState()
        self.state_history: List[SystemState] = []
        
        # Decision making
        self.decision_history: List[Decision] = []
        self.outcome_history: List[DecisionOutcome] = []
        self.pending_decisions: Dict[str, Decision] = {}
        
        # Q-learning components
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.state_action_counts: Dict[str, Dict[str, int]] = {}
        
        # Reward function weights
        self.reward_weights = {
            "cpu_improvement": 1.0,
            "memory_improvement": 1.0,
            "latency_improvement": 2.0,
            "error_rate_improvement": 3.0,
            "response_time_improvement": 2.0,
            "stability_bonus": 0.5,
            "action_penalty": -0.1  # Small penalty for taking action
        }
        
        # Decision rules and constraints
        self.decision_rules: Dict[DecisionType, List[Callable]] = {}
        self.action_constraints: Dict[ActionType, Dict[str, Any]] = {}
        
        # State management
        self.is_running = False
        self._decision_task: Optional[asyncio.Task] = None
        
        # Initialize decision rules
        self._initialize_decision_rules()
        self._initialize_action_constraints()
    
    def _initialize_decision_rules(self):
        """Initialize decision-making rules."""
        
        # Resource allocation rules
        def high_cpu_rule(state: SystemState) -> List[ActionType]:
            if state.cpu_usage > 80:
                return [ActionType.SCALE_UP, ActionType.REALLOCATE_RESOURCES]
            elif state.cpu_usage < 20:
                return [ActionType.SCALE_DOWN]
            return [ActionType.NO_ACTION]
        
        def high_memory_rule(state: SystemState) -> List[ActionType]:
            if state.memory_usage > 85:
                return [ActionType.SCALE_UP, ActionType.ENABLE_CACHE]
            elif state.memory_usage < 30:
                return [ActionType.SCALE_DOWN, ActionType.DISABLE_CACHE]
            return [ActionType.NO_ACTION]
        
        def high_latency_rule(state: SystemState) -> List[ActionType]:
            if state.network_latency > 100:  # 100ms threshold
                return [ActionType.REDIRECT_TRAFFIC, ActionType.INCREASE_TIMEOUT]
            elif state.network_latency < 20:
                return [ActionType.DECREASE_TIMEOUT]
            return [ActionType.NO_ACTION]
        
        def high_error_rate_rule(state: SystemState) -> List[ActionType]:
            if state.error_rate > 0.05:  # 5% error rate
                return [ActionType.START_MAINTENANCE, ActionType.REDIRECT_TRAFFIC]
            return [ActionType.NO_ACTION]
        
        self.decision_rules = {
            DecisionType.RESOURCE_ALLOCATION: [high_cpu_rule, high_memory_rule],
            DecisionType.LOAD_BALANCING: [high_latency_rule],
            DecisionType.SCALING: [high_cpu_rule, high_memory_rule],
            DecisionType.SECURITY: [high_error_rate_rule],
            DecisionType.MAINTENANCE: [high_error_rate_rule]
        }
    
    def _initialize_action_constraints(self):
        """Initialize action constraints and cooldowns."""
        self.action_constraints = {
            ActionType.SCALE_UP: {
                "cooldown": 300,  # 5 minutes
                "max_concurrent": 1,
                "prerequisites": []
            },
            ActionType.SCALE_DOWN: {
                "cooldown": 600,  # 10 minutes
                "max_concurrent": 1,
                "prerequisites": []
            },
            ActionType.REALLOCATE_RESOURCES: {
                "cooldown": 180,  # 3 minutes
                "max_concurrent": 2,
                "prerequisites": []
            },
            ActionType.REDIRECT_TRAFFIC: {
                "cooldown": 60,  # 1 minute
                "max_concurrent": 3,
                "prerequisites": []
            },
            ActionType.START_MAINTENANCE: {
                "cooldown": 3600,  # 1 hour
                "max_concurrent": 1,
                "prerequisites": []
            }
        }
    
    def update_state(
        self,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        network_latency: Optional[float] = None,
        request_rate: Optional[float] = None,
        error_rate: Optional[float] = None,
        active_connections: Optional[int] = None,
        queue_length: Optional[int] = None,
        response_time: Optional[float] = None,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Update current system state."""
        if cpu_usage is not None:
            self.current_state.cpu_usage = cpu_usage
        if memory_usage is not None:
            self.current_state.memory_usage = memory_usage
        if network_latency is not None:
            self.current_state.network_latency = network_latency
        if request_rate is not None:
            self.current_state.request_rate = request_rate
        if error_rate is not None:
            self.current_state.error_rate = error_rate
        if active_connections is not None:
            self.current_state.active_connections = active_connections
        if queue_length is not None:
            self.current_state.queue_length = queue_length
        if response_time is not None:
            self.current_state.response_time = response_time
        if custom_metrics is not None:
            self.current_state.custom_metrics.update(custom_metrics)
        
        self.current_state.timestamp = time.time()
        
        # Add to history
        self.state_history.append(self.current_state)
        
        # Keep history manageable
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-500:]
    
    def _state_to_key(self, state: SystemState) -> str:
        """Convert system state to a discrete key for Q-table."""
        # Discretize continuous values
        cpu_bucket = int(state.cpu_usage / 10) * 10  # 0, 10, 20, ..., 90, 100+
        memory_bucket = int(state.memory_usage / 10) * 10
        latency_bucket = int(state.network_latency / 50) * 50  # 0, 50, 100, 150+
        error_bucket = int(state.error_rate * 100 / 5) * 5  # 0, 5, 10% etc
        
        return f"cpu:{cpu_bucket},mem:{memory_bucket},lat:{latency_bucket},err:{error_bucket}"
    
    def _get_possible_actions(self, decision_type: DecisionType, state: SystemState) -> List[ActionType]:
        """Get possible actions for given decision type and state."""
        if decision_type not in self.decision_rules:
            return [ActionType.NO_ACTION]
        
        all_actions = set()
        
        # Apply decision rules
        for rule in self.decision_rules[decision_type]:
            try:
                actions = rule(state)
                all_actions.update(actions)
            except Exception as e:
                logger.error(f"Error applying decision rule: {e}")
        
        # Filter by constraints
        valid_actions = []
        current_time = time.time()
        
        for action in all_actions:
            if self._is_action_allowed(action, current_time):
                valid_actions.append(action)
        
        return valid_actions if valid_actions else [ActionType.NO_ACTION]
    
    def _is_action_allowed(self, action: ActionType, current_time: float) -> bool:
        """Check if action is allowed based on constraints."""
        if action not in self.action_constraints:
            return True
        
        constraints = self.action_constraints[action]
        
        # Check cooldown
        cooldown = constraints.get("cooldown", 0)
        if cooldown > 0:
            recent_actions = [
                d for d in self.decision_history[-10:]  # Check last 10 decisions
                if d.action == action and current_time - d.timestamp < cooldown
            ]
            if recent_actions:
                return False
        
        # Check max concurrent
        max_concurrent = constraints.get("max_concurrent", float('inf'))
        if max_concurrent < float('inf'):
            concurrent_count = sum(
                1 for d in self.pending_decisions.values()
                if d.action == action
            )
            if concurrent_count >= max_concurrent:
                return False
        
        return True
    
    def _select_action(self, state: SystemState, possible_actions: List[ActionType]) -> ActionType:
        """Select action using epsilon-greedy Q-learning."""
        state_key = self._state_to_key(state)
        
        # Initialize Q-values if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            self.state_action_counts[state_key] = {}
        
        for action in possible_actions:
            action_key = action.value
            if action_key not in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0
                self.state_action_counts[state_key][action_key] = 0
        
        # Epsilon-greedy selection
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            return random.choice(possible_actions)
        else:
            # Exploit: choose action with highest Q-value
            best_action = max(
                possible_actions,
                key=lambda a: self.q_table[state_key][a.value]
            )
            return best_action
    
    def _calculate_confidence(self, state: SystemState, action: ActionType) -> float:
        """Calculate confidence in the decision."""
        state_key = self._state_to_key(state)
        action_key = action.value
        
        if state_key not in self.q_table or action_key not in self.q_table[state_key]:
            return 0.1  # Low confidence for unknown state-action pairs
        
        q_value = self.q_table[state_key][action_key]
        experience_count = self.state_action_counts[state_key][action_key]
        
        # Confidence based on Q-value and experience
        experience_factor = min(1.0, experience_count / 10.0)  # More experience = higher confidence
        q_value_factor = max(0.1, min(1.0, (q_value + 1.0) / 2.0))  # Normalize Q-value
        
        return experience_factor * q_value_factor
    
    def _generate_reasoning(self, state: SystemState, action: ActionType) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []
        
        # State analysis
        if state.cpu_usage > 80:
            reasoning_parts.append(f"High CPU usage ({state.cpu_usage:.1f}%)")
        if state.memory_usage > 80:
            reasoning_parts.append(f"High memory usage ({state.memory_usage:.1f}%)")
        if state.network_latency > 100:
            reasoning_parts.append(f"High network latency ({state.network_latency:.1f}ms)")
        if state.error_rate > 0.05:
            reasoning_parts.append(f"High error rate ({state.error_rate*100:.1f}%)")
        
        # Action justification
        action_justifications = {
            ActionType.SCALE_UP: "to handle increased load",
            ActionType.SCALE_DOWN: "to reduce resource costs",
            ActionType.REALLOCATE_RESOURCES: "to optimize resource utilization",
            ActionType.REDIRECT_TRAFFIC: "to improve response times",
            ActionType.ENABLE_CACHE: "to reduce memory pressure",
            ActionType.DISABLE_CACHE: "to free up memory",
            ActionType.INCREASE_TIMEOUT: "to accommodate higher latency",
            ActionType.DECREASE_TIMEOUT: "to optimize response handling",
            ActionType.START_MAINTENANCE: "to address system issues",
            ActionType.NO_ACTION: "as system is operating within normal parameters"
        }
        
        justification = action_justifications.get(action, "for system optimization")
        
        if reasoning_parts:
            return f"Detected: {', '.join(reasoning_parts)}. Taking action {justification}."
        else:
            return f"System analysis suggests {action.value} {justification}."
    
    def _estimate_impact(self, state: SystemState, action: ActionType) -> Dict[str, float]:
        """Estimate expected impact of the action."""
        impact = {}
        
        # Impact estimations based on action type
        if action == ActionType.SCALE_UP:
            impact = {
                "cpu_usage": -10.0,  # Expected reduction
                "memory_usage": -5.0,
                "response_time": -20.0,
                "error_rate": -0.01
            }
        elif action == ActionType.SCALE_DOWN:
            impact = {
                "cpu_usage": 5.0,  # Expected increase
                "memory_usage": 3.0,
                "response_time": 10.0
            }
        elif action == ActionType.REALLOCATE_RESOURCES:
            impact = {
                "cpu_usage": -5.0,
                "memory_usage": -3.0,
                "response_time": -10.0
            }
        elif action == ActionType.REDIRECT_TRAFFIC:
            impact = {
                "network_latency": -20.0,
                "response_time": -15.0
            }
        elif action == ActionType.ENABLE_CACHE:
            impact = {
                "memory_usage": 10.0,
                "response_time": -30.0,
                "cpu_usage": -5.0
            }
        elif action == ActionType.DISABLE_CACHE:
            impact = {
                "memory_usage": -15.0,
                "response_time": 25.0,
                "cpu_usage": 3.0
            }
        else:
            impact = {}  # No expected impact
        
        return impact
    
    async def make_decision(self, decision_type: DecisionType) -> Optional[Decision]:
        """Make an autonomous decision."""
        try:
            current_time = time.time()
            
            # Get possible actions
            possible_actions = self._get_possible_actions(decision_type, self.current_state)
            
            if not possible_actions or possible_actions == [ActionType.NO_ACTION]:
                return None
            
            # Select action using Q-learning
            selected_action = self._select_action(self.current_state, possible_actions)
            
            # Calculate confidence
            confidence = self._calculate_confidence(self.current_state, selected_action)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(self.current_state, selected_action)
            
            # Estimate impact
            expected_impact = self._estimate_impact(self.current_state, selected_action)
            
            # Create decision
            decision_id = f"decision_{int(current_time)}_{random.randint(1000, 9999)}"
            decision = Decision(
                decision_id=decision_id,
                decision_type=decision_type,
                action=selected_action,
                confidence=confidence,
                reasoning=reasoning,
                state_snapshot=self.current_state,
                expected_impact=expected_impact
            )
            
            # Record decision
            self.decision_history.append(decision)
            if selected_action != ActionType.NO_ACTION:
                self.pending_decisions[decision_id] = decision
            
            # Keep history manageable
            if len(self.decision_history) > 500:
                self.decision_history = self.decision_history[-250:]
            
            logger.info(f"Made decision {decision_id}: {selected_action.value} (confidence: {confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return None
    
    def record_outcome(
        self,
        decision_id: str,
        success: bool,
        actual_impact: Dict[str, float],
        custom_reward: Optional[float] = None
    ):
        """Record the outcome of a decision for learning."""
        if decision_id not in self.pending_decisions:
            logger.warning(f"Unknown decision ID: {decision_id}")
            return
        
        decision = self.pending_decisions[decision_id]
        
        # Calculate reward
        if custom_reward is not None:
            reward = custom_reward
        else:
            reward = self._calculate_reward(decision, actual_impact, success)
        
        # Create outcome
        outcome = DecisionOutcome(
            decision_id=decision_id,
            success=success,
            actual_impact=actual_impact,
            reward=reward
        )
        
        # Record outcome
        self.outcome_history.append(outcome)
        
        # Update Q-learning
        self._update_q_values(decision, outcome)
        
        # Remove from pending
        del self.pending_decisions[decision_id]
        
        # Keep outcome history manageable
        if len(self.outcome_history) > 500:
            self.outcome_history = self.outcome_history[-250:]
        
        logger.info(f"Recorded outcome for {decision_id}: success={success}, reward={reward:.3f}")
    
    def _calculate_reward(
        self,
        decision: Decision,
        actual_impact: Dict[str, float],
        success: bool
    ) -> float:
        """Calculate reward for reinforcement learning."""
        if not success:
            return -1.0  # Negative reward for failed actions
        
        reward = 0.0
        
        # Reward based on positive impacts
        for metric, actual_change in actual_impact.items():
            weight = self.reward_weights.get(f"{metric}_improvement", 0.0)
            
            # Positive change for metrics where lower is better (cpu, memory, latency, errors)
            if metric in ["cpu_usage", "memory_usage", "network_latency", "error_rate", "response_time"]:
                if actual_change < 0:  # Improvement (reduction)
                    reward += abs(actual_change) * weight / 100.0
                else:  # Degradation
                    reward -= actual_change * weight / 100.0
            else:
                # For metrics where higher is better
                reward += actual_change * weight / 100.0
        
        # Stability bonus if system remains stable
        if all(abs(change) < 10 for change in actual_impact.values()):
            reward += self.reward_weights.get("stability_bonus", 0.0)
        
        # Small penalty for taking action (encourage efficiency)
        if decision.action != ActionType.NO_ACTION:
            reward += self.reward_weights.get("action_penalty", 0.0)
        
        return reward
    
    def _update_q_values(self, decision: Decision, outcome: DecisionOutcome):
        """Update Q-values using temporal difference learning."""
        state_key = self._state_to_key(decision.state_snapshot)
        action_key = decision.action.value
        
        # Initialize if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
            self.state_action_counts[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
            self.state_action_counts[state_key][action_key] = 0
        
        # Current Q-value
        current_q = self.q_table[state_key][action_key]
        
        # Future reward (simplified - assume current state is next state)
        next_state_key = self._state_to_key(self.current_state)
        max_future_q = 0.0
        
        if next_state_key in self.q_table and self.q_table[next_state_key]:
            max_future_q = max(self.q_table[next_state_key].values())
        
        # Temporal difference update
        td_target = outcome.reward + self.discount_factor * max_future_q
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state_key][action_key] += self.learning_rate * td_error
        self.state_action_counts[state_key][action_key] += 1
    
    async def _decision_loop(self):
        """Main decision-making loop."""
        logger.info("Autonomous decision engine started")
        
        while self.is_running:
            try:
                # Make decisions for different types
                decision_types = [
                    DecisionType.RESOURCE_ALLOCATION,
                    DecisionType.LOAD_BALANCING,
                    DecisionType.SCALING,
                    DecisionType.OPTIMIZATION
                ]
                
                for decision_type in decision_types:
                    decision = await self.make_decision(decision_type)
                    if decision and decision.action != ActionType.NO_ACTION:
                        # Log the decision
                        logger.info(f"Decision made: {decision.action.value} - {decision.reasoning}")
                        break  # One decision per cycle
                
                await asyncio.sleep(self.decision_interval)
                
            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(self.decision_interval)
    
    async def start(self):
        """Start the autonomous decision engine."""
        if self.is_running:
            logger.warning("Decision engine is already running")
            return
        
        self.is_running = True
        self._decision_task = asyncio.create_task(self._decision_loop())
        logger.info("Autonomous decision engine started")
    
    async def stop(self):
        """Stop the autonomous decision engine."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._decision_task:
            self._decision_task.cancel()
            try:
                await self._decision_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous decision engine stopped")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get decision engine summary and statistics."""
        total_decisions = len(self.decision_history)
        successful_outcomes = sum(1 for outcome in self.outcome_history if outcome.success)
        total_outcomes = len(self.outcome_history)
        
        action_counts = {}
        for decision in self.decision_history:
            action = decision.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        avg_confidence = statistics.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0
        avg_reward = statistics.mean([o.reward for o in self.outcome_history]) if self.outcome_history else 0.0
        
        return {
            "total_decisions": total_decisions,
            "pending_decisions": len(self.pending_decisions),
            "success_rate": successful_outcomes / max(total_outcomes, 1),
            "average_confidence": avg_confidence,
            "average_reward": avg_reward,
            "action_distribution": action_counts,
            "q_table_size": len(self.q_table),
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "current_state": {
                "cpu_usage": self.current_state.cpu_usage,
                "memory_usage": self.current_state.memory_usage,
                "network_latency": self.current_state.network_latency,
                "error_rate": self.current_state.error_rate
            }
        }
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for analysis."""
        return {
            "q_table": self.q_table,
            "state_action_counts": self.state_action_counts,
            "decision_history": [
                {
                    "decision_id": d.decision_id,
                    "action": d.action.value,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp
                }
                for d in self.decision_history[-100:]  # Last 100 decisions
            ],
            "outcome_history": [
                {
                    "decision_id": o.decision_id,
                    "success": o.success,
                    "reward": o.reward,
                    "timestamp": o.timestamp
                }
                for o in self.outcome_history[-100:]  # Last 100 outcomes
            ],
            "reward_weights": self.reward_weights.copy()
        }