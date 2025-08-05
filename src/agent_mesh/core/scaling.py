"""Auto-scaling and load balancing system for Agent Mesh.

This module provides intelligent auto-scaling, load balancing,
and adaptive resource management for optimal performance.
"""

import asyncio
import time
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from uuid import UUID, uuid4

import structlog

from .monitoring import PerformanceMetrics
from .error_handling import AgentMeshError


class ScalingDirection(Enum):
    """Scaling directions."""
    
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASHING = "consistent_hashing"
    ADAPTIVE = "adaptive"


class ScalingTrigger(Enum):
    """Scaling trigger types."""
    
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingPolicy:
    """Auto-scaling policy definition."""
    
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = 1
    cooldown_period: float = 300.0  # 5 minutes
    min_instances: int = 1
    max_instances: int = 10
    evaluation_period: float = 60.0  # 1 minute
    datapoints_to_alarm: int = 2
    is_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancerTarget:
    """Load balancer target definition."""
    
    target_id: UUID
    address: str
    port: int
    weight: float = 1.0
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    is_healthy: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
    
    @property
    def load_score(self) -> float:
        """Calculate load score (lower is better)."""
        connection_factor = self.current_connections / max(1, self.weight)
        response_time_factor = self.avg_response_time / 1000.0  # Convert to seconds
        failure_penalty = (1.0 - self.success_rate) * 10
        
        return connection_factor + response_time_factor + failure_penalty


@dataclass
class ScalingEvent:
    """Scaling event record."""
    
    event_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    policy_name: str = ""
    trigger: ScalingTrigger = ScalingTrigger.CPU_UTILIZATION
    direction: ScalingDirection = ScalingDirection.STABLE
    old_capacity: int = 0
    new_capacity: int = 0
    trigger_value: float = 0.0
    threshold: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Intelligent load balancer with multiple strategies."""
    
    def __init__(
        self, 
        name: str,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        self.name = name
        self.strategy = strategy
        self.logger = structlog.get_logger("load_balancer", name=name)
        
        # Target management
        self.targets: Dict[UUID, LoadBalancerTarget] = {}
        self.healthy_targets: List[UUID] = []
        
        # Load balancing state
        self.round_robin_index = 0
        self.request_count = 0
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.response_times: deque = deque(maxlen=1000)
        
        # Health checking
        self.health_check_interval = 30.0
        self.health_check_timeout = 10.0
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Strategy optimization
        self.strategy_performance: Dict[LoadBalancingStrategy, float] = {}
        self.last_strategy_evaluation = 0.0
    
    async def start(self) -> None:
        """Start load balancer."""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self.logger.info("Load balancer started", strategy=self.strategy.value)
    
    async def stop(self) -> None:
        """Stop load balancer."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Load balancer stopped")
    
    def add_target(self, target: LoadBalancerTarget) -> None:
        """Add target to load balancer."""
        self.targets[target.target_id] = target
        if target.is_healthy:
            self.healthy_targets.append(target.target_id)
        
        self.logger.info("Target added", 
                        target_id=str(target.target_id),
                        address=target.address,
                        weight=target.weight)
    
    def remove_target(self, target_id: UUID) -> bool:
        """Remove target from load balancer."""
        if target_id not in self.targets:
            return False
        
        del self.targets[target_id]
        if target_id in self.healthy_targets:
            self.healthy_targets.remove(target_id)
        
        self.logger.info("Target removed", target_id=str(target_id))
        return True
    
    def select_target(self, request_context: Optional[Dict[str, Any]] = None) -> Optional[LoadBalancerTarget]:
        """Select target based on load balancing strategy."""
        if not self.healthy_targets:
            return None
        
        if self.strategy == LoadBalancingStrategy.ADAPTIVE:
            strategy = self._choose_optimal_strategy()
        else:
            strategy = self.strategy
        
        target_id = None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            target_id = self._round_robin_select()
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            target_id = self._least_connections_select()
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            target_id = self._weighted_round_robin_select()
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            target_id = self._least_response_time_select()
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASHING:
            target_id = self._consistent_hash_select(request_context)
        else:
            target_id = self._round_robin_select()  # Fallback
        
        if target_id and target_id in self.targets:
            target = self.targets[target_id]
            target.current_connections += 1
            return target
        
        return None
    
    def record_request_completion(
        self, 
        target_id: UUID, 
        response_time_ms: float, 
        success: bool
    ) -> None:
        """Record completion of request to target."""
        if target_id not in self.targets:
            return
        
        target = self.targets[target_id]
        target.current_connections = max(0, target.current_connections - 1)
        target.total_requests += 1
        
        if not success:
            target.failed_requests += 1
        
        # Update response time (exponential moving average)
        alpha = 0.1
        target.avg_response_time = (
            alpha * response_time_ms + 
            (1 - alpha) * target.avg_response_time
        )
        
        # Update global statistics
        self.total_requests += 1
        if not success:
            self.total_errors += 1
        
        self.response_times.append(response_time_ms)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_count = len(self.healthy_targets)
        total_count = len(self.targets)
        
        avg_response_time = 0.0
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
        
        error_rate = 0.0
        if self.total_requests > 0:
            error_rate = self.total_errors / self.total_requests
        
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "total_targets": total_count,
            "healthy_targets": healthy_count,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "avg_response_time_ms": avg_response_time,
            "target_details": [
                {
                    "target_id": str(target_id),
                    "address": target.address,
                    "is_healthy": target.is_healthy,
                    "current_connections": target.current_connections,
                    "total_requests": target.total_requests,
                    "success_rate": target.success_rate,
                    "avg_response_time": target.avg_response_time
                }
                for target_id, target in self.targets.items()
            ]
        }
    
    # Private methods
    
    def _round_robin_select(self) -> Optional[UUID]:
        """Round-robin target selection."""
        if not self.healthy_targets:
            return None
        
        target_id = self.healthy_targets[self.round_robin_index % len(self.healthy_targets)]
        self.round_robin_index += 1
        return target_id
    
    def _least_connections_select(self) -> Optional[UUID]:
        """Select target with least active connections."""
        if not self.healthy_targets:
            return None
        
        min_connections = float('inf')
        selected_target = None
        
        for target_id in self.healthy_targets:
            target = self.targets[target_id]
            weighted_connections = target.current_connections / max(0.1, target.weight)
            
            if weighted_connections < min_connections:
                min_connections = weighted_connections
                selected_target = target_id
        
        return selected_target
    
    def _weighted_round_robin_select(self) -> Optional[UUID]:
        """Weighted round-robin selection."""
        if not self.healthy_targets:
            return None
        
        # Calculate total weight
        total_weight = sum(self.targets[tid].weight for tid in self.healthy_targets)
        if total_weight == 0:
            return self._round_robin_select()
        
        # Select based on weight distribution
        import random
        rand_weight = random.random() * total_weight
        cumulative_weight = 0.0
        
        for target_id in self.healthy_targets:
            cumulative_weight += self.targets[target_id].weight
            if cumulative_weight >= rand_weight:
                return target_id
        
        return self.healthy_targets[0]  # Fallback
    
    def _least_response_time_select(self) -> Optional[UUID]:
        """Select target with lowest response time."""
        if not self.healthy_targets:
            return None
        
        min_response_time = float('inf')
        selected_target = None
        
        for target_id in self.healthy_targets:
            target = self.targets[target_id]
            
            if target.avg_response_time < min_response_time:
                min_response_time = target.avg_response_time
                selected_target = target_id
        
        return selected_target
    
    def _consistent_hash_select(self, request_context: Optional[Dict[str, Any]]) -> Optional[UUID]:
        """Consistent hashing selection."""
        if not self.healthy_targets:
            return None
        
        # Use request context to generate hash key
        hash_key = "default"
        if request_context:
            # Create hash from context
            context_str = str(sorted(request_context.items()))
            hash_key = str(hash(context_str))
        
        # Simple consistent hashing
        hash_value = hash(hash_key) % len(self.healthy_targets)
        return self.healthy_targets[hash_value]
    
    def _choose_optimal_strategy(self) -> LoadBalancingStrategy:
        """Choose optimal strategy based on performance."""
        current_time = time.time()
        
        # Evaluate strategies periodically
        if current_time - self.last_strategy_evaluation < 300:  # 5 minutes
            # Use current strategy or default
            return getattr(self, '_current_optimal_strategy', LoadBalancingStrategy.LEAST_CONNECTIONS)
        
        self.last_strategy_evaluation = current_time
        
        # Analyze current performance
        if not self.response_times:
            return LoadBalancingStrategy.LEAST_CONNECTIONS
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        error_rate = self.total_errors / max(1, self.total_requests)
        
        # Choose strategy based on conditions
        if error_rate > 0.05:  # High error rate
            strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        elif avg_response_time > 1000:  # High response time
            strategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME
        elif len(self.healthy_targets) <= 3:  # Few targets
            strategy = LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
        else:
            strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
        
        self._current_optimal_strategy = strategy
        return strategy
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all targets."""
        current_time = time.time()
        
        for target_id, target in self.targets.items():
            try:
                # Simple health check simulation
                # In production, this would be actual HTTP/TCP health checks
                is_healthy = await self._check_target_health(target)
                
                target.last_health_check = current_time
                
                if is_healthy != target.is_healthy:
                    target.is_healthy = is_healthy
                    
                    if is_healthy:
                        if target_id not in self.healthy_targets:
                            self.healthy_targets.append(target_id)
                        self.logger.info("Target became healthy", 
                                       target_id=str(target_id))
                    else:
                        if target_id in self.healthy_targets:
                            self.healthy_targets.remove(target_id)
                        self.logger.warning("Target became unhealthy", 
                                          target_id=str(target_id))
                
            except Exception as e:
                self.logger.error("Health check failed", 
                                target_id=str(target_id), error=str(e))
                
                # Mark as unhealthy on check failure
                if target.is_healthy:
                    target.is_healthy = False
                    if target_id in self.healthy_targets:
                        self.healthy_targets.remove(target_id)
    
    async def _check_target_health(self, target: LoadBalancerTarget) -> bool:
        """Check if target is healthy."""
        try:
            # Simulate health check - in production would be actual network check
            await asyncio.sleep(0.01)  # Simulate network delay
            
            # Consider target unhealthy if error rate is too high
            if target.success_rate < 0.5:
                return False
            
            # Consider target unhealthy if response time is too high
            if target.avg_response_time > 5000:  # 5 seconds
                return False
            
            return True
            
        except Exception:
            return False


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger("auto_scaler", name=name)
        
        # Scaling policies
        self.policies: Dict[str, ScalingPolicy] = {}
        
        # Current capacity
        self.current_capacity = 1
        self.min_capacity = 1
        self.max_capacity = 10
        
        # Scaling state
        self.scaling_in_progress = False
        self.last_scaling_action: Optional[float] = None
        
        # Metrics history
        self.metrics_history: Dict[ScalingTrigger, deque] = {
            trigger: deque(maxlen=100) for trigger in ScalingTrigger
        }
        
        # Scaling events
        self.scaling_events: List[ScalingEvent] = []
        
        # Custom metric handlers
        self.metric_handlers: Dict[str, Callable[[], float]] = {}
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_enabled = False
    
    async def start(self) -> None:
        """Start auto-scaler monitoring."""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Auto-scaler started", current_capacity=self.current_capacity)
    
    async def stop(self) -> None:
        """Stop auto-scaler monitoring."""
        self._monitoring_enabled = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Auto-scaler stopped")
    
    def add_policy(self, policy: ScalingPolicy) -> None:
        """Add scaling policy."""
        self.policies[policy.name] = policy
        
        self.logger.info("Scaling policy added", 
                        name=policy.name,
                        trigger=policy.trigger.value,
                        scale_up_threshold=policy.scale_up_threshold,
                        scale_down_threshold=policy.scale_down_threshold)
    
    def register_metric_handler(self, metric_name: str, handler: Callable[[], float]) -> None:
        """Register custom metric handler."""
        self.metric_handlers[metric_name] = handler
        self.logger.info("Custom metric handler registered", metric_name=metric_name)
    
    async def get_current_metric_value(self, trigger: ScalingTrigger, metric_name: Optional[str] = None) -> float:
        """Get current value for scaling trigger."""
        try:
            if trigger == ScalingTrigger.CPU_UTILIZATION:
                # Simulate CPU usage - in production would get from system
                import psutil
                return psutil.cpu_percent(interval=1)
            
            elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
                import psutil
                return psutil.virtual_memory().percent
            
            elif trigger == ScalingTrigger.REQUEST_RATE:
                # Would get from application metrics
                return len(self.metrics_history[ScalingTrigger.REQUEST_RATE])
            
            elif trigger == ScalingTrigger.RESPONSE_TIME:
                # Would get from application metrics
                recent_times = list(self.metrics_history[ScalingTrigger.RESPONSE_TIME])
                if recent_times:
                    return sum(recent_times) / len(recent_times)
                return 0.0
            
            elif trigger == ScalingTrigger.ERROR_RATE:
                # Would calculate from application metrics
                return 0.0
            
            elif trigger == ScalingTrigger.CUSTOM_METRIC and metric_name:
                if metric_name in self.metric_handlers:
                    return await asyncio.to_thread(self.metric_handlers[metric_name])
                return 0.0
            
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error("Failed to get metric value", 
                            trigger=trigger.value, error=str(e))
            return 0.0
    
    async def evaluate_scaling_policies(self) -> Optional[ScalingDirection]:
        """Evaluate all scaling policies and determine scaling direction."""
        scaling_decisions = []
        
        for policy_name, policy in self.policies.items():
            if not policy.is_enabled:
                continue
            
            # Check cooldown period
            if (self.last_scaling_action and 
                time.time() - self.last_scaling_action < policy.cooldown_period):
                continue
            
            # Get current metric value
            current_value = await self.get_current_metric_value(
                policy.trigger, 
                policy.metadata.get('metric_name')
            )
            
            # Store metric value
            self.metrics_history[policy.trigger].append(current_value)
            
            # Evaluate thresholds
            recent_values = list(self.metrics_history[policy.trigger])[-policy.datapoints_to_alarm:]
            
            if len(recent_values) >= policy.datapoints_to_alarm:
                avg_value = sum(recent_values) / len(recent_values)
                
                if avg_value > policy.scale_up_threshold:
                    if self.current_capacity < policy.max_instances:
                        scaling_decisions.append((
                            ScalingDirection.UP, 
                            policy, 
                            avg_value,
                            f"Average {policy.trigger.value} ({avg_value:.2f}) > threshold ({policy.scale_up_threshold})"
                        ))
                
                elif avg_value < policy.scale_down_threshold:
                    if self.current_capacity > policy.min_instances:
                        scaling_decisions.append((
                            ScalingDirection.DOWN, 
                            policy, 
                            avg_value,
                            f"Average {policy.trigger.value} ({avg_value:.2f}) < threshold ({policy.scale_down_threshold})"
                        ))
        
        # Prioritize scaling decisions (scale up has priority)
        if scaling_decisions:
            # Sort by priority: scale up first, then by threshold distance
            scaling_decisions.sort(key=lambda x: (
                x[0] != ScalingDirection.UP,  # Scale up first
                abs(x[2] - (x[1].scale_up_threshold if x[0] == ScalingDirection.UP else x[1].scale_down_threshold))
            ))
            
            direction, policy, value, reason = scaling_decisions[0]
            
            # Execute scaling action
            await self._execute_scaling_action(direction, policy, value, reason)
            
            return direction
        
        return ScalingDirection.STABLE
    
    async def manual_scale(self, target_capacity: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to target capacity."""
        if target_capacity < self.min_capacity or target_capacity > self.max_capacity:
            self.logger.warning("Target capacity out of bounds", 
                              target=target_capacity,
                              min_capacity=self.min_capacity,
                              max_capacity=self.max_capacity)
            return False
        
        if target_capacity == self.current_capacity:
            return True
        
        direction = ScalingDirection.UP if target_capacity > self.current_capacity else ScalingDirection.DOWN
        
        # Create scaling event
        event = ScalingEvent(
            policy_name="manual",
            trigger=ScalingTrigger.CUSTOM_METRIC,
            direction=direction,
            old_capacity=self.current_capacity,
            new_capacity=target_capacity,
            reason=reason
        )
        
        # Execute scaling
        success = await self._scale_to_capacity(target_capacity)
        
        if success:
            self.scaling_events.append(event)
            self.last_scaling_action = time.time()
            
            self.logger.info("Manual scaling completed", 
                           old_capacity=self.current_capacity,
                           new_capacity=target_capacity,
                           reason=reason)
        
        return success
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        recent_events = [e for e in self.scaling_events if time.time() - e.timestamp < 3600]
        scale_up_events = [e for e in recent_events if e.direction == ScalingDirection.UP]
        scale_down_events = [e for e in recent_events if e.direction == ScalingDirection.DOWN]
        
        return {
            "name": self.name,
            "current_capacity": self.current_capacity,
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "scaling_in_progress": self.scaling_in_progress,
            "active_policies": len([p for p in self.policies.values() if p.is_enabled]),
            "total_policies": len(self.policies),
            "recent_scaling_events": len(recent_events),
            "recent_scale_up_events": len(scale_up_events),
            "recent_scale_down_events": len(scale_down_events),
            "last_scaling_action": self.last_scaling_action,
            "current_metrics": {
                trigger.value: list(history)[-1] if history else 0.0
                for trigger, history in self.metrics_history.items()
            }
        }
    
    # Private methods
    
    async def _monitoring_loop(self) -> None:
        """Main auto-scaling monitoring loop."""
        while self._monitoring_enabled:
            try:
                if not self.scaling_in_progress:
                    await self.evaluate_scaling_policies()
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Auto-scaling monitoring error", error=str(e))
                await asyncio.sleep(30)
    
    async def _execute_scaling_action(
        self, 
        direction: ScalingDirection, 
        policy: ScalingPolicy, 
        trigger_value: float,
        reason: str
    ) -> None:
        """Execute scaling action based on policy."""
        if self.scaling_in_progress:
            return
        
        old_capacity = self.current_capacity
        
        if direction == ScalingDirection.UP:
            new_capacity = min(
                self.current_capacity + policy.scale_up_adjustment,
                policy.max_instances,
                self.max_capacity
            )
        else:
            new_capacity = max(
                self.current_capacity - policy.scale_down_adjustment,
                policy.min_instances,
                self.min_capacity
            )
        
        if new_capacity == old_capacity:
            return
        
        # Create scaling event
        event = ScalingEvent(
            policy_name=policy.name,
            trigger=policy.trigger,
            direction=direction,
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            trigger_value=trigger_value,
            threshold=policy.scale_up_threshold if direction == ScalingDirection.UP else policy.scale_down_threshold,
            reason=reason
        )
        
        # Execute scaling
        self.scaling_in_progress = True
        
        try:
            success = await self._scale_to_capacity(new_capacity)
            
            if success:
                self.scaling_events.append(event)
                self.last_scaling_action = time.time()
                
                self.logger.info("Auto-scaling action completed", 
                               policy=policy.name,
                               direction=direction.value,
                               old_capacity=old_capacity,
                               new_capacity=new_capacity,
                               trigger_value=trigger_value,
                               reason=reason)
            else:
                self.logger.error("Auto-scaling action failed", 
                                policy=policy.name,
                                direction=direction.value)
        
        finally:
            self.scaling_in_progress = False
    
    async def _scale_to_capacity(self, target_capacity: int) -> bool:
        """Scale to target capacity."""
        try:
            # Simulate scaling operation
            # In production, this would:
            # - Launch/terminate instances
            # - Update load balancer targets
            # - Wait for instances to be ready
            # - Verify scaling success
            
            scaling_time = abs(target_capacity - self.current_capacity) * 2.0  # 2 seconds per instance
            await asyncio.sleep(min(scaling_time, 30.0))  # Cap at 30 seconds
            
            self.current_capacity = target_capacity
            
            return True
            
        except Exception as e:
            self.logger.error("Scaling operation failed", 
                            target_capacity=target_capacity, error=str(e))
            return False


class ComprehensiveScalingSystem:
    """Comprehensive auto-scaling and load balancing system."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("scaling_system", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Components
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self.auto_scalers: Dict[str, AutoScaler] = {}
        
        # Global scaling coordination
        self.global_scaling_enabled = True
        self.resource_constraints = {
            'max_total_instances': 50,
            'max_cpu_percent': 80,
            'max_memory_percent': 85
        }
        
        # System state
        self.system_started = False
    
    async def start(self) -> None:
        """Start comprehensive scaling system."""
        if self.system_started:
            return
        
        # Start all load balancers
        for lb in self.load_balancers.values():
            await lb.start()
        
        # Start all auto-scalers
        for scaler in self.auto_scalers.values():
            await scaler.start()
        
        self.system_started = True
        self.logger.info("Comprehensive scaling system started")
    
    async def stop(self) -> None:
        """Stop comprehensive scaling system."""
        if not self.system_started:
            return
        
        # Stop all auto-scalers
        for scaler in self.auto_scalers.values():
            await scaler.stop()
        
        # Stop all load balancers
        for lb in self.load_balancers.values():
            await lb.stop()
        
        self.system_started = False
        self.logger.info("Comprehensive scaling system stopped")
    
    def create_load_balancer(
        self, 
        name: str, 
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ) -> LoadBalancer:
        """Create and register load balancer."""
        lb = LoadBalancer(name, strategy)
        self.load_balancers[name] = lb
        
        self.logger.info("Load balancer created", name=name, strategy=strategy.value)
        return lb
    
    def create_auto_scaler(self, name: str) -> AutoScaler:
        """Create and register auto-scaler."""
        scaler = AutoScaler(name)
        self.auto_scalers[name] = scaler
        
        self.logger.info("Auto-scaler created", name=name)
        return scaler
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        lb_stats = {
            name: lb.get_statistics() 
            for name, lb in self.load_balancers.items()
        }
        
        scaler_stats = {
            name: scaler.get_scaling_statistics() 
            for name, scaler in self.auto_scalers.items()
        }
        
        total_capacity = sum(scaler.current_capacity for scaler in self.auto_scalers.values())
        total_healthy_targets = sum(
            len(lb.healthy_targets) for lb in self.load_balancers.values()
        )
        
        return {
            "system_started": self.system_started,
            "global_scaling_enabled": self.global_scaling_enabled,
            "total_load_balancers": len(self.load_balancers),
            "total_auto_scalers": len(self.auto_scalers),
            "total_capacity": total_capacity,
            "total_healthy_targets": total_healthy_targets,
            "resource_constraints": self.resource_constraints,
            "load_balancer_stats": lb_stats,
            "auto_scaler_stats": scaler_stats
        }


# Global scaling system instance
_global_scaling_system: Optional[ComprehensiveScalingSystem] = None


def get_scaling_system(node_id: Optional[UUID] = None) -> ComprehensiveScalingSystem:
    """Get global scaling system instance."""
    global _global_scaling_system
    if _global_scaling_system is None:
        _global_scaling_system = ComprehensiveScalingSystem(node_id)
    return _global_scaling_system


async def start_scaling_system(node_id: Optional[UUID] = None) -> None:
    """Start global scaling system."""
    system = get_scaling_system(node_id)
    await system.start()


async def stop_scaling_system() -> None:
    """Stop global scaling system."""
    if _global_scaling_system:
        await _global_scaling_system.stop()