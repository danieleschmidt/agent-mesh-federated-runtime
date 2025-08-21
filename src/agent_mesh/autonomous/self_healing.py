"""Self-Healing Manager for autonomous system recovery."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

@dataclass
class RecoveryAction:
    """Recovery action definition."""
    name: str
    priority: int  # Lower = higher priority
    action_func: Callable
    conditions: List[str]
    cooldown_seconds: float = 60.0
    max_retries: int = 3
    last_executed: float = 0.0
    retry_count: int = 0

class SelfHealingManager:
    """Autonomous self-healing and recovery management."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_metrics: Dict[str, HealthMetric] = {}
        self.recovery_actions: List[RecoveryAction] = []
        self.recovery_history: List[Dict[str, Any]] = []
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Register default recovery actions
        self._register_default_actions()
        
    def _register_default_actions(self):
        """Register default recovery actions."""
        
        async def restart_failing_components():
            """Restart components with critical health status."""
            logger.info("Executing component restart recovery")
            # Implementation would restart specific components
            return {"action": "restart_components", "status": "success"}
            
        async def scale_resources():
            """Scale up resources when under pressure."""
            logger.info("Executing resource scaling recovery")
            # Implementation would scale up CPU/memory
            return {"action": "scale_resources", "status": "success"}
            
        async def network_partition_recovery():
            """Recover from network partitions."""
            logger.info("Executing network partition recovery")
            # Implementation would reconnect to network
            return {"action": "network_recovery", "status": "success"}
            
        async def memory_cleanup():
            """Clean up memory and caches."""
            logger.info("Executing memory cleanup recovery")
            # Implementation would clear caches, GC
            return {"action": "memory_cleanup", "status": "success"}
            
        # Register recovery actions with priorities
        self.register_recovery_action(
            "restart_failing_components",
            restart_failing_components,
            priority=1,
            conditions=["component_health_critical"],
            cooldown_seconds=300.0
        )
        
        self.register_recovery_action(
            "scale_resources", 
            scale_resources,
            priority=2,
            conditions=["cpu_usage_high", "memory_usage_high"],
            cooldown_seconds=120.0
        )
        
        self.register_recovery_action(
            "network_partition_recovery",
            network_partition_recovery, 
            priority=1,
            conditions=["network_partitioned"],
            cooldown_seconds=60.0
        )
        
        self.register_recovery_action(
            "memory_cleanup",
            memory_cleanup,
            priority=3,
            conditions=["memory_usage_critical"],
            cooldown_seconds=180.0
        )
    
    def register_health_metric(
        self,
        name: str,
        threshold_warning: float,
        threshold_critical: float
    ):
        """Register a health metric to monitor."""
        self.health_metrics[name] = HealthMetric(
            name=name,
            value=0.0,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical
        )
        logger.info(f"Registered health metric: {name}")
    
    def update_health_metric(self, name: str, value: float):
        """Update a health metric value."""
        if name in self.health_metrics:
            self.health_metrics[name].value = value
            self.health_metrics[name].timestamp = time.time()
            
            # Log status changes
            status = self.health_metrics[name].status
            if status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                logger.warning(f"Health metric {name} is {status.value}: {value}")
    
    def register_recovery_action(
        self,
        name: str,
        action_func: Callable,
        priority: int = 5,
        conditions: List[str] = None,
        cooldown_seconds: float = 60.0,
        max_retries: int = 3
    ):
        """Register a recovery action."""
        action = RecoveryAction(
            name=name,
            priority=priority,
            action_func=action_func,
            conditions=conditions or [],
            cooldown_seconds=cooldown_seconds,
            max_retries=max_retries
        )
        self.recovery_actions.append(action)
        self.recovery_actions.sort(key=lambda x: x.priority)
        logger.info(f"Registered recovery action: {name} (priority {priority})")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_metrics:
            return {"status": HealthStatus.HEALTHY.value, "metrics": {}}
            
        metric_statuses = [metric.status for metric in self.health_metrics.values()]
        
        # Determine overall status
        if HealthStatus.FAILED in metric_statuses:
            overall_status = HealthStatus.FAILED
        elif HealthStatus.CRITICAL in metric_statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in metric_statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
            
        return {
            "status": overall_status.value,
            "metrics": {
                name: {
                    "value": metric.value,
                    "status": metric.status.value,
                    "timestamp": metric.timestamp
                }
                for name, metric in self.health_metrics.items()
            },
            "last_check": time.time()
        }
    
    def _should_execute_action(self, action: RecoveryAction) -> bool:
        """Check if recovery action should be executed."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - action.last_executed < action.cooldown_seconds:
            return False
            
        # Check retry limit
        if action.retry_count >= action.max_retries:
            return False
            
        # Check conditions
        if not action.conditions:
            return True
            
        # Check if any condition is met
        for condition in action.conditions:
            if self._evaluate_condition(condition):
                return True
                
        return False
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a recovery condition."""
        # Map conditions to health metrics
        condition_mappings = {
            "component_health_critical": lambda: any(
                metric.status == HealthStatus.CRITICAL 
                for metric in self.health_metrics.values()
            ),
            "cpu_usage_high": lambda: self.health_metrics.get("cpu_usage", HealthMetric("", 0, 70, 90)).value > 70,
            "memory_usage_high": lambda: self.health_metrics.get("memory_usage", HealthMetric("", 0, 80, 95)).value > 80,
            "memory_usage_critical": lambda: self.health_metrics.get("memory_usage", HealthMetric("", 0, 80, 95)).value > 95,
            "network_partitioned": lambda: self.health_metrics.get("network_connectivity", HealthMetric("", 100, 50, 20)).value < 50,
        }
        
        evaluator = condition_mappings.get(condition)
        if evaluator:
            try:
                return evaluator()
            except Exception as e:
                logger.error(f"Error evaluating condition {condition}: {e}")
                return False
        
        return False
    
    async def _execute_recovery_action(self, action: RecoveryAction) -> bool:
        """Execute a recovery action."""
        try:
            logger.info(f"Executing recovery action: {action.name}")
            
            # Execute the action
            result = await action.action_func()
            
            # Update action state
            action.last_executed = time.time()
            action.retry_count += 1
            
            # Record in history
            history_entry = {
                "action": action.name,
                "timestamp": time.time(),
                "result": result,
                "retry_count": action.retry_count
            }
            self.recovery_history.append(history_entry)
            
            # Keep history size manageable
            if len(self.recovery_history) > 100:
                self.recovery_history = self.recovery_history[-50:]
                
            logger.info(f"Recovery action {action.name} completed: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Recovery action {action.name} failed: {e}")
            action.last_executed = time.time()
            action.retry_count += 1
            
            # Record failure in history
            history_entry = {
                "action": action.name,
                "timestamp": time.time(),
                "result": {"error": str(e)},
                "retry_count": action.retry_count
            }
            self.recovery_history.append(history_entry)
            
            return False
    
    async def _monitoring_loop(self):
        """Main monitoring and recovery loop."""
        logger.info("Self-healing monitoring started")
        
        while self.is_running:
            try:
                # Check system health
                health_status = self.get_system_health()
                
                # Execute recovery actions if needed
                if health_status["status"] in [HealthStatus.DEGRADED.value, HealthStatus.CRITICAL.value]:
                    for action in self.recovery_actions:
                        if self._should_execute_action(action):
                            await self._execute_recovery_action(action)
                            # Allow some time between actions
                            await asyncio.sleep(5.0)
                            break  # Execute one action per cycle
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in self-healing monitoring loop: {e}")
                await asyncio.sleep(10.0)  # Wait longer on error
    
    async def start(self):
        """Start the self-healing monitoring."""
        if self.is_running:
            logger.warning("Self-healing manager is already running")
            return
            
        self.is_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Self-healing manager started")
    
    async def stop(self):
        """Stop the self-healing monitoring."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Self-healing manager stopped")
    
    def get_recovery_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent recovery history."""
        return self.recovery_history[-limit:]
    
    def reset_action_retries(self, action_name: Optional[str] = None):
        """Reset retry counts for actions."""
        if action_name:
            for action in self.recovery_actions:
                if action.name == action_name:
                    action.retry_count = 0
                    logger.info(f"Reset retry count for action: {action_name}")
                    break
        else:
            for action in self.recovery_actions:
                action.retry_count = 0
            logger.info("Reset retry counts for all actions")