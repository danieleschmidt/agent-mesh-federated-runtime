"""Auto-scaling and dynamic resource management system.

This module provides intelligent auto-scaling capabilities including:
- Predictive scaling based on historical patterns
- Event-driven scaling for sudden load changes  
- Resource optimization and right-sizing
- Multi-metric scaling decisions
"""

import asyncio
import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from uuid import UUID, uuid4
from collections import deque

import structlog


class ScalingAction(Enum):
    """Types of scaling actions."""
    
    SCALE_OUT = "scale_out"  # Add more instances
    SCALE_IN = "scale_in"    # Remove instances  
    SCALE_UP = "scale_up"    # Increase resources per instance
    SCALE_DOWN = "scale_down" # Decrease resources per instance
    NO_ACTION = "no_action"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    NETWORK_THROUGHPUT = "network_throughput"
    REQUEST_RATE = "request_rate"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    
    rule_id: str
    trigger: ScalingTrigger
    metric_name: str
    threshold_high: float
    threshold_low: float
    scale_out_action: ScalingAction = ScalingAction.SCALE_OUT
    scale_in_action: ScalingAction = ScalingAction.SCALE_IN
    cooldown_seconds: float = 300.0  # 5 minutes default
    evaluation_periods: int = 2  # Number of periods threshold must be breached
    scaling_adjustment: int = 1  # Number of instances to add/remove
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    
    event_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    rule_id: str = ""
    trigger: ScalingTrigger = ScalingTrigger.CPU_UTILIZATION
    action: ScalingAction = ScalingAction.NO_ACTION
    metric_value: float = 0.0
    threshold_breached: float = 0.0
    instances_before: int = 0
    instances_after: int = 0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ResourceRequirements:
    """Resource requirements for scaling."""
    
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    storage_gb: float = 10.0
    network_mbps: float = 100.0
    gpu_count: int = 0


@dataclass 
class InstanceMetrics:
    """Metrics for a single instance."""
    
    instance_id: UUID
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_throughput_mbps: float = 0.0
    requests_per_second: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def is_stale(self, max_age_seconds: float = 60.0) -> bool:
        """Check if metrics are stale."""
        return time.time() - self.last_updated > max_age_seconds


class PredictiveScaler:
    """Predictive scaling based on historical patterns."""
    
    def __init__(self, history_hours: int = 24):
        self.history_hours = history_hours
        self.metric_history: Dict[str, deque] = {}
        self.patterns: Dict[str, List[float]] = {}
        self.logger = structlog.get_logger("predictive_scaler")
        
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record metric for pattern analysis."""
        if timestamp is None:
            timestamp = time.time()
            
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(maxlen=self.history_hours * 60)  # 1 per minute
            
        self.metric_history[metric_name].append((timestamp, value))
        
    def predict_metric(self, metric_name: str, minutes_ahead: int = 15) -> Optional[float]:
        """Predict metric value for future time."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) < 10:
            return None
            
        history = list(self.metric_history[metric_name])
        
        # Simple trend analysis (would be more sophisticated in production)
        if len(history) >= 5:
            recent_values = [v for _, v in history[-5:]]
            older_values = [v for _, v in history[-10:-5]] if len(history) >= 10 else recent_values
            
            recent_avg = sum(recent_values) / len(recent_values)
            older_avg = sum(older_values) / len(older_values)
            
            trend = (recent_avg - older_avg) / max(older_avg, 0.1)  # Avoid division by zero
            
            # Extrapolate trend
            predicted_value = recent_avg + (trend * recent_avg * (minutes_ahead / 15.0))
            
            # Bound the prediction to reasonable limits
            max_value = max(v for _, v in history) * 1.5
            min_value = min(v for _, v in history) * 0.5
            
            return max(min_value, min(max_value, predicted_value))
            
        return None
        
    def detect_patterns(self, metric_name: str) -> Dict[str, Any]:
        """Detect patterns in metric history."""
        if metric_name not in self.metric_history:
            return {}
            
        history = list(self.metric_history[metric_name])
        if len(history) < 60:  # Need at least 1 hour of data
            return {}
            
        values = [v for _, v in history]
        
        # Calculate basic statistics
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Detect spikes (values > 2 standard deviations)
        spike_threshold = avg + (2 * std_dev)
        spikes = [i for i, v in enumerate(values) if v > spike_threshold]
        
        # Detect periodicity (simplified)
        periodicity_detected = False
        if len(values) >= 120:  # 2 hours of data
            # Check for hourly patterns
            hourly_averages = []
            for hour in range(min(24, len(values) // 60)):
                hour_start = hour * 60
                hour_end = min((hour + 1) * 60, len(values))
                hour_values = values[hour_start:hour_end]
                if hour_values:
                    hourly_averages.append(sum(hour_values) / len(hour_values))
                    
            if len(hourly_averages) >= 2:
                hour_variance = sum((v - avg) ** 2 for v in hourly_averages) / len(hourly_averages)
                if hour_variance > variance * 0.1:  # Significant hourly variation
                    periodicity_detected = True
                    
        return {
            "average": avg,
            "std_deviation": std_dev,
            "spike_count": len(spikes),
            "periodicity_detected": periodicity_detected,
            "trend": "increasing" if values[-10:] > values[-20:-10] else "decreasing" if values[-10:] < values[-20:-10] else "stable"
        }


class AutoScaler:
    """Main auto-scaling engine."""
    
    def __init__(self, cluster_name: str):
        self.cluster_name = cluster_name
        self.logger = structlog.get_logger("auto_scaler", cluster=cluster_name)
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.instance_metrics: Dict[UUID, InstanceMetrics] = {}
        self.scaling_events: List[ScalingEvent] = []
        
        # Scaling constraints  
        self.min_instances = 1
        self.max_instances = 100
        self.current_instances = 1
        
        # Resource configuration
        self.base_resource_requirements = ResourceRequirements()
        
        # Components
        self.predictive_scaler = PredictiveScaler()
        
        # Scaling callbacks
        self.scale_out_callback: Optional[Callable[[int], bool]] = None
        self.scale_in_callback: Optional[Callable[[List[UUID]], bool]] = None
        
        # Background tasks
        self.evaluation_task: Optional[asyncio.Task] = None
        self.prediction_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Cooldown tracking
        self.last_scaling_action: Dict[str, float] = {}
        
    async def start(self):
        """Start auto-scaling engine."""
        self.logger.info("Starting auto-scaler")
        self.running = True
        
        # Start background evaluation
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        self.prediction_task = asyncio.create_task(self._prediction_loop())
        
        self.logger.info("Auto-scaler started")
        
    async def stop(self):
        """Stop auto-scaling engine."""
        self.logger.info("Stopping auto-scaler")
        self.running = False
        
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
                
        if self.prediction_task:
            self.prediction_task.cancel()
            try:
                await self.prediction_task
            except asyncio.CancelledError:
                pass
                
        self.logger.info("Auto-scaler stopped")
        
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a scaling rule."""
        self.scaling_rules[rule.rule_id] = rule
        self.logger.info("Scaling rule added", 
                        rule_id=rule.rule_id,
                        trigger=rule.trigger.value,
                        threshold_high=rule.threshold_high)
        
    def remove_scaling_rule(self, rule_id: str):
        """Remove a scaling rule."""
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            self.logger.info("Scaling rule removed", rule_id=rule_id)
            
    def update_instance_metrics(self, instance_id: UUID, metrics: InstanceMetrics):
        """Update metrics for an instance."""
        metrics.last_updated = time.time()
        self.instance_metrics[instance_id] = metrics
        
        # Record metrics for predictive scaling
        self.predictive_scaler.record_metric("cpu_utilization", metrics.cpu_utilization)
        self.predictive_scaler.record_metric("memory_utilization", metrics.memory_utilization)
        self.predictive_scaler.record_metric("requests_per_second", metrics.requests_per_second)
        
    def set_scaling_callbacks(
        self,
        scale_out_callback: Callable[[int], bool],
        scale_in_callback: Callable[[List[UUID]], bool]
    ):
        """Set callbacks for actual scaling operations."""
        self.scale_out_callback = scale_out_callback
        self.scale_in_callback = scale_in_callback
        
    def set_instance_limits(self, min_instances: int, max_instances: int):
        """Set minimum and maximum instance limits."""
        self.min_instances = max(1, min_instances)
        self.max_instances = max(self.min_instances, max_instances)
        self.logger.info("Instance limits updated", 
                        min=self.min_instances, max=self.max_instances)
        
    async def trigger_immediate_evaluation(self):
        """Trigger immediate scaling evaluation."""
        await self._evaluate_scaling_rules()
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        active_instances = [
            iid for iid, metrics in self.instance_metrics.items()
            if not metrics.is_stale()
        ]
        
        # Calculate cluster-wide averages
        if active_instances:
            avg_cpu = sum(self.instance_metrics[iid].cpu_utilization for iid in active_instances) / len(active_instances)
            avg_memory = sum(self.instance_metrics[iid].memory_utilization for iid in active_instances) / len(active_instances)
            total_rps = sum(self.instance_metrics[iid].requests_per_second for iid in active_instances)
        else:
            avg_cpu = avg_memory = total_rps = 0.0
            
        recent_events = [e for e in self.scaling_events if time.time() - e.timestamp < 3600]  # Last hour
        
        return {
            "cluster_name": self.cluster_name,
            "current_instances": len(active_instances),
            "target_instances": self.current_instances,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "average_cpu_utilization": avg_cpu,
            "average_memory_utilization": avg_memory,
            "total_requests_per_second": total_rps,
            "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
            "recent_scaling_events": len(recent_events),
            "last_scaling_action": max(self.last_scaling_action.values()) if self.last_scaling_action else 0
        }
        
    def get_predictive_insights(self) -> Dict[str, Any]:
        """Get predictive scaling insights."""
        cpu_prediction = self.predictive_scaler.predict_metric("cpu_utilization", 15)
        memory_prediction = self.predictive_scaler.predict_metric("memory_utilization", 15)
        rps_prediction = self.predictive_scaler.predict_metric("requests_per_second", 15)
        
        cpu_patterns = self.predictive_scaler.detect_patterns("cpu_utilization")
        
        return {
            "predictions": {
                "cpu_utilization_15min": cpu_prediction,
                "memory_utilization_15min": memory_prediction,
                "requests_per_second_15min": rps_prediction
            },
            "patterns": {
                "cpu": cpu_patterns
            },
            "recommendations": self._generate_scaling_recommendations()
        }
        
    async def _evaluation_loop(self):
        """Background loop for scaling rule evaluation."""
        while self.running:
            try:
                await self._evaluate_scaling_rules()
                await asyncio.sleep(30)  # Evaluate every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Scaling evaluation error", error=str(e))
                await asyncio.sleep(30)
                
    async def _prediction_loop(self):
        """Background loop for predictive scaling."""
        while self.running:
            try:
                await self._evaluate_predictive_scaling()
                await asyncio.sleep(300)  # Predict every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Predictive scaling error", error=str(e))
                await asyncio.sleep(300)
                
    async def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules and take action if needed."""
        current_time = time.time()
        
        # Clean up stale metrics
        stale_instances = [
            iid for iid, metrics in self.instance_metrics.items()
            if metrics.is_stale()
        ]
        
        for iid in stale_instances:
            del self.instance_metrics[iid]
            
        active_instances = list(self.instance_metrics.keys())
        if not active_instances:
            return
            
        # Calculate cluster metrics
        cluster_metrics = self._calculate_cluster_metrics(active_instances)
        
        # Evaluate each rule
        for rule in self.scaling_rules.values():
            if not rule.enabled:
                continue
                
            # Check cooldown
            if rule.rule_id in self.last_scaling_action:
                time_since_last = current_time - self.last_scaling_action[rule.rule_id]
                if time_since_last < rule.cooldown_seconds:
                    continue
                    
            await self._evaluate_rule(rule, cluster_metrics)
            
    async def _evaluate_rule(self, rule: ScalingRule, cluster_metrics: Dict[str, float]):
        """Evaluate a single scaling rule."""
        metric_value = cluster_metrics.get(rule.metric_name, 0.0)
        
        action = ScalingAction.NO_ACTION
        threshold_breached = 0.0
        
        # Determine scaling action
        if metric_value > rule.threshold_high:
            action = rule.scale_out_action
            threshold_breached = rule.threshold_high
        elif metric_value < rule.threshold_low:
            action = rule.scale_in_action  
            threshold_breached = rule.threshold_low
            
        if action != ScalingAction.NO_ACTION:
            success = await self._execute_scaling_action(action, rule.scaling_adjustment)
            
            # Record scaling event
            event = ScalingEvent(
                rule_id=rule.rule_id,
                trigger=rule.trigger,
                action=action,
                metric_value=metric_value,
                threshold_breached=threshold_breached,
                instances_before=len(self.instance_metrics),
                instances_after=self.current_instances,
                success=success
            )
            
            self.scaling_events.append(event)
            
            # Update cooldown
            if success:
                self.last_scaling_action[rule.rule_id] = time.time()
                
            self.logger.info("Scaling action executed",
                           rule_id=rule.rule_id,
                           action=action.value,
                           metric_value=metric_value,
                           threshold=threshold_breached,
                           success=success)
                           
    async def _evaluate_predictive_scaling(self):
        """Evaluate predictive scaling opportunities."""
        # Get predictions
        cpu_prediction = self.predictive_scaler.predict_metric("cpu_utilization", 15)
        memory_prediction = self.predictive_scaler.predict_metric("memory_utilization", 15)
        
        if cpu_prediction and cpu_prediction > 80:
            # Predicted high CPU load - consider preemptive scaling
            self.logger.info("Predictive scaling: High CPU predicted",
                           predicted_cpu=cpu_prediction)
            
            # Create temporary rule for predictive scaling
            temp_rule = ScalingRule(
                rule_id="predictive_cpu",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                metric_name="cpu_utilization",
                threshold_high=75,  # Lower threshold for predictive
                threshold_low=0,
                cooldown_seconds=600  # Longer cooldown for predictive
            )
            
            cluster_metrics = {"cpu_utilization": cpu_prediction}
            await self._evaluate_rule(temp_rule, cluster_metrics)
            
    async def _execute_scaling_action(self, action: ScalingAction, adjustment: int) -> bool:
        """Execute the actual scaling action."""
        if action == ScalingAction.SCALE_OUT:
            if self.current_instances + adjustment <= self.max_instances:
                if self.scale_out_callback:
                    success = await self.scale_out_callback(adjustment)
                    if success:
                        self.current_instances += adjustment
                    return success
                    
        elif action == ScalingAction.SCALE_IN:
            if self.current_instances - adjustment >= self.min_instances:
                # Select instances to remove (least utilized)
                instances_to_remove = self._select_instances_for_removal(adjustment)
                
                if self.scale_in_callback and instances_to_remove:
                    success = await self.scale_in_callback(instances_to_remove)
                    if success:
                        self.current_instances -= len(instances_to_remove)
                        # Remove metrics for scaled-in instances
                        for iid in instances_to_remove:
                            self.instance_metrics.pop(iid, None)
                    return success
                    
        return False
        
    def _calculate_cluster_metrics(self, instance_ids: List[UUID]) -> Dict[str, float]:
        """Calculate cluster-wide metrics from instance metrics."""
        if not instance_ids:
            return {}
            
        metrics = [self.instance_metrics[iid] for iid in instance_ids]
        
        return {
            "cpu_utilization": sum(m.cpu_utilization for m in metrics) / len(metrics),
            "memory_utilization": sum(m.memory_utilization for m in metrics) / len(metrics),
            "network_throughput": sum(m.network_throughput_mbps for m in metrics),
            "request_rate": sum(m.requests_per_second for m in metrics),
            "response_time": sum(m.response_time_ms for m in metrics) / len(metrics),
            "error_rate": sum(m.error_rate for m in metrics) / len(metrics),
            "queue_length": sum(m.queue_length for m in metrics)
        }
        
    def _select_instances_for_removal(self, count: int) -> List[UUID]:
        """Select instances to remove during scale-in."""
        # Sort instances by utilization (remove least utilized first)
        sorted_instances = sorted(
            self.instance_metrics.items(),
            key=lambda x: (x[1].cpu_utilization + x[1].memory_utilization) / 2
        )
        
        return [iid for iid, _ in sorted_instances[:count]]
        
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate scaling recommendations based on current state."""
        recommendations = []
        
        if not self.instance_metrics:
            return ["No instance metrics available"]
            
        cluster_metrics = self._calculate_cluster_metrics(list(self.instance_metrics.keys()))
        
        # CPU recommendations
        avg_cpu = cluster_metrics.get("cpu_utilization", 0)
        if avg_cpu > 80:
            recommendations.append("Consider scaling out due to high CPU utilization")
        elif avg_cpu < 20 and self.current_instances > self.min_instances:
            recommendations.append("Consider scaling in due to low CPU utilization")
            
        # Memory recommendations  
        avg_memory = cluster_metrics.get("memory_utilization", 0)
        if avg_memory > 85:
            recommendations.append("Consider scaling out due to high memory utilization")
            
        # Response time recommendations
        avg_response_time = cluster_metrics.get("response_time", 0)
        if avg_response_time > 1000:  # 1 second
            recommendations.append("Consider scaling out due to high response times")
            
        return recommendations