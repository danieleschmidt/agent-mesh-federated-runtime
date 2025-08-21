"""Comprehensive monitoring system with predictive analytics."""

import asyncio
import logging
import time
import statistics
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertState(Enum):
    """Alert states."""
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class Metric:
    """Metric definition and data."""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class MetricHistory:
    """Historical metric data."""
    name: str
    values: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, value)
    aggregations: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    name: str
    condition: str
    severity: AlertSeverity
    state: AlertState
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    triggered_at: Optional[float] = None
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None

@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    interval: float
    timeout: float
    last_check: Optional[float] = None
    last_result: Optional[bool] = None
    consecutive_failures: int = 0

class ComprehensiveMonitor:
    """Advanced monitoring system with predictive capabilities."""
    
    def __init__(
        self,
        collection_interval: float = 15.0,
        retention_period: float = 86400.0,  # 24 hours
        alert_evaluation_interval: float = 30.0
    ):
        self.collection_interval = collection_interval
        self.retention_period = retention_period
        self.alert_evaluation_interval = alert_evaluation_interval
        
        # Metrics storage
        self.metrics: Dict[str, Metric] = {}
        self.metric_history: Dict[str, MetricHistory] = {}
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Alerting
        self.alert_rules: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.system_health: Dict[str, Any] = {}
        
        # Predictive analytics
        self.anomaly_detectors: Dict[str, Any] = {}
        self.trend_analyzers: Dict[str, Any] = {}
        self.forecasting_models: Dict[str, Any] = {}
        
        # Performance tracking
        self.sla_targets: Dict[str, float] = {}
        self.sla_current: Dict[str, float] = {}
        
        # State management
        self.is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._alert_task: Optional[asyncio.Task] = None
        
        # Initialize default metrics and checks
        self._initialize_system_metrics()
        self._initialize_default_alerts()
        self._initialize_health_checks()
    
    def _initialize_system_metrics(self):
        """Initialize standard system metrics."""
        system_metrics = [
            ("cpu_usage_percent", MetricType.GAUGE, "CPU usage percentage", "%"),
            ("memory_usage_percent", MetricType.GAUGE, "Memory usage percentage", "%"),
            ("disk_usage_percent", MetricType.GAUGE, "Disk usage percentage", "%"),
            ("network_latency_ms", MetricType.GAUGE, "Network latency", "ms"),
            ("request_rate", MetricType.GAUGE, "Request rate", "req/s"),
            ("error_rate", MetricType.GAUGE, "Error rate", "errors/s"),
            ("response_time_ms", MetricType.HISTOGRAM, "Response time", "ms"),
            ("active_connections", MetricType.GAUGE, "Active connections", "count"),
            ("queue_length", MetricType.GAUGE, "Queue length", "count"),
            ("throughput", MetricType.GAUGE, "System throughput", "ops/s")
        ]
        
        for name, metric_type, description, unit in system_metrics:
            metric = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit
            )
            self.metrics[name] = metric
            self.metric_history[name] = MetricHistory(name=name)
    
    def _initialize_default_alerts(self):
        """Initialize default alert rules."""
        alert_rules = [
            {
                "name": "HighCPUUsage",
                "condition": "cpu_usage_percent > 85",
                "severity": AlertSeverity.WARNING,
                "message": "CPU usage is above 85%"
            },
            {
                "name": "CriticalCPUUsage", 
                "condition": "cpu_usage_percent > 95",
                "severity": AlertSeverity.CRITICAL,
                "message": "CPU usage is critically high (>95%)"
            },
            {
                "name": "HighMemoryUsage",
                "condition": "memory_usage_percent > 90",
                "severity": AlertSeverity.WARNING,
                "message": "Memory usage is above 90%"
            },
            {
                "name": "HighErrorRate",
                "condition": "error_rate > 10",
                "severity": AlertSeverity.ERROR,
                "message": "Error rate is above 10 errors/second"
            },
            {
                "name": "HighLatency",
                "condition": "network_latency_ms > 500",
                "severity": AlertSeverity.WARNING,
                "message": "Network latency is above 500ms"
            },
            {
                "name": "LowThroughput",
                "condition": "throughput < 10",
                "severity": AlertSeverity.WARNING,
                "message": "System throughput is below 10 ops/s"
            }
        ]
        
        for rule in alert_rules:
            alert_id = f"alert_{rule['name'].lower()}"
            alert = Alert(
                alert_id=alert_id,
                name=rule["name"],
                condition=rule["condition"],
                severity=rule["severity"],
                state=AlertState.RESOLVED,
                message=rule["message"]
            )
            self.alert_rules[alert_id] = alert
    
    def _initialize_health_checks(self):
        """Initialize system health checks."""
        
        async def check_disk_space():
            """Check available disk space."""
            # Simulate disk space check
            disk_usage = self.get_metric_value("disk_usage_percent")
            return disk_usage < 90
        
        async def check_memory_pressure():
            """Check memory pressure."""
            memory_usage = self.get_metric_value("memory_usage_percent")
            return memory_usage < 95
        
        async def check_response_time():
            """Check average response time."""
            response_time = self.get_metric_value("response_time_ms")
            return response_time < 1000  # 1 second
        
        async def check_error_rate():
            """Check error rate."""
            error_rate = self.get_metric_value("error_rate")
            return error_rate < 5  # 5 errors per second
        
        health_checks = [
            ("disk_space", check_disk_space, 60.0, 10.0),
            ("memory_pressure", check_memory_pressure, 30.0, 5.0),
            ("response_time", check_response_time, 30.0, 5.0),
            ("error_rate", check_error_rate, 15.0, 3.0)
        ]
        
        for name, check_func, interval, timeout in health_checks:
            self.health_checks[name] = HealthCheck(
                name=name,
                check_function=check_func,
                interval=interval,
                timeout=timeout
            )
    
    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Register a new metric."""
        metric = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit,
            labels=labels or {}
        )
        
        self.metrics[name] = metric
        self.metric_history[name] = MetricHistory(name=name)
        
        logger.info(f"Registered metric: {name} ({metric_type.value})")
        return metric
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return
        
        timestamp = time.time()
        metric = self.metrics[name]
        
        # Update metric
        if metric.metric_type == MetricType.COUNTER:
            metric.value += value  # Counters accumulate
        else:
            metric.value = value  # Gauges, histograms use latest value
        
        metric.timestamp = timestamp
        if labels:
            metric.labels.update(labels)
        
        # Add to history
        history = self.metric_history[name]
        history.values.append((timestamp, value))
        
        # Keep history within retention period
        cutoff_time = timestamp - self.retention_period
        history.values = [
            (t, v) for t, v in history.values
            if t > cutoff_time
        ]
        
        # Update aggregations
        self._update_metric_aggregations(name)
    
    def _update_metric_aggregations(self, name: str):
        """Update metric aggregations."""
        history = self.metric_history[name]
        
        if not history.values:
            return
        
        values = [v for _, v in history.values]
        
        history.aggregations = {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values)
        }
        
        # Percentiles
        if len(values) >= 10:
            sorted_values = sorted(values)
            history.aggregations.update({
                "p50": sorted_values[int(len(sorted_values) * 0.5)],
                "p90": sorted_values[int(len(sorted_values) * 0.9)],
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)]
            })
    
    def get_metric_value(self, name: str) -> float:
        """Get current metric value."""
        if name in self.metrics:
            return self.metrics[name].value
        return 0.0
    
    def get_metric_history(self, name: str, duration: Optional[float] = None) -> List[Tuple[float, float]]:
        """Get metric history."""
        if name not in self.metric_history:
            return []
        
        history = self.metric_history[name].values
        
        if duration:
            cutoff_time = time.time() - duration
            history = [(t, v) for t, v in history if t > cutoff_time]
        
        return history
    
    def register_custom_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register custom metric collector."""
        self.custom_collectors[name] = collector_func
        logger.info(f"Registered custom collector: {name}")
    
    def add_alert_rule(
        self,
        name: str,
        condition: str,
        severity: AlertSeverity,
        message: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None
    ) -> str:
        """Add new alert rule."""
        alert_id = f"alert_{name.lower().replace(' ', '_')}"
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            condition=condition,
            severity=severity,
            state=AlertState.RESOLVED,
            message=message,
            labels=labels or {},
            annotations=annotations or {}
        )
        
        self.alert_rules[alert_id] = alert
        logger.info(f"Added alert rule: {name}")
        return alert_id
    
    def _evaluate_alert_condition(self, condition: str) -> bool:
        """Evaluate alert condition."""
        try:
            # Simple condition evaluation (in production, use proper expression parser)
            # Replace metric names with their current values
            expression = condition
            
            for metric_name in self.metrics:
                if metric_name in expression:
                    value = self.get_metric_value(metric_name)
                    expression = expression.replace(metric_name, str(value))
            
            # Evaluate the expression
            return eval(expression)
            
        except Exception as e:
            logger.error(f"Error evaluating alert condition '{condition}': {e}")
            return False
    
    async def _evaluate_alerts(self):
        """Evaluate all alert rules."""
        current_time = time.time()
        
        for alert_id, alert in self.alert_rules.items():
            try:
                condition_met = self._evaluate_alert_condition(alert.condition)
                
                if condition_met and alert.state == AlertState.RESOLVED:
                    # Fire alert
                    alert.state = AlertState.FIRING
                    alert.triggered_at = current_time
                    self.active_alerts[alert_id] = alert
                    
                    # Record alert event
                    alert_event = {
                        "timestamp": current_time,
                        "alert_id": alert_id,
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "state": "fired",
                        "message": alert.message
                    }
                    self.alert_history.append(alert_event)
                    
                    logger.warning(f"ALERT FIRED: {alert.name} - {alert.message}")
                
                elif not condition_met and alert.state == AlertState.FIRING:
                    # Resolve alert
                    alert.state = AlertState.RESOLVED
                    alert.resolved_at = current_time
                    
                    if alert_id in self.active_alerts:
                        del self.active_alerts[alert_id]
                    
                    # Record resolution
                    alert_event = {
                        "timestamp": current_time,
                        "alert_id": alert_id,
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "state": "resolved",
                        "message": f"Alert resolved: {alert.message}"
                    }
                    self.alert_history.append(alert_event)
                    
                    logger.info(f"ALERT RESOLVED: {alert.name}")
                    
            except Exception as e:
                logger.error(f"Error evaluating alert {alert_id}: {e}")
        
        # Keep alert history manageable
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]
    
    async def _run_health_checks(self):
        """Run all health checks."""
        current_time = time.time()
        
        for name, health_check in self.health_checks.items():
            try:
                # Check if it's time to run this check
                if (health_check.last_check is None or 
                    current_time - health_check.last_check >= health_check.interval):
                    
                    # Run the health check
                    try:
                        result = await asyncio.wait_for(
                            health_check.check_function(),
                            timeout=health_check.timeout
                        )
                        
                        health_check.last_result = result
                        health_check.last_check = current_time
                        
                        if result:
                            health_check.consecutive_failures = 0
                        else:
                            health_check.consecutive_failures += 1
                            logger.warning(f"Health check failed: {name} "
                                         f"(consecutive failures: {health_check.consecutive_failures})")
                    
                    except asyncio.TimeoutError:
                        health_check.last_result = False
                        health_check.last_check = current_time
                        health_check.consecutive_failures += 1
                        logger.error(f"Health check timed out: {name}")
                    
                    except Exception as e:
                        health_check.last_result = False
                        health_check.last_check = current_time
                        health_check.consecutive_failures += 1
                        logger.error(f"Health check error: {name} - {e}")
                        
            except Exception as e:
                logger.error(f"Error running health check {name}: {e}")
        
        # Update overall system health
        self._update_system_health()
    
    def _update_system_health(self):
        """Update overall system health status."""
        total_checks = len(self.health_checks)
        if total_checks == 0:
            self.system_health = {"status": "unknown", "details": "No health checks configured"}
            return
        
        healthy_checks = sum(
            1 for check in self.health_checks.values()
            if check.last_result is True
        )
        
        failed_checks = [
            name for name, check in self.health_checks.items()
            if check.last_result is False
        ]
        
        critical_failures = [
            name for name, check in self.health_checks.items()
            if check.consecutive_failures >= 3
        ]
        
        # Determine overall status
        if len(critical_failures) > 0:
            status = "critical"
        elif len(failed_checks) > total_checks * 0.5:
            status = "degraded"
        elif len(failed_checks) > 0:
            status = "warning"
        else:
            status = "healthy"
        
        self.system_health = {
            "status": status,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "failed_checks": failed_checks,
            "critical_failures": critical_failures,
            "last_updated": time.time()
        }
    
    async def _collect_metrics(self):
        """Collect metrics from custom collectors."""
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                metrics_data = collector_func()
                
                for metric_name, value in metrics_data.items():
                    if metric_name in self.metrics:
                        self.record_metric(metric_name, value)
                    else:
                        # Auto-register unknown metrics as gauges
                        self.register_metric(
                            metric_name,
                            MetricType.GAUGE,
                            f"Auto-registered from {collector_name}",
                            "unknown"
                        )
                        self.record_metric(metric_name, value)
                        
            except Exception as e:
                logger.error(f"Error collecting metrics from {collector_name}: {e}")
    
    def detect_anomalies(self, metric_name: str, window_size: int = 50) -> Dict[str, Any]:
        """Detect anomalies in metric data using statistical methods."""
        if metric_name not in self.metric_history:
            return {"anomalies": [], "baseline": None}
        
        values = [v for _, v in self.metric_history[metric_name].values[-window_size:]]
        
        if len(values) < 10:
            return {"anomalies": [], "baseline": None}
        
        # Calculate statistical baseline
        mean_val = statistics.mean(values[:-5])  # Exclude recent values
        std_val = statistics.stdev(values[:-5]) if len(values[:-5]) > 1 else 0
        
        # Detect anomalies in recent values
        recent_values = values[-5:]
        anomalies = []
        
        for i, value in enumerate(recent_values):
            if std_val > 0:
                z_score = abs(value - mean_val) / std_val
                if z_score > 2.5:  # 2.5 standard deviations
                    anomalies.append({
                        "index": len(values) - 5 + i,
                        "value": value,
                        "z_score": z_score,
                        "severity": "high" if z_score > 3.0 else "medium"
                    })
        
        return {
            "anomalies": anomalies,
            "baseline": {"mean": mean_val, "std_dev": std_val},
            "threshold": 2.5
        }
    
    def predict_metric_trend(self, metric_name: str, horizon_minutes: int = 30) -> Dict[str, Any]:
        """Predict metric trend using simple linear regression."""
        if metric_name not in self.metric_history:
            return {"prediction": None, "confidence": 0.0}
        
        values = self.metric_history[metric_name].values[-20:]  # Last 20 values
        
        if len(values) < 5:
            return {"prediction": None, "confidence": 0.0}
        
        # Simple linear regression
        n = len(values)
        timestamps = [t for t, _ in values]
        metric_values = [v for _, v in values]
        
        # Normalize timestamps
        start_time = timestamps[0]
        x_values = [(t - start_time) / 60.0 for t in timestamps]  # Convert to minutes
        
        # Calculate regression coefficients
        sum_x = sum(x_values)
        sum_y = sum(metric_values)
        sum_xy = sum(x * y for x, y in zip(x_values, metric_values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return {"prediction": None, "confidence": 0.0}
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_x = x_values[-1] + horizon_minutes
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence based on R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in metric_values)
        ss_res = sum((metric_values[i] - (slope * x_values[i] + intercept)) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0.0, min(1.0, r_squared))
        
        return {
            "prediction": predicted_value,
            "confidence": confidence,
            "trend": "increasing" if slope > 0 else "decreasing",
            "slope": slope,
            "horizon_minutes": horizon_minutes
        }
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Comprehensive monitoring started")
        
        while self.is_running:
            try:
                # Collect metrics
                await self._collect_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _alert_loop(self):
        """Alert evaluation loop."""
        while self.is_running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(self.alert_evaluation_interval)
                
            except Exception as e:
                logger.error(f"Error in alert loop: {e}")
                await asyncio.sleep(self.alert_evaluation_interval)
    
    async def _health_check_loop(self):
        """Health check loop."""
        while self.is_running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(10.0)  # Run health checks every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10.0)
    
    async def start(self):
        """Start the comprehensive monitor."""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        self.is_running = True
        
        # Start all monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._alert_task = asyncio.create_task(self._alert_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Comprehensive monitoring started")
    
    async def stop(self):
        """Stop the comprehensive monitor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all tasks
        for task in [self._monitoring_task, self._alert_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Comprehensive monitoring stopped")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        return {
            "system_health": self.system_health,
            "metrics": {
                "total": len(self.metrics),
                "with_data": len([h for h in self.metric_history.values() if h.values])
            },
            "alerts": {
                "total_rules": len(self.alert_rules),
                "active": len(self.active_alerts),
                "recent_24h": len([
                    a for a in self.alert_history
                    if time.time() - a["timestamp"] <= 86400
                ])
            },
            "health_checks": {
                "total": len(self.health_checks),
                "healthy": len([
                    c for c in self.health_checks.values()
                    if c.last_result is True
                ]),
                "failed": len([
                    c for c in self.health_checks.values()
                    if c.last_result is False
                ])
            },
            "collectors": len(self.custom_collectors)
        }
    
    def export_monitoring_data(self) -> Dict[str, Any]:
        """Export comprehensive monitoring data."""
        return {
            "metrics": {
                name: {
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "current_value": metric.value,
                    "history_points": len(self.metric_history[name].values),
                    "aggregations": self.metric_history[name].aggregations
                }
                for name, metric in self.metrics.items()
            },
            "alerts": {
                "rules": [
                    {
                        "alert_id": alert.alert_id,
                        "name": alert.name,
                        "condition": alert.condition,
                        "severity": alert.severity.value,
                        "state": alert.state.value,
                        "message": alert.message
                    }
                    for alert in self.alert_rules.values()
                ],
                "active": [
                    {
                        "alert_id": alert.alert_id,
                        "name": alert.name,
                        "triggered_at": alert.triggered_at,
                        "severity": alert.severity.value
                    }
                    for alert in self.active_alerts.values()
                ],
                "history": self.alert_history[-50:]  # Last 50 events
            },
            "system_health": self.system_health,
            "summary": self.get_monitoring_summary()
        }