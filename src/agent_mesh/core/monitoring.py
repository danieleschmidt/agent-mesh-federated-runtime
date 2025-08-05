"""Advanced monitoring, metrics, and observability for Agent Mesh.

This module provides comprehensive monitoring capabilities including
metrics collection, health checks, performance tracking, and alerting.
"""

import asyncio
import json
import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Union
from uuid import UUID, uuid4

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest

from .validation import sanitize_for_logging


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class HealthStatus(Enum):
    """Health check status levels."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    objectives: Optional[Dict[float, float]] = None  # For summaries


@dataclass
class HealthCheck:
    """Health check definition."""
    
    name: str
    check_function: Callable[[], bool]
    interval_seconds: float = 30.0
    timeout_seconds: float = 10.0
    critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition."""
    
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: Union[int, float]
    alert_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average: List[float] = field(default_factory=list)
    open_file_descriptors: int = 0
    thread_count: int = 0
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("metrics_collector", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Prometheus metrics registry
        self.registry = CollectorRegistry()
        
        # Built-in metrics
        self._setup_builtin_metrics()
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Time series data storage
        self.time_series_data = defaultdict(lambda: deque(maxlen=1000))
        
        # Metric collection state
        self.collection_enabled = False
        self.collection_interval = 10.0  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self._last_network_stats = None
    
    def _setup_builtin_metrics(self) -> None:
        """Setup built-in Prometheus metrics."""
        # System metrics
        self.cpu_usage = Gauge('agent_mesh_cpu_usage_percent', 
                              'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('agent_mesh_memory_usage_percent', 
                                 'Memory usage percentage', registry=self.registry)
        self.memory_used = Gauge('agent_mesh_memory_used_bytes', 
                                'Memory used in bytes', registry=self.registry)
        
        # Network metrics
        self.network_bytes_sent = Counter('agent_mesh_network_bytes_sent_total', 
                                         'Total bytes sent over network', registry=self.registry)
        self.network_bytes_received = Counter('agent_mesh_network_bytes_received_total', 
                                            'Total bytes received over network', registry=self.registry)
        
        # Application metrics
        self.task_duration = Histogram('agent_mesh_task_duration_seconds', 
                                      'Task execution duration',
                                      buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
                                      registry=self.registry)
        self.task_counter = Counter('agent_mesh_tasks_total', 
                                   'Total number of tasks', 
                                   ['status'], registry=self.registry)
        
        # Consensus metrics
        self.consensus_rounds = Counter('agent_mesh_consensus_rounds_total', 
                                       'Total consensus rounds', registry=self.registry)
        self.consensus_duration = Histogram('agent_mesh_consensus_duration_seconds', 
                                           'Consensus round duration',
                                           buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                                           registry=self.registry)
        
        # P2P network metrics
        self.peer_connections = Gauge('agent_mesh_peer_connections', 
                                     'Number of peer connections', registry=self.registry)
        self.messages_sent = Counter('agent_mesh_messages_sent_total', 
                                    'Total messages sent', ['message_type'], registry=self.registry)
        self.messages_received = Counter('agent_mesh_messages_received_total', 
                                        'Total messages received', ['message_type'], registry=self.registry)
    
    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self.collection_enabled:
            return
        
        self.collection_enabled = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        self.logger.info("Metrics collection started", 
                        interval=self.collection_interval)
    
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.collection_enabled = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Metrics collection stopped")
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a custom metric."""
        if definition.metric_type == MetricType.COUNTER:
            metric = Counter(definition.name, definition.description, 
                           definition.labels, registry=self.registry)
        elif definition.metric_type == MetricType.GAUGE:
            metric = Gauge(definition.name, definition.description, 
                          definition.labels, registry=self.registry)
        elif definition.metric_type == MetricType.HISTOGRAM:
            metric = Histogram(definition.name, definition.description, 
                             definition.labels, definition.buckets or [0.1, 0.5, 1.0, 2.0, 5.0],
                             registry=self.registry)
        elif definition.metric_type == MetricType.SUMMARY:
            metric = Summary(definition.name, definition.description, 
                           definition.labels, registry=self.registry)
        else:
            raise ValueError(f"Unsupported metric type: {definition.metric_type}")
        
        self.custom_metrics[definition.name] = metric
        self.metric_definitions[definition.name] = definition
        
        self.logger.info("Custom metric registered", 
                        name=definition.name, 
                        type=definition.metric_type.value)
    
    def record_metric(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        if name not in self.custom_metrics:
            self.logger.warning("Unknown metric", name=name)
            return
        
        metric = self.custom_metrics[name]
        definition = self.metric_definitions[name]
        
        try:
            if definition.metric_type == MetricType.COUNTER:
                if labels:
                    metric.labels(**labels).inc(value)
                else:
                    metric.inc(value)
            elif definition.metric_type == MetricType.GAUGE:
                if labels:
                    metric.labels(**labels).set(value)
                else:
                    metric.set(value)
            elif definition.metric_type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                if labels:
                    metric.labels(**labels).observe(value)
                else:
                    metric.observe(value)
            
            # Store in time series
            self.time_series_data[name].append((time.time(), value, labels or {}))
            
        except Exception as e:
            self.logger.error("Failed to record metric", 
                            name=name, value=value, error=str(e))
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            
            metrics = PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [],
                open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
                thread_count=process.num_threads()
            )
            
            # Update Prometheus metrics
            self.cpu_usage.set(cpu_percent)
            self.memory_usage.set(memory.percent)
            self.memory_used.set(memory.used)
            
            # Network delta calculation
            if self._last_network_stats:
                bytes_sent_delta = network.bytes_sent - self._last_network_stats.bytes_sent
                bytes_recv_delta = network.bytes_recv - self._last_network_stats.bytes_recv
                if bytes_sent_delta > 0:
                    self.network_bytes_sent.inc(bytes_sent_delta)
                if bytes_recv_delta > 0:
                    self.network_bytes_received.inc(bytes_recv_delta)
            
            self._last_network_stats = network
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect performance metrics", error=str(e))
            return PerformanceMetrics()
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_time_series_data(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        if metric_name not in self.time_series_data:
            return []
        
        cutoff_time = time.time() - (hours * 3600)
        data = []
        
        for timestamp, value, labels in self.time_series_data[metric_name]:
            if timestamp >= cutoff_time:
                data.append({
                    'timestamp': timestamp,
                    'value': value,
                    'labels': labels
                })
        
        return data
    
    async def _collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self.collection_enabled:
            try:
                # Collect system performance metrics
                metrics = self.get_performance_metrics()
                self.performance_history.append(metrics)
                
                # Log key metrics periodically
                if len(self.performance_history) % 6 == 0:  # Every minute if 10s interval
                    self.logger.info("System performance",
                                    cpu_percent=metrics.cpu_percent,
                                    memory_percent=metrics.memory_percent,
                                    memory_used_mb=round(metrics.memory_used_mb, 1),
                                    thread_count=metrics.thread_count)
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(self.collection_interval)


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("health_monitor", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        self.health_results: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring state
        self.monitoring_enabled = False
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        
        # Overall system health
        self.overall_health = HealthStatus.HEALTHY
        self.health_history: deque = deque(maxlen=100)
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = HealthStatus.HEALTHY
        self.health_results[health_check.name] = {
            "status": HealthStatus.HEALTHY.value,
            "last_check": time.time(),
            "consecutive_failures": 0,
            "total_checks": 0,
            "total_failures": 0
        }
        
        self.logger.info("Health check registered", 
                        name=health_check.name,
                        critical=health_check.critical,
                        interval=health_check.interval_seconds)
    
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        # Start monitoring tasks for each health check
        for name, health_check in self.health_checks.items():
            task = asyncio.create_task(self._health_check_loop(name, health_check))
            self._monitoring_tasks[name] = task
        
        self.logger.info("Health monitoring started", 
                        checks=len(self.health_checks))
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_enabled = False
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        
        self._monitoring_tasks.clear()
        self.logger.info("Health monitoring stopped")
    
    async def check_health(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform health check(s) and return results."""
        if check_name:
            if check_name not in self.health_checks:
                return {"error": f"Unknown health check: {check_name}"}
            
            return await self._perform_health_check(check_name, self.health_checks[check_name])
        else:
            # Check all health checks
            results = {}
            for name, health_check in self.health_checks.items():
                results[name] = await self._perform_health_check(name, health_check)
            
            # Calculate overall health
            self._update_overall_health()
            results["overall"] = {
                "status": self.overall_health.value,
                "timestamp": time.time()
            }
            
            return results
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        healthy_checks = sum(1 for status in self.health_status.values() 
                           if status == HealthStatus.HEALTHY)
        total_checks = len(self.health_status)
        
        return {
            "overall_health": self.overall_health.value,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "health_percentage": (healthy_checks / max(1, total_checks)) * 100,
            "check_details": {
                name: {
                    "status": status.value,
                    "last_check": self.health_results[name]["last_check"],
                    "consecutive_failures": self.health_results[name]["consecutive_failures"]
                }
                for name, status in self.health_status.items()
            }
        }
    
    async def _health_check_loop(self, name: str, health_check: HealthCheck) -> None:
        """Background loop for individual health check."""
        while self.monitoring_enabled:
            try:
                await self._perform_health_check(name, health_check)
                await asyncio.sleep(health_check.interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check loop error", 
                                check=name, error=str(e))
                await asyncio.sleep(health_check.interval_seconds)
    
    async def _perform_health_check(self, name: str, health_check: HealthCheck) -> Dict[str, Any]:
        """Perform individual health check."""
        start_time = time.time()
        result = self.health_results[name]
        result["total_checks"] += 1
        
        try:
            # Execute health check with timeout
            check_result = await asyncio.wait_for(
                asyncio.to_thread(health_check.check_function),
                timeout=health_check.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            if check_result:
                self.health_status[name] = HealthStatus.HEALTHY
                result["consecutive_failures"] = 0
                result["status"] = HealthStatus.HEALTHY.value
            else:
                self.health_status[name] = HealthStatus.UNHEALTHY
                result["consecutive_failures"] += 1
                result["total_failures"] += 1
                result["status"] = HealthStatus.UNHEALTHY.value
                
                self.logger.warning("Health check failed", 
                                  check=name,
                                  consecutive_failures=result["consecutive_failures"])
            
            result["last_check"] = time.time()
            result["duration"] = duration
            
            return {
                "status": result["status"],
                "duration": duration,
                "timestamp": result["last_check"],
                "consecutive_failures": result["consecutive_failures"]
            }
            
        except asyncio.TimeoutError:
            self.health_status[name] = HealthStatus.UNHEALTHY
            result["consecutive_failures"] += 1
            result["total_failures"] += 1
            result["status"] = HealthStatus.UNHEALTHY.value
            result["last_check"] = time.time()
            
            self.logger.error("Health check timeout", 
                            check=name, timeout=health_check.timeout_seconds)
            
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "error": "timeout",
                "timeout": health_check.timeout_seconds,
                "timestamp": result["last_check"]
            }
            
        except Exception as e:
            self.health_status[name] = HealthStatus.UNHEALTHY
            result["consecutive_failures"] += 1
            result["total_failures"] += 1
            result["status"] = HealthStatus.UNHEALTHY.value
            result["last_check"] = time.time()
            
            self.logger.error("Health check exception", 
                            check=name, error=str(e))
            
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "error": str(e),
                "timestamp": result["last_check"]
            }
    
    def _update_overall_health(self) -> None:
        """Update overall system health based on individual checks."""
        if not self.health_status:
            self.overall_health = HealthStatus.HEALTHY
            return
        
        critical_checks = [name for name, check in self.health_checks.items() if check.critical]
        
        # Check critical health checks first
        critical_unhealthy = any(
            self.health_status[name] == HealthStatus.UNHEALTHY 
            for name in critical_checks
        )
        
        if critical_unhealthy:
            self.overall_health = HealthStatus.CRITICAL
            return
        
        # Count overall health status
        unhealthy_count = sum(1 for status in self.health_status.values() 
                            if status == HealthStatus.UNHEALTHY)
        total_count = len(self.health_status)
        
        if unhealthy_count == 0:
            self.overall_health = HealthStatus.HEALTHY
        elif unhealthy_count / total_count < 0.3:  # Less than 30% unhealthy
            self.overall_health = HealthStatus.DEGRADED
        else:
            self.overall_health = HealthStatus.UNHEALTHY
        
        # Record health history
        self.health_history.append({
            "timestamp": time.time(),
            "status": self.overall_health.value,
            "unhealthy_count": unhealthy_count,
            "total_count": total_count
        })


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("alert_manager", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Alert storage
        self.active_alerts: Dict[UUID, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Alert rules and thresholds
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
    
    def add_alert_rule(
        self, 
        name: str, 
        metric_name: str, 
        condition: str, 
        threshold: Union[int, float],
        severity: AlertSeverity = AlertSeverity.WARNING
    ) -> None:
        """Add alert rule for metric monitoring."""
        self.alert_rules[name] = {
            "metric_name": metric_name,
            "condition": condition,  # "greater_than", "less_than", "equals"
            "threshold": threshold,
            "severity": severity,
            "created_at": time.time()
        }
        
        self.logger.info("Alert rule added", 
                        name=name, metric=metric_name, 
                        condition=condition, threshold=threshold)
    
    def register_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register notification handler for alerts."""
        self.notification_handlers.append(handler)
        self.logger.info("Notification handler registered")
    
    async def check_alert_conditions(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check alert conditions against current metrics."""
        new_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule["metric_name"]
            
            if metric_name not in metrics:
                continue
            
            current_value = metrics[metric_name]
            threshold = rule["threshold"]
            condition = rule["condition"]
            
            alert_triggered = False
            
            if condition == "greater_than" and current_value > threshold:
                alert_triggered = True
            elif condition == "less_than" and current_value < threshold:
                alert_triggered = True
            elif condition == "equals" and current_value == threshold:
                alert_triggered = True
            
            if alert_triggered:
                # Check if alert already exists
                existing_alert = None
                for alert in self.active_alerts.values():
                    if alert.name == rule_name and not alert.resolved:
                        existing_alert = alert
                        break
                
                if not existing_alert:
                    alert = Alert(
                        name=rule_name,
                        description=f"Metric {metric_name} {condition} {threshold} (current: {current_value})",
                        severity=rule["severity"],
                        condition=f"{metric_name} {condition} {threshold}",
                        threshold=threshold,
                        metadata={
                            "metric_name": metric_name,
                            "current_value": current_value,
                            "rule": rule
                        }
                    )
                    
                    await self._trigger_alert(alert)
                    new_alerts.append(alert)
        
        return new_alerts
    
    async def acknowledge_alert(self, alert_id: UUID, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.metadata["acknowledged_by"] = acknowledged_by
        alert.metadata["acknowledged_at"] = time.time()
        
        self.logger.info("Alert acknowledged", 
                        alert_id=str(alert_id), 
                        name=alert.name,
                        acknowledged_by=acknowledged_by)
        
        return True
    
    async def resolve_alert(self, alert_id: UUID, resolved_by: str = "system") -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.metadata["resolved_by"] = resolved_by
        alert.metadata["resolved_at"] = time.time()
        
        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        self.logger.info("Alert resolved", 
                        alert_id=str(alert_id), 
                        name=alert.name,
                        resolved_by=resolved_by)
        
        return True
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        total_count = active_count + len(self.alert_history)
        
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        return {
            "active_alerts": active_count,
            "total_alerts": total_count,
            "severity_breakdown": dict(severity_counts),
            "alert_rules": len(self.alert_rules),
            "notification_handlers": len(self.notification_handlers)
        }
    
    async def _trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and send notifications."""
        self.active_alerts[alert.alert_id] = alert
        
        self.logger.warning("Alert triggered", 
                          alert_id=str(alert.alert_id),
                          name=alert.name,
                          severity=alert.severity.value,
                          condition=alert.condition)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                await asyncio.to_thread(handler, alert)
            except Exception as e:
                self.logger.error("Notification handler failed", 
                                error=str(e), alert_id=str(alert.alert_id))


class ComprehensiveMonitor:
    """Comprehensive monitoring system combining metrics, health, and alerts."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("comprehensive_monitor", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Component systems
        self.metrics_collector = MetricsCollector(node_id)
        self.health_monitor = HealthMonitor(node_id)
        self.alert_manager = AlertManager(node_id)
        
        # Monitoring state
        self.monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    async def start(self) -> None:
        """Start comprehensive monitoring."""
        if self.monitoring_enabled:
            return
        
        self.monitoring_enabled = True
        
        # Start component systems
        await self.metrics_collector.start_collection()
        await self.health_monitor.start_monitoring()
        
        # Start integrated monitoring loop
        self._monitoring_task = asyncio.create_task(self._integrated_monitoring_loop())
        
        self.logger.info("Comprehensive monitoring started")
    
    async def stop(self) -> None:
        """Stop comprehensive monitoring."""
        self.monitoring_enabled = False
        
        # Stop integrated monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop component systems
        await self.metrics_collector.stop_collection()
        await self.health_monitor.stop_monitoring()
        
        self.logger.info("Comprehensive monitoring stopped")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            "timestamp": time.time(),
            "node_id": str(self.node_id) if self.node_id else None,
            "system_performance": self.metrics_collector.get_performance_metrics().__dict__,
            "health_summary": self.health_monitor.get_health_summary(),
            "alert_statistics": self.alert_manager.get_alert_statistics(),
            "active_alerts": [
                {
                    "id": str(alert.alert_id),
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "description": alert.description,
                    "timestamp": alert.timestamp,
                    "acknowledged": alert.acknowledged
                }
                for alert in self.alert_manager.get_active_alerts()
            ]
        }
    
    async def _integrated_monitoring_loop(self) -> None:
        """Integrated monitoring loop combining all systems."""
        while self.monitoring_enabled:
            try:
                # Get current performance metrics
                perf_metrics = self.metrics_collector.get_performance_metrics()
                
                # Convert to dict for alert checking
                metrics_dict = {
                    "cpu_percent": perf_metrics.cpu_percent,
                    "memory_percent": perf_metrics.memory_percent,
                    "disk_usage_percent": perf_metrics.disk_usage_percent,
                    "thread_count": float(perf_metrics.thread_count)
                }
                
                # Check alert conditions
                await self.alert_manager.check_alert_conditions(metrics_dict)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Integrated monitoring loop error", error=str(e))
                await asyncio.sleep(30)
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        # System health check
        def system_health_check() -> bool:
            try:
                # Check if system resources are within reasonable limits
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                return cpu_percent < 95 and memory.percent < 95
            except Exception:
                return False
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="system_resources",
                check_function=system_health_check,
                interval_seconds=30.0,
                critical=True
            )
        )
        
        # Network connectivity check
        def network_health_check() -> bool:
            try:
                # Simple check if network interfaces are up
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                return True
            except Exception:
                return False
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="network_connectivity",
                check_function=network_health_check,
                interval_seconds=60.0,
                critical=False
            )
        )
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        # High CPU usage
        self.alert_manager.add_alert_rule(
            "high_cpu_usage",
            "cpu_percent",
            "greater_than",
            80.0,
            AlertSeverity.WARNING
        )
        
        # High memory usage
        self.alert_manager.add_alert_rule(
            "high_memory_usage",
            "memory_percent",
            "greater_than",
            85.0,
            AlertSeverity.WARNING
        )
        
        # Critical CPU usage
        self.alert_manager.add_alert_rule(
            "critical_cpu_usage",
            "cpu_percent",
            "greater_than",
            95.0,
            AlertSeverity.CRITICAL
        )
        
        # Critical memory usage
        self.alert_manager.add_alert_rule(
            "critical_memory_usage",
            "memory_percent",
            "greater_than",
            95.0,
            AlertSeverity.CRITICAL
        )


# Global monitor instance
_global_monitor: Optional[ComprehensiveMonitor] = None


def get_monitor(node_id: Optional[UUID] = None) -> ComprehensiveMonitor:
    """Get global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ComprehensiveMonitor(node_id)
    return _global_monitor


async def start_monitoring(node_id: Optional[UUID] = None) -> None:
    """Start global monitoring."""
    monitor = get_monitor(node_id)
    await monitor.start()


async def stop_monitoring() -> None:
    """Stop global monitoring."""
    if _global_monitor:
        await _global_monitor.stop()