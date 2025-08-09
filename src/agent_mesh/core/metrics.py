"""Comprehensive metrics collection and monitoring for Agent Mesh.

Provides real-time metrics collection, aggregation, and export for monitoring
system performance, health, and operational insights.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from uuid import UUID

import structlog
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)


logger = structlog.get_logger("metrics")


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms


@dataclass
class MetricValue:
    """A metric value with timestamp and labels."""
    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricCollector(ABC):
    """Abstract base class for metric collectors."""
    
    @abstractmethod
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect metrics from the system."""
        pass
    
    @property
    @abstractmethod
    def collector_name(self) -> str:
        """Get collector name."""
        pass


class NetworkMetricsCollector(MetricCollector):
    """Metrics collector for network operations."""
    
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.logger = structlog.get_logger("network_metrics")
    
    @property
    def collector_name(self) -> str:
        return "network"
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect network metrics."""
        metrics = []
        
        try:
            # Get network statistics
            stats = await self.network_manager.get_statistics()
            
            # Connection metrics
            metrics.extend([
                MetricValue("mesh_connections_active", stats.get("connections_active", 0)),
                MetricValue("mesh_connections_total", stats.get("connections_total", 0)),
                MetricValue("mesh_messages_sent_total", stats.get("messages_sent", 0)),
                MetricValue("mesh_messages_received_total", stats.get("messages_received", 0)),
                MetricValue("mesh_bytes_sent_total", stats.get("bytes_sent", 0)),
                MetricValue("mesh_bytes_received_total", stats.get("bytes_received", 0)),
                MetricValue("mesh_avg_latency_ms", stats.get("avg_latency_ms", 0)),
                MetricValue("mesh_uptime_seconds", stats.get("uptime_seconds", 0))
            ])
            
            # Peer metrics
            peers = await self.network_manager.get_connected_peers()
            metrics.append(MetricValue("mesh_peers_connected", len(peers)))
            
            # Protocol distribution
            protocol_counts = {}
            for peer in peers:
                for protocol in peer.protocols:
                    protocol_name = protocol.value
                    protocol_counts[protocol_name] = protocol_counts.get(protocol_name, 0) + 1
            
            for protocol, count in protocol_counts.items():
                metrics.append(MetricValue(
                    "mesh_peers_by_protocol", 
                    count, 
                    labels={"protocol": protocol}
                ))
            
            # Error metrics if available
            if hasattr(self.network_manager, 'get_error_statistics'):
                error_stats = self.network_manager.get_error_statistics()
                
                for operation, count in error_stats.get("error_counts", {}).items():
                    metrics.append(MetricValue(
                        "mesh_network_errors_total",
                        count,
                        labels={"operation": operation}
                    ))
                
                metrics.extend([
                    MetricValue("mesh_failed_peers", error_stats.get("failed_peers_count", 0)),
                    MetricValue("mesh_blacklisted_peers", error_stats.get("blacklisted_peers_count", 0))
                ])
            
        except Exception as e:
            self.logger.error("Failed to collect network metrics", error=str(e))
        
        return metrics


class SystemMetricsCollector(MetricCollector):
    """Metrics collector for system resources."""
    
    def __init__(self):
        self.logger = structlog.get_logger("system_metrics")
        
        # Try to import psutil for system metrics
        try:
            import psutil
            self.psutil = psutil
            self.has_psutil = True
        except ImportError:
            self.psutil = None
            self.has_psutil = False
            self.logger.warning("psutil not available, system metrics will be limited")
    
    @property
    def collector_name(self) -> str:
        return "system"
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect system metrics."""
        metrics = []
        
        try:
            if self.has_psutil:
                # CPU metrics
                cpu_percent = self.psutil.cpu_percent()
                metrics.append(MetricValue("system_cpu_usage_percent", cpu_percent))
                
                # Memory metrics
                memory = self.psutil.virtual_memory()
                metrics.extend([
                    MetricValue("system_memory_usage_percent", memory.percent),
                    MetricValue("system_memory_used_bytes", memory.used),
                    MetricValue("system_memory_total_bytes", memory.total)
                ])
                
                # Disk metrics
                disk = self.psutil.disk_usage('/')
                metrics.extend([
                    MetricValue("system_disk_usage_percent", (disk.used / disk.total) * 100),
                    MetricValue("system_disk_used_bytes", disk.used),
                    MetricValue("system_disk_total_bytes", disk.total)
                ])
                
                # Network interface metrics
                net_io = self.psutil.net_io_counters()
                metrics.extend([
                    MetricValue("system_network_bytes_sent", net_io.bytes_sent),
                    MetricValue("system_network_bytes_recv", net_io.bytes_recv),
                    MetricValue("system_network_packets_sent", net_io.packets_sent),
                    MetricValue("system_network_packets_recv", net_io.packets_recv)
                ])
            
            # Basic metrics without psutil
            metrics.append(MetricValue("system_timestamp", time.time()))
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
        
        return metrics


class DatabaseMetricsCollector(MetricCollector):
    """Metrics collector for database operations."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = structlog.get_logger("database_metrics")
    
    @property
    def collector_name(self) -> str:
        return "database"
    
    async def collect_metrics(self) -> List[MetricValue]:
        """Collect database metrics."""
        metrics = []
        
        try:
            # Health check
            is_healthy = await self.db_manager.health_check()
            metrics.append(MetricValue("database_healthy", 1 if is_healthy else 0))
            
            # Connection info
            conn_info = self.db_manager.get_connection_info()
            metrics.extend([
                MetricValue("database_pool_size", conn_info.get("pool_size", 0)),
                MetricValue("database_max_overflow", conn_info.get("max_overflow", 0))
            ])
            
        except Exception as e:
            self.logger.error("Failed to collect database metrics", error=str(e))
            metrics.append(MetricValue("database_healthy", 0))
        
        return metrics


class PrometheusExporter:
    """
    Prometheus metrics exporter.
    
    Exports collected metrics in Prometheus format for scraping.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus exporter.
        
        Args:
            registry: Prometheus registry (None for default)
        """
        self.registry = registry or CollectorRegistry()
        self.metrics: Dict[str, Any] = {}  # metric_name -> prometheus metric
        self.logger = structlog.get_logger("prometheus_exporter")
        
        self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """Initialize standard Prometheus metrics."""
        # Network metrics
        self.metrics['mesh_connections_active'] = Gauge(
            'mesh_connections_active',
            'Number of active mesh connections',
            registry=self.registry
        )
        
        self.metrics['mesh_connections_total'] = Counter(
            'mesh_connections_total',
            'Total number of mesh connections',
            registry=self.registry
        )
        
        self.metrics['mesh_messages_sent_total'] = Counter(
            'mesh_messages_sent_total',
            'Total messages sent',
            registry=self.registry
        )
        
        self.metrics['mesh_messages_received_total'] = Counter(
            'mesh_messages_received_total',
            'Total messages received',
            registry=self.registry
        )
        
        self.metrics['mesh_peers_connected'] = Gauge(
            'mesh_peers_connected',
            'Number of connected peers',
            registry=self.registry
        )
        
        self.metrics['mesh_latency_histogram'] = Histogram(
            'mesh_latency_seconds',
            'Message latency distribution',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        # System metrics
        self.metrics['system_cpu_usage_percent'] = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_memory_usage_percent'] = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        # Database metrics
        self.metrics['database_healthy'] = Gauge(
            'database_healthy',
            'Database health status (1=healthy, 0=unhealthy)',
            registry=self.registry
        )
    
    def update_metrics(self, metric_values: List[MetricValue]):
        """Update Prometheus metrics with collected values."""
        for metric_value in metric_values:
            try:
                # Handle different metric types
                if metric_value.name in self.metrics:
                    prom_metric = self.metrics[metric_value.name]
                    
                    if hasattr(prom_metric, '_type') and prom_metric._type == 'counter':
                        # For counters, we need to track the difference
                        if hasattr(self, f'_last_{metric_value.name}'):
                            last_value = getattr(self, f'_last_{metric_value.name}')
                            if metric_value.value > last_value:
                                prom_metric.inc(metric_value.value - last_value)
                        setattr(self, f'_last_{metric_value.name}', metric_value.value)
                    else:
                        # For gauges, set the value directly
                        if metric_value.labels:
                            prom_metric.labels(**metric_value.labels).set(metric_value.value)
                        else:
                            prom_metric.set(metric_value.value)
                
            except Exception as e:
                self.logger.error("Failed to update metric", 
                                metric=metric_value.name, error=str(e))
    
    def get_metrics_output(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


class MetricsManager:
    """
    Central metrics management system.
    
    Coordinates metric collection from multiple collectors and handles
    export in various formats.
    """
    
    def __init__(self, collection_interval: float = 15.0):
        """
        Initialize metrics manager.
        
        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.collectors: Dict[str, MetricCollector] = {}
        self.exporters: List[Any] = []
        
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Metric storage
        self._metric_history: Dict[str, List[MetricValue]] = {}
        self._max_history_size = 1000  # Keep last 1000 values per metric
        
        self.logger = structlog.get_logger("metrics_manager")
        
        # Initialize Prometheus exporter by default
        self.prometheus_exporter = PrometheusExporter()
        self.exporters.append(self.prometheus_exporter)
    
    def add_collector(self, collector: MetricCollector):
        """Add a metric collector."""
        self.collectors[collector.collector_name] = collector
        self.logger.info("Metric collector added", collector=collector.collector_name)
    
    def add_exporter(self, exporter: Any):
        """Add a metric exporter."""
        self.exporters.append(exporter)
        self.logger.info("Metric exporter added", exporter=type(exporter).__name__)
    
    async def start(self):
        """Start metrics collection."""
        self.logger.info("Starting metrics collection", interval=self.collection_interval)
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop(self):
        """Stop metrics collection."""
        self.logger.info("Stopping metrics collection")
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
    
    async def collect_all_metrics(self) -> Dict[str, List[MetricValue]]:
        """Collect metrics from all collectors."""
        all_metrics = {}
        
        for name, collector in self.collectors.items():
            try:
                metrics = await collector.collect_metrics()
                all_metrics[name] = metrics
                
                # Store in history
                for metric in metrics:
                    if metric.name not in self._metric_history:
                        self._metric_history[metric.name] = []
                    
                    history = self._metric_history[metric.name]
                    history.append(metric)
                    
                    # Trim history if too long
                    if len(history) > self._max_history_size:
                        history.pop(0)
                
            except Exception as e:
                self.logger.error("Collector failed", collector=name, error=str(e))
                all_metrics[name] = []
        
        return all_metrics
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return self.prometheus_exporter.get_metrics_output()
    
    def get_metric_history(self, metric_name: str, limit: Optional[int] = None) -> List[MetricValue]:
        """Get historical values for a metric."""
        history = self._metric_history.get(metric_name, [])
        
        if limit:
            return history[-limit:]
        return history
    
    def get_latest_metrics(self) -> Dict[str, MetricValue]:
        """Get latest value for each metric."""
        latest = {}
        
        for metric_name, history in self._metric_history.items():
            if history:
                latest[metric_name] = history[-1]
        
        return latest
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics system."""
        return {
            "collectors_count": len(self.collectors),
            "exporters_count": len(self.exporters),
            "metrics_count": len(self._metric_history),
            "collection_interval": self.collection_interval,
            "is_running": self._running,
            "collectors": list(self.collectors.keys())
        }
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                # Collect metrics from all collectors
                all_metrics = await self.collect_all_metrics()
                
                # Flatten metrics for exporters
                flat_metrics = []
                for collector_metrics in all_metrics.values():
                    flat_metrics.extend(collector_metrics)
                
                # Update exporters
                for exporter in self.exporters:
                    try:
                        if hasattr(exporter, 'update_metrics'):
                            exporter.update_metrics(flat_metrics)
                    except Exception as e:
                        self.logger.error("Exporter update failed", 
                                        exporter=type(exporter).__name__, error=str(e))
                
                # Log collection summary
                if flat_metrics:
                    self.logger.debug("Metrics collected", 
                                    count=len(flat_metrics),
                                    collectors=len(all_metrics))
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(self.collection_interval)


# Global metrics manager instance
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get global metrics manager instance."""
    global _metrics_manager
    
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    
    return _metrics_manager


async def initialize_metrics(
    network_manager=None,
    db_manager=None,
    collection_interval: float = 15.0
) -> MetricsManager:
    """Initialize metrics system with standard collectors."""
    manager = get_metrics_manager()
    manager.collection_interval = collection_interval
    
    # Add standard collectors
    if network_manager:
        manager.add_collector(NetworkMetricsCollector(network_manager))
    
    if db_manager:
        manager.add_collector(DatabaseMetricsCollector(db_manager))
    
    manager.add_collector(SystemMetricsCollector())
    
    await manager.start()
    return manager