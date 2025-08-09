"""Health monitoring and circuit breaker patterns for Agent Mesh.

Provides comprehensive health monitoring, circuit breakers, retry mechanisms,
and self-healing capabilities for robust distributed operations.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from uuid import UUID

import structlog


logger = structlog.get_logger("health")


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"       # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        """Get status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)
    check_count: int = 0
    consecutive_failures: int = 0
    
    def add_metric(self, name: str, value: float, warning_threshold: float, critical_threshold: float, unit: str = ""):
        """Add a health metric."""
        metric = HealthMetric(name, value, warning_threshold, critical_threshold, unit)
        self.metrics.append(metric)
        return metric
    
    def add_error(self, error: str):
        """Add an error to the health status."""
        self.errors.append(f"{time.time()}: {error}")
        if len(self.errors) > 10:  # Keep only last 10 errors
            self.errors.pop(0)
    
    def get_worst_status(self) -> HealthStatus:
        """Get the worst status among all metrics."""
        if self.errors:
            return HealthStatus.UNHEALTHY
        
        if not self.metrics:
            return self.status
        
        statuses = [metric.status for metric in self.metrics]
        
        # Return worst status
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


class HealthChecker(ABC):
    """Abstract health checker interface."""
    
    @abstractmethod
    async def check_health(self) -> ComponentHealth:
        """Check component health."""
        pass
    
    @property
    @abstractmethod
    def component_name(self) -> str:
        """Get component name."""
        pass


class NetworkHealthChecker(HealthChecker):
    """Health checker for network components."""
    
    def __init__(self, network_manager):
        self.network_manager = network_manager
        self.logger = structlog.get_logger("network_health")
    
    @property
    def component_name(self) -> str:
        return "network"
    
    async def check_health(self) -> ComponentHealth:
        """Check network health."""
        health = ComponentHealth("network", HealthStatus.HEALTHY)
        
        try:
            # Check network statistics
            stats = await self.network_manager.get_statistics()
            
            # Connection metrics
            active_connections = stats.get("connections_active", 0)
            health.add_metric("active_connections", active_connections, 80, 100, "connections")
            
            # Latency metrics
            avg_latency = stats.get("avg_latency_ms", 0)
            health.add_metric("avg_latency", avg_latency, 100, 500, "ms")
            
            # Message rate metrics
            uptime = stats.get("uptime_seconds", 1)
            message_rate = stats.get("messages_sent", 0) / uptime
            health.add_metric("message_rate", message_rate, 1000, 5000, "msg/s")
            
            # Check peer connectivity
            peers = await self.network_manager.get_connected_peers()
            connected_peers = len(peers)
            health.add_metric("connected_peers", connected_peers, 1, 0, "peers")
            
            # Test connectivity with random peer
            if peers:
                test_peer = peers[0]
                try:
                    is_responsive = await self.network_manager.is_peer_responsive(test_peer.peer_id)
                    if not is_responsive:
                        health.add_error("Peer connectivity test failed")
                except Exception as e:
                    health.add_error(f"Peer connectivity test error: {e}")
            
            health.status = health.get_worst_status()
            health.last_check = time.time()
            health.check_count += 1
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.add_error(f"Health check failed: {e}")
            health.consecutive_failures += 1
            self.logger.error("Network health check failed", error=str(e))
        
        return health


class DatabaseHealthChecker(HealthChecker):
    """Health checker for database components."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = structlog.get_logger("database_health")
    
    @property
    def component_name(self) -> str:
        return "database"
    
    async def check_health(self) -> ComponentHealth:
        """Check database health."""
        health = ComponentHealth("database", HealthStatus.HEALTHY)
        
        try:
            # Basic connectivity test
            is_healthy = await self.db_manager.health_check()
            
            if not is_healthy:
                health.status = HealthStatus.CRITICAL
                health.add_error("Database connectivity failed")
                health.consecutive_failures += 1
            else:
                health.status = HealthStatus.HEALTHY
                health.consecutive_failures = 0
            
            # Connection pool metrics
            conn_info = self.db_manager.get_connection_info()
            pool_size = conn_info.get("pool_size", 0)
            health.add_metric("pool_size", pool_size, pool_size * 0.8, pool_size * 0.95, "connections")
            
            health.last_check = time.time()
            health.check_count += 1
            
        except Exception as e:
            health.status = HealthStatus.CRITICAL
            health.add_error(f"Health check failed: {e}")
            health.consecutive_failures += 1
            self.logger.error("Database health check failed", error=str(e))
        
        return health


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascade failures by failing fast when error rates are high.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            recovery_timeout: Time to wait before testing recovery
            success_threshold: Successes needed to close circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = structlog.get_logger("circuit_breaker", name=name)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise CircuitBreakerOpenException(f"Circuit breaker {self.name} is OPEN")
            else:
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker entering HALF_OPEN state")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._close()
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
            
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_failure(self):
        """Record a failure and update state."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self._open()
    
    def _open(self):
        """Open the circuit breaker."""
        self.state = CircuitBreakerState.OPEN
        self.logger.warning("Circuit breaker OPENED", failure_count=self.failure_count)
    
    def _close(self):
        """Close the circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info("Circuit breaker CLOSED")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }


class RetryManager:
    """
    Retry mechanism with exponential backoff and jitter.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """
        Initialize retry manager.
        
        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Exponential backoff base
            jitter: Whether to add jitter to delay
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = structlog.get_logger("retry_manager")
    
    async def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info("Function succeeded on retry", attempt=attempt + 1)
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt, don't delay
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter if enabled
                if self.jitter:
                    import random
                    delay = delay * (0.5 + random.random() * 0.5)
                
                self.logger.warning("Function failed, retrying", 
                                  attempt=attempt + 1, 
                                  delay=delay, 
                                  error=str(e))
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        self.logger.error("All retry attempts failed", attempts=self.max_attempts)
        raise last_exception


class HealthMonitor:
    """
    Comprehensive health monitoring system.
    
    Coordinates multiple health checkers and provides system-wide health status.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[Dict[str, ComponentHealth]], None]] = []
        
        self.logger = structlog.get_logger("health_monitor")
    
    def add_health_checker(self, checker: HealthChecker):
        """Add a health checker."""
        self.health_checkers[checker.component_name] = checker
        self.logger.info("Health checker added", component=checker.component_name)
    
    def add_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Add a circuit breaker."""
        breaker = CircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = breaker
        self.logger.info("Circuit breaker added", name=name)
        return breaker
    
    def add_health_callback(self, callback: Callable[[Dict[str, ComponentHealth]], None]):
        """Add callback for health status changes."""
        self._callbacks.append(callback)
    
    async def start(self):
        """Start health monitoring."""
        self.logger.info("Starting health monitoring", interval=self.check_interval)
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop health monitoring."""
        self.logger.info("Stopping health monitoring")
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def check_all_health(self) -> Dict[str, ComponentHealth]:
        """Check health of all components."""
        health_results = {}
        
        for name, checker in self.health_checkers.items():
            try:
                health = await checker.check_health()
                health_results[name] = health
                self.component_health[name] = health
                
            except Exception as e:
                self.logger.error("Health checker failed", component=name, error=str(e))
                
                # Create error health status
                error_health = ComponentHealth(name, HealthStatus.CRITICAL)
                error_health.add_error(f"Health checker exception: {e}")
                health_results[name] = error_health
                self.component_health[name] = error_health
        
        return health_results
    
    def get_system_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.component_health:
            return HealthStatus.UNKNOWN
        
        statuses = [health.get_worst_status() for health in self.component_health.values()]
        
        # Return worst status
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        return {
            "system_status": self.get_system_health().value,
            "components": {
                name: {
                    "status": health.get_worst_status().value,
                    "metrics_count": len(health.metrics),
                    "errors_count": len(health.errors),
                    "last_check": health.last_check,
                    "consecutive_failures": health.consecutive_failures
                }
                for name, health in self.component_health.items()
            },
            "circuit_breakers": {
                name: breaker.get_status()
                for name, breaker in self.circuit_breakers.items()
            },
            "timestamp": time.time()
        }
    
    async def _monitor_loop(self):
        """Main health monitoring loop."""
        while self._running:
            try:
                # Check all component health
                health_results = await self.check_all_health()
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(health_results)
                    except Exception as e:
                        self.logger.error("Health callback failed", error=str(e))
                
                # Log system status
                system_status = self.get_system_health()
                if system_status != HealthStatus.HEALTHY:
                    self.logger.warning("System health degraded", status=system_status.value)
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health monitor loop error", error=str(e))
                await asyncio.sleep(self.check_interval)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    return _health_monitor


async def initialize_health_monitoring(network_manager=None, db_manager=None) -> HealthMonitor:
    """Initialize health monitoring with standard checkers."""
    monitor = get_health_monitor()
    
    # Add standard health checkers
    if network_manager:
        monitor.add_health_checker(NetworkHealthChecker(network_manager))
    
    if db_manager:
        monitor.add_health_checker(DatabaseHealthChecker(db_manager))
    
    # Add standard circuit breakers
    monitor.add_circuit_breaker("network_operations", failure_threshold=3, recovery_timeout=30.0)
    monitor.add_circuit_breaker("database_operations", failure_threshold=5, recovery_timeout=60.0)
    monitor.add_circuit_breaker("consensus_operations", failure_threshold=3, recovery_timeout=45.0)
    
    await monitor.start()
    return monitor