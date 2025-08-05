"""Advanced error handling and recovery system for Agent Mesh.

This module provides comprehensive error handling, retry mechanisms,
circuit breakers, and recovery strategies for resilient operation.
"""

import asyncio
import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type, Union
from uuid import UUID, uuid4

import structlog


class ErrorSeverity(Enum):
    """Error severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    
    NETWORK = "network"
    SECURITY = "security"
    CONSENSUS = "consensus"
    TASK_EXECUTION = "task_execution"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    
    RETRY = "retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAK = "circuit_break"
    FAILOVER = "failover"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    IMMEDIATE_FAIL = "immediate_fail"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    error_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    node_id: Optional[UUID] = None
    operation: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.UNKNOWN
    error_type: str = ""
    error_message: str = ""
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    is_open: bool = False
    is_half_open: bool = False
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    success_threshold: int = 3  # successes needed to close


class AgentMeshError(Exception):
    """Base exception for Agent Mesh system."""
    
    def __init__(
        self, 
        message: str, 
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.metadata = metadata or {}
        self.timestamp = time.time()


class NetworkError(AgentMeshError):
    """Network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class SecurityError(AgentMeshError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.SECURITY, 
                        severity=ErrorSeverity.HIGH, **kwargs)


class ConsensusError(AgentMeshError):
    """Consensus-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONSENSUS, **kwargs)


class TaskExecutionError(AgentMeshError):
    """Task execution errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TASK_EXECUTION, **kwargs)


class ValidationError(AgentMeshError):
    """Input validation errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ResourceError(AgentMeshError):
    """Resource-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, **kwargs)


class TimeoutError(AgentMeshError):
    """Timeout errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.name = name
        self.state = CircuitBreakerState(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold
        )
        self.logger = structlog.get_logger("circuit_breaker", name=name)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        current_time = time.time()
        
        # Check if circuit is open
        if self.state.is_open:
            if current_time - self.state.last_failure_time < self.state.recovery_timeout:
                raise AgentMeshError(
                    f"Circuit breaker '{self.name}' is open",
                    category=ErrorCategory.RESOURCE,
                    severity=ErrorSeverity.HIGH
                )
            else:
                # Try to move to half-open state
                self.state.is_half_open = True
                self.state.is_open = False
                self.logger.info("Circuit breaker moved to half-open state")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            await self._record_success()
            return result
            
        except Exception as e:
            await self._record_failure(e)
            raise
    
    async def _record_success(self) -> None:
        """Record a successful operation."""
        self.state.success_count += 1
        self.state.last_success_time = time.time()
        
        if self.state.is_half_open:
            if self.state.success_count >= self.state.success_threshold:
                self.state.is_half_open = False
                self.state.is_open = False
                self.state.failure_count = 0
                self.state.success_count = 0
                self.logger.info("Circuit breaker closed after successful recovery")
    
    async def _record_failure(self, error: Exception) -> None:
        """Record a failed operation."""
        self.state.failure_count += 1
        self.state.last_failure_time = time.time()
        
        if self.state.failure_count >= self.state.failure_threshold:
            self.state.is_open = True
            self.state.is_half_open = False
            self.logger.warning("Circuit breaker opened due to failures",
                              failure_count=self.state.failure_count,
                              error=str(error))


class RetryManager:
    """Advanced retry manager with multiple strategies."""
    
    def __init__(self):
        self.logger = structlog.get_logger("retry_manager")
    
    async def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        *args,
        **kwargs
    ) -> Any:
        """Retry function with exponential backoff."""
        import random
        
        retryable_exceptions = retryable_exceptions or [Exception]
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                if attempt == max_retries:
                    self.logger.error("All retry attempts exhausted",
                                    function=func.__name__,
                                    attempts=attempt + 1,
                                    error=str(e))
                    raise
                
                if not any(isinstance(e, exc_type) for exc_type in retryable_exceptions):
                    self.logger.error("Non-retryable exception encountered",
                                    function=func.__name__,
                                    error_type=type(e).__name__,
                                    error=str(e))
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_factor ** attempt), max_delay)
                
                if jitter:
                    delay = delay * (0.5 + random.random() * 0.5)
                
                self.logger.warning("Retrying after failure",
                                  function=func.__name__,
                                  attempt=attempt + 1,
                                  delay=delay,
                                  error=str(e))
                
                await asyncio.sleep(delay)
    
    async def retry_with_circuit_breaker(
        self,
        func: Callable,
        circuit_breaker: CircuitBreaker,
        max_retries: int = 3,
        base_delay: float = 1.0,
        *args,
        **kwargs
    ) -> Any:
        """Retry function with circuit breaker protection."""
        for attempt in range(max_retries + 1):
            try:
                return await circuit_breaker.call(func, *args, **kwargs)
                
            except Exception as e:
                if attempt == max_retries:
                    raise
                
                delay = base_delay * (2 ** attempt)
                self.logger.warning("Retrying with circuit breaker protection",
                                  function=func.__name__,
                                  attempt=attempt + 1,
                                  delay=delay,
                                  error=str(e))
                
                await asyncio.sleep(delay)


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, node_id: Optional[UUID] = None):
        self.node_id = node_id
        self.logger = structlog.get_logger("error_handler", 
                                         node_id=str(node_id) if node_id else "unknown")
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}
        
        # Recovery components
        self.retry_manager = RetryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error handlers by category
        self.error_handlers: Dict[ErrorCategory, Callable] = {
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.SECURITY: self._handle_security_error,
            ErrorCategory.CONSENSUS: self._handle_consensus_error,
            ErrorCategory.TASK_EXECUTION: self._handle_task_error,
            ErrorCategory.RESOURCE: self._handle_resource_error,
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.TIMEOUT: self._handle_timeout_error,
        }
        
        # Configuration
        self.max_error_history = 1000
        self.error_rate_threshold = 0.1  # 10% error rate
        self.error_burst_threshold = 10   # 10 errors in short time
    
    async def handle_error(
        self, 
        error: Exception, 
        context: Optional[ErrorContext] = None
    ) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy."""
        if context is None:
            context = self._create_error_context(error)
        
        # Log error
        self.logger.error("Error occurred",
                         error_id=str(context.error_id),
                         category=context.category.value,
                         severity=context.severity.value,
                         error=str(error))
        
        # Record error
        await self._record_error(context, error)
        
        # Determine recovery strategy
        strategy = await self._determine_recovery_strategy(context, error)
        
        # Execute recovery
        return await self._execute_recovery(strategy, context, error)
    
    def get_circuit_breaker(
        self, 
        name: str, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name, failure_threshold, recovery_timeout
            )
        return self.circuit_breakers[name]
    
    def with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        exponential_factor: float = 2.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """Decorator for automatic retry with exponential backoff."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.retry_manager.retry_with_backoff(
                    func, max_retries, base_delay, 60.0, exponential_factor,
                    True, retryable_exceptions, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def with_circuit_breaker(self, breaker_name: str):
        """Decorator for circuit breaker protection."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                circuit_breaker = self.get_circuit_breaker(breaker_name)
                return await circuit_breaker.call(func, *args, **kwargs)
            return wrapper
        return decorator
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and system health metrics."""
        current_time = time.time()
        recent_errors = [
            e for e in self.error_history 
            if current_time - e.timestamp < 3600  # Last hour
        ]
        
        category_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_rate": len(recent_errors) / max(1, len(self.error_history)),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "circuit_breaker_states": {
                name: {
                    "is_open": cb.state.is_open,
                    "is_half_open": cb.state.is_half_open,
                    "failure_count": cb.state.failure_count,
                    "success_count": cb.state.success_count
                }
                for name, cb in self.circuit_breakers.items()
            }
        }
    
    # Private methods
    
    def _create_error_context(self, error: Exception) -> ErrorContext:
        """Create error context from exception."""
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        if isinstance(error, AgentMeshError):
            category = error.category
            severity = error.severity
        elif isinstance(error, (ConnectionError, OSError)):
            category = ErrorCategory.NETWORK
        elif isinstance(error, TimeoutError):
            category = ErrorCategory.TIMEOUT
        elif isinstance(error, (ValueError, TypeError)):
            category = ErrorCategory.VALIDATION
        
        return ErrorContext(
            node_id=self.node_id,
            category=category,
            severity=severity,
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    async def _record_error(self, context: ErrorContext, error: Exception) -> None:
        """Record error in history and update counters."""
        self.error_history.append(context)
        
        # Maintain history size
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history//2:]
        
        # Update error counts
        error_key = f"{context.category.value}:{context.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    async def _determine_recovery_strategy(
        self, 
        context: ErrorContext, 
        error: Exception
    ) -> RecoveryStrategy:
        """Determine appropriate recovery strategy."""
        # High severity errors - immediate fail
        if context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.IMMEDIATE_FAIL
        
        # Security errors - immediate fail
        if context.category == ErrorCategory.SECURITY:
            return RecoveryStrategy.IMMEDIATE_FAIL
        
        # Network errors - retry with backoff
        if context.category == ErrorCategory.NETWORK:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        # Timeout errors - retry
        if context.category == ErrorCategory.TIMEOUT:
            return RecoveryStrategy.RETRY
        
        # Resource errors - circuit break
        if context.category == ErrorCategory.RESOURCE:
            return RecoveryStrategy.CIRCUIT_BREAK
        
        # Task execution errors - retry
        if context.category == ErrorCategory.TASK_EXECUTION:
            return RecoveryStrategy.RETRY
        
        # Default strategy
        return RecoveryStrategy.RETRY
    
    async def _execute_recovery(
        self, 
        strategy: RecoveryStrategy, 
        context: ErrorContext, 
        error: Exception
    ) -> Optional[Any]:
        """Execute recovery strategy."""
        self.logger.info("Executing recovery strategy",
                        strategy=strategy.value,
                        error_id=str(context.error_id))
        
        if strategy == RecoveryStrategy.IMMEDIATE_FAIL:
            raise error
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation(context, error)
        
        # For other strategies, the calling code should handle retry logic
        return None
    
    async def _graceful_degradation(self, context: ErrorContext, error: Exception) -> Any:
        """Implement graceful degradation."""
        self.logger.info("Applying graceful degradation",
                        error_id=str(context.error_id))
        
        # Return a default or cached result
        return {"status": "degraded", "error": str(error)}
    
    # Error-specific handlers
    
    async def _handle_network_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle network-specific errors."""
        return await self._graceful_degradation(context, error)
    
    async def _handle_security_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle security-specific errors."""
        self.logger.critical("Security error detected",
                           error_id=str(context.error_id),
                           error=str(error))
        raise error
    
    async def _handle_consensus_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle consensus-specific errors."""
        return await self._graceful_degradation(context, error)
    
    async def _handle_task_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle task execution errors."""
        return await self._graceful_degradation(context, error)
    
    async def _handle_resource_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle resource-related errors."""
        return await self._graceful_degradation(context, error)
    
    async def _handle_validation_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle validation errors."""
        raise error  # Validation errors should not be retried
    
    async def _handle_timeout_error(self, context: ErrorContext, error: Exception) -> Any:
        """Handle timeout errors."""
        return await self._graceful_degradation(context, error)


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler(node_id: Optional[UUID] = None) -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(node_id)
    return _global_error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler


# Convenience functions
async def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> Optional[Any]:
    """Handle error using global error handler."""
    handler = get_error_handler()
    return await handler.handle_error(error, context)


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for automatic retry using global error handler."""
    handler = get_error_handler()
    return handler.with_retry(max_retries, base_delay)


def with_circuit_breaker(breaker_name: str):
    """Decorator for circuit breaker protection using global error handler."""
    handler = get_error_handler()
    return handler.with_circuit_breaker(breaker_name)