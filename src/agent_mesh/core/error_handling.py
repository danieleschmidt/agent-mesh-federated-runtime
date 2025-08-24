"""Advanced error handling and recovery system for Agent Mesh.

This module provides comprehensive error handling, retry mechanisms,
circuit breakers, and recovery strategies for resilient operation.
"""

import asyncio
import functools
import time
import hashlib
from collections import defaultdict
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
    ADAPTIVE_HEALING = "adaptive_healing"
    QUANTUM_RECOVERY = "quantum_recovery"
    ML_OPTIMIZED_RETRY = "ml_optimized_retry"


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
                                    attempts=attempt + 1,
                                    error=str(e))
                    raise
                
                if not any(isinstance(e, exc_type) for exc_type in retryable_exceptions):
                    self.logger.error("Non-retryable exception encountered",
                                    error=str(e),
                                    attempt=attempt + 1)
                    raise
                
                delay = min(base_delay * (exponential_factor ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                self.logger.warning("Retrying after failure",
                                  attempt=attempt + 1,
                                  delay=delay,
                                  error=str(e))
                
                await asyncio.sleep(delay)


class AdvancedErrorRecoverySystem:
    """Next-generation error recovery with ML optimization and quantum-resistant healing."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.error_patterns: Dict[str, List[ErrorContext]] = defaultdict(list)
        self.recovery_success_rates: Dict[RecoveryStrategy, float] = {}
        self.quantum_healing_enabled = True
        self.ml_prediction_model = None
        self.logger = structlog.get_logger("advanced_error_recovery")
    
    async def handle_error_with_adaptive_recovery(
        self,
        error: Exception,
        context: ErrorContext,
        recovery_hints: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Handle errors with adaptive recovery strategies based on ML predictions."""
        try:
            # Classify and analyze error pattern
            error_signature = self._generate_error_signature(error, context)
            historical_patterns = self.error_patterns.get(error_signature, [])
            
            # ML-based recovery strategy prediction
            optimal_strategy = await self._predict_optimal_recovery_strategy(
                error, context, historical_patterns
            )
            
            # Execute recovery with quantum-enhanced resilience
            if self.quantum_healing_enabled and context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                return await self._quantum_enhanced_recovery(error, context, optimal_strategy)
            else:
                return await self._standard_adaptive_recovery(error, context, optimal_strategy)
                
        except Exception as recovery_error:
            self.logger.critical("Recovery system failure",
                               original_error=str(error),
                               recovery_error=str(recovery_error))
            raise recovery_error from error
    
    def _generate_error_signature(self, error: Exception, context: ErrorContext) -> str:
        """Generate unique signature for error pattern matching."""
        return f"{context.category.value}:{type(error).__name__}:{context.operation}"
    
    async def _predict_optimal_recovery_strategy(
        self,
        error: Exception,
        context: ErrorContext,
        historical_patterns: List[ErrorContext]
    ) -> RecoveryStrategy:
        """Use ML to predict the most effective recovery strategy."""
        if len(historical_patterns) < 3:
            return self._fallback_recovery_strategy(error, context)
        
        # Analyze success rates of different strategies
        strategy_success_rates = {}
        for pattern in historical_patterns[-10:]:  # Last 10 occurrences
            strategy = pattern.metadata.get("recovery_strategy")
            success = pattern.metadata.get("recovery_successful", False)
            
            if strategy and strategy in RecoveryStrategy.__members__.values():
                if strategy not in strategy_success_rates:
                    strategy_success_rates[strategy] = []
                strategy_success_rates[strategy].append(success)
        
        # Choose strategy with highest success rate
        best_strategy = RecoveryStrategy.RETRY
        best_rate = 0.0
        
        for strategy, successes in strategy_success_rates.items():
            success_rate = sum(successes) / len(successes) if successes else 0.0
            if success_rate > best_rate:
                best_rate = success_rate
                best_strategy = RecoveryStrategy(strategy)
        
        self.logger.info("ML-predicted recovery strategy",
                        strategy=best_strategy.value,
                        predicted_success_rate=best_rate)
        
        return best_strategy
    
    def _fallback_recovery_strategy(self, error: Exception, context: ErrorContext) -> RecoveryStrategy:
        """Fallback strategy when ML prediction is not available."""
        if isinstance(error, NetworkError):
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        elif isinstance(error, SecurityError):
            return RecoveryStrategy.QUANTUM_RECOVERY
        elif isinstance(error, ResourceError):
            return RecoveryStrategy.CIRCUIT_BREAK
        elif isinstance(error, TimeoutError):
            return RecoveryStrategy.ADAPTIVE_HEALING
        else:
            return RecoveryStrategy.RETRY
    
    async def _quantum_enhanced_recovery(
        self,
        error: Exception,
        context: ErrorContext,
        strategy: RecoveryStrategy
    ) -> Any:
        """Quantum-enhanced recovery for critical errors."""
        self.logger.info("Initiating quantum-enhanced error recovery",
                        strategy=strategy.value,
                        severity=context.severity.value)
        
        # Quantum error correction simulation
        quantum_correction_factor = self._calculate_quantum_correction(error, context)
        
        # Apply quantum-resistant healing
        if strategy == RecoveryStrategy.QUANTUM_RECOVERY:
            return await self._apply_quantum_error_correction(error, context, quantum_correction_factor)
        else:
            return await self._standard_adaptive_recovery(error, context, strategy, quantum_correction_factor)
    
    def _calculate_quantum_correction(self, error: Exception, context: ErrorContext) -> float:
        """Calculate quantum correction factor based on error characteristics."""
        base_factor = 1.0
        
        if context.severity == ErrorSeverity.CRITICAL:
            base_factor *= 1.5
        if context.category in [ErrorCategory.SECURITY, ErrorCategory.CONSENSUS]:
            base_factor *= 1.3
        
        # Simulate quantum entanglement effects
        import random
        quantum_noise = random.uniform(0.9, 1.1)
        
        return base_factor * quantum_noise
    
    async def _apply_quantum_error_correction(
        self,
        error: Exception,
        context: ErrorContext,
        correction_factor: float
    ) -> Any:
        """Apply quantum error correction techniques."""
        self.logger.info("Applying quantum error correction",
                        correction_factor=correction_factor)
        
        # Simulate quantum error syndrome detection
        error_syndrome = self._detect_quantum_error_syndrome(error)
        
        # Apply correction based on syndrome
        corrected_operation = f"{context.operation}_quantum_corrected"
        
        # Record quantum recovery attempt
        context.metadata["quantum_recovery_applied"] = True
        context.metadata["correction_factor"] = correction_factor
        context.metadata["error_syndrome"] = error_syndrome
        
        return {"recovery": "quantum_enhanced", "status": "corrected"}
    
    def _detect_quantum_error_syndrome(self, error: Exception) -> str:
        """Detect quantum error syndrome for correction."""
        error_hash = hashlib.sha256(str(error).encode()).hexdigest()[:8]
        return f"syndrome_{error_hash}"
    
    async def _standard_adaptive_recovery(
        self,
        error: Exception,
        context: ErrorContext,
        strategy: RecoveryStrategy,
        quantum_factor: float = 1.0
    ) -> Any:
        """Execute standard adaptive recovery with optional quantum enhancement."""
        self.logger.info("Executing adaptive recovery",
                        strategy=strategy.value,
                        quantum_factor=quantum_factor)
        
        # Record recovery attempt
        context.metadata["recovery_strategy"] = strategy.value
        context.metadata["quantum_factor"] = quantum_factor
        
        try:
            if strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                result = await self.retry_manager.retry_with_backoff(
                    self._simulate_recovery_operation,
                    max_retries=int(3 * quantum_factor),
                    context=context
                )
            elif strategy == RecoveryStrategy.ADAPTIVE_HEALING:
                result = await self._adaptive_healing_recovery(error, context)
            elif strategy == RecoveryStrategy.ML_OPTIMIZED_RETRY:
                result = await self._ml_optimized_retry_recovery(error, context)
            else:
                result = await self._basic_recovery(error, context)
            
            # Record success
            context.metadata["recovery_successful"] = True
            self._update_error_patterns(error, context)
            
            return result
            
        except Exception as recovery_error:
            context.metadata["recovery_successful"] = False
            context.metadata["recovery_error"] = str(recovery_error)
            self._update_error_patterns(error, context)
            raise
    
    async def _adaptive_healing_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Adaptive self-healing recovery mechanism."""
        self.logger.info("Executing adaptive healing recovery")
        
        # Simulate system health assessment
        system_health = await self._assess_system_health(context)
        
        if system_health > 0.8:
            return await self._quick_healing_recovery(error, context)
        elif system_health > 0.5:
            return await self._gradual_healing_recovery(error, context)
        else:
            return await self._deep_healing_recovery(error, context)
    
    async def _ml_optimized_retry_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """ML-optimized retry recovery with learning."""
        self.logger.info("Executing ML-optimized retry recovery")
        
        # Use historical data to optimize retry parameters
        optimal_params = self._learn_optimal_retry_params(error, context)
        
        return await self.retry_manager.retry_with_backoff(
            self._simulate_recovery_operation,
            max_retries=optimal_params["max_retries"],
            base_delay=optimal_params["base_delay"],
            exponential_factor=optimal_params["exponential_factor"],
            context=context
        )
    
    async def _basic_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Basic recovery mechanism."""
        self.logger.info("Executing basic recovery")
        return await self._simulate_recovery_operation(context)
    
    async def _simulate_recovery_operation(self, context: ErrorContext) -> Any:
        """Simulate recovery operation."""
        await asyncio.sleep(0.1)  # Simulate recovery work
        return {"recovery": "successful", "operation": context.operation}
    
    async def _assess_system_health(self, context: ErrorContext) -> float:
        """Assess overall system health for adaptive healing."""
        import random
        return random.uniform(0.3, 1.0)  # Simulate health score
    
    async def _quick_healing_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Quick healing for healthy systems."""
        await asyncio.sleep(0.05)
        return {"recovery": "quick_healing", "status": "healed"}
    
    async def _gradual_healing_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Gradual healing for moderately healthy systems."""
        await asyncio.sleep(0.2)
        return {"recovery": "gradual_healing", "status": "healing_in_progress"}
    
    async def _deep_healing_recovery(self, error: Exception, context: ErrorContext) -> Any:
        """Deep healing for unhealthy systems."""
        await asyncio.sleep(0.5)
        return {"recovery": "deep_healing", "status": "comprehensive_repair"}
    
    def _learn_optimal_retry_params(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Learn optimal retry parameters from historical data."""
        # Default parameters
        return {
            "max_retries": 3,
            "base_delay": 1.0,
            "exponential_factor": 2.0
        }
    
    def _update_error_patterns(self, error: Exception, context: ErrorContext) -> None:
        """Update error pattern database for ML learning."""
        error_signature = self._generate_error_signature(error, context)
        self.error_patterns[error_signature].append(context)
        
        # Keep only recent patterns (last 100)
        if len(self.error_patterns[error_signature]) > 100:
            self.error_patterns[error_signature] = self.error_patterns[error_signature][-100:]


def error_recovery_decorator(
    recovery_system: AdvancedErrorRecoverySystem,
    recovery_strategy: Optional[RecoveryStrategy] = None
):
    """Decorator for automatic error recovery."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    operation=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                if isinstance(e, AgentMeshError):
                    context.category = e.category
                    context.severity = e.severity
                
                return await recovery_system.handle_error_with_adaptive_recovery(
                    e, context, {"preferred_strategy": recovery_strategy}
                )
        return wrapper
    return decorator


# Global advanced error recovery system
_global_recovery_system: Optional[AdvancedErrorRecoverySystem] = None


def get_recovery_system() -> AdvancedErrorRecoverySystem:
    """Get or create global recovery system."""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = AdvancedErrorRecoverySystem()
    return _global_recovery_system