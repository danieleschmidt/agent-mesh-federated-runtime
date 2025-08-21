"""Continuous Learning Coordinator for autonomous system evolution."""

import asyncio
import logging
import time
import json
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

logger = logging.getLogger(__name__)

class LearningType(Enum):
    """Types of learning activities."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    PATTERN_RECOGNITION = "pattern_recognition"
    PREDICTIVE_MODELING = "predictive_modeling"
    ADAPTATION = "adaptation"
    META_LEARNING = "meta_learning"

class LearningPhase(Enum):
    """Learning phases."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    ADAPTATION = "adaptation"
    VALIDATION = "validation"

@dataclass
class LearningEvent:
    """Learning event data."""
    event_id: str
    learning_type: LearningType
    phase: LearningPhase
    data: Dict[str, Any]
    insights: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class Pattern:
    """Discovered pattern."""
    pattern_id: str
    pattern_type: str
    description: str
    strength: float  # 0.0 to 1.0
    occurrences: int
    last_seen: float
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Prediction:
    """System prediction."""
    prediction_id: str
    target_metric: str
    predicted_value: float
    confidence: float
    time_horizon: float  # seconds into future
    actual_value: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

class ContinuousLearningCoordinator:
    """Coordinates continuous learning across all system components."""
    
    def __init__(
        self,
        learning_interval: float = 60.0,
        pattern_threshold: float = 0.7,
        prediction_horizon: float = 300.0  # 5 minutes
    ):
        self.learning_interval = learning_interval
        self.pattern_threshold = pattern_threshold
        self.prediction_horizon = prediction_horizon
        
        # Learning state
        self.current_phase = LearningPhase.EXPLORATION
        self.learning_events: List[LearningEvent] = []
        self.discovered_patterns: Dict[str, Pattern] = {}
        self.predictions: Dict[str, Prediction] = {}
        self.knowledge_base: Dict[str, Any] = {}
        
        # Meta-learning
        self.meta_knowledge: Dict[str, float] = {
            "exploration_effectiveness": 0.5,
            "exploitation_efficiency": 0.5,
            "adaptation_speed": 0.5,
            "prediction_accuracy": 0.5
        }
        
        # Data collection
        self.metric_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, value)
        self.system_events: List[Dict[str, Any]] = []
        
        # Learning algorithms
        self.pattern_detectors: Dict[str, Callable] = {}
        self.predictive_models: Dict[str, Callable] = {}
        
        # State management
        self.is_running = False
        self._learning_task: Optional[asyncio.Task] = None
        
        # Initialize learning components
        self._initialize_pattern_detectors()
        self._initialize_predictive_models()
    
    def _initialize_pattern_detectors(self):
        """Initialize pattern detection algorithms."""
        
        def detect_cyclic_patterns(metric_name: str, data: List[Tuple[float, float]]) -> List[Pattern]:
            """Detect cyclic patterns in metrics."""
            patterns = []
            
            if len(data) < 10:
                return patterns
            
            values = [v for _, v in data[-100:]]  # Last 100 values
            timestamps = [t for t, _ in data[-100:]]
            
            # Simple peak detection for daily cycles
            peaks = []
            for i in range(1, len(values) - 1):
                if values[i] > values[i-1] and values[i] > values[i+1]:
                    peaks.append((timestamps[i], values[i]))
            
            # Check for regular intervals between peaks
            if len(peaks) >= 3:
                intervals = []
                for i in range(1, len(peaks)):
                    interval = peaks[i][0] - peaks[i-1][0]
                    intervals.append(interval)
                
                if intervals:
                    avg_interval = statistics.mean(intervals)
                    std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
                    
                    # If intervals are consistent (low standard deviation)
                    if std_interval / avg_interval < 0.2:  # Within 20%
                        pattern = Pattern(
                            pattern_id=f"cyclic_{metric_name}_{int(time.time())}",
                            pattern_type="cyclic",
                            description=f"Cyclic pattern in {metric_name} with {avg_interval:.0f}s intervals",
                            strength=1.0 - (std_interval / avg_interval),
                            occurrences=len(peaks),
                            last_seen=time.time(),
                            parameters={"interval": avg_interval, "std_deviation": std_interval}
                        )
                        patterns.append(pattern)
            
            return patterns
        
        def detect_trend_patterns(metric_name: str, data: List[Tuple[float, float]]) -> List[Pattern]:
            """Detect trend patterns in metrics."""
            patterns = []
            
            if len(data) < 5:
                return patterns
            
            values = [v for _, v in data[-20:]]  # Last 20 values
            
            # Calculate slope using linear regression
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * values[i] for i in range(n))
            sum_x2 = sum(i * i for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Determine trend strength
            if abs(slope) > 0.1:  # Significant trend
                trend_type = "increasing" if slope > 0 else "decreasing"
                strength = min(1.0, abs(slope) / max(values))
                
                pattern = Pattern(
                    pattern_id=f"trend_{metric_name}_{int(time.time())}",
                    pattern_type="trend",
                    description=f"{trend_type.capitalize()} trend in {metric_name}",
                    strength=strength,
                    occurrences=1,
                    last_seen=time.time(),
                    parameters={"slope": slope, "trend_type": trend_type}
                )
                patterns.append(pattern)
            
            return patterns
        
        def detect_anomaly_patterns(metric_name: str, data: List[Tuple[float, float]]) -> List[Pattern]:
            """Detect anomaly patterns in metrics."""
            patterns = []
            
            if len(data) < 10:
                return patterns
            
            values = [v for _, v in data[-50:]]  # Last 50 values
            
            if len(values) < 10:
                return patterns
            
            # Calculate statistical thresholds
            mean_val = statistics.mean(values[:-5])  # Exclude recent values
            std_val = statistics.stdev(values[:-5]) if len(values[:-5]) > 1 else 0
            
            # Check recent values for anomalies
            recent_values = values[-5:]
            anomalies = []
            
            for i, value in enumerate(recent_values):
                if std_val > 0:
                    z_score = abs(value - mean_val) / std_val
                    if z_score > 2.0:  # 2 standard deviations
                        anomalies.append((i, value, z_score))
            
            if anomalies:
                max_z_score = max(z for _, _, z in anomalies)
                
                pattern = Pattern(
                    pattern_id=f"anomaly_{metric_name}_{int(time.time())}",
                    pattern_type="anomaly",
                    description=f"Anomalous values detected in {metric_name}",
                    strength=min(1.0, max_z_score / 3.0),  # Normalize to 0-1
                    occurrences=len(anomalies),
                    last_seen=time.time(),
                    parameters={"max_z_score": max_z_score, "anomalies": len(anomalies)}
                )
                patterns.append(pattern)
            
            return patterns
        
        self.pattern_detectors = {
            "cyclic": detect_cyclic_patterns,
            "trend": detect_trend_patterns,
            "anomaly": detect_anomaly_patterns
        }
    
    def _initialize_predictive_models(self):
        """Initialize predictive models."""
        
        def linear_predictor(metric_name: str, data: List[Tuple[float, float]], horizon: float) -> Optional[Prediction]:
            """Simple linear prediction model."""
            if len(data) < 5:
                return None
            
            values = [v for _, v in data[-10:]]  # Last 10 values
            timestamps = [t for t, _ in data[-10:]]
            
            # Calculate linear trend
            n = len(values)
            sum_x = sum(range(n))
            sum_y = sum(values)
            sum_xy = sum(i * values[i] for i in range(n))
            sum_x2 = sum(i * i for i in range(n))
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return None
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict future value
            future_x = n + (horizon / 60.0)  # Assuming 1-minute intervals
            predicted_value = slope * future_x + intercept
            
            # Calculate confidence based on trend consistency
            residuals = [values[i] - (slope * i + intercept) for i in range(n)]
            mse = sum(r * r for r in residuals) / n
            confidence = max(0.1, 1.0 - min(1.0, mse / max(values)))
            
            return Prediction(
                prediction_id=f"linear_{metric_name}_{int(time.time())}",
                target_metric=metric_name,
                predicted_value=predicted_value,
                confidence=confidence,
                time_horizon=horizon
            )
        
        def moving_average_predictor(metric_name: str, data: List[Tuple[float, float]], horizon: float) -> Optional[Prediction]:
            """Moving average prediction model."""
            if len(data) < 3:
                return None
            
            values = [v for _, v in data[-5:]]  # Last 5 values
            predicted_value = statistics.mean(values)
            
            # Confidence based on variance
            variance = statistics.variance(values) if len(values) > 1 else 0
            confidence = max(0.1, 1.0 - min(1.0, variance / (predicted_value * predicted_value + 1)))
            
            return Prediction(
                prediction_id=f"ma_{metric_name}_{int(time.time())}",
                target_metric=metric_name,
                predicted_value=predicted_value,
                confidence=confidence,
                time_horizon=horizon
            )
        
        self.predictive_models = {
            "linear": linear_predictor,
            "moving_average": moving_average_predictor
        }
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a metric value for learning."""
        if timestamp is None:
            timestamp = time.time()
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # Keep history manageable
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name] = self.metric_history[metric_name][-500:]
    
    def record_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record a system event for learning."""
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "data": event_data
        }
        
        self.system_events.append(event)
        
        # Keep events manageable
        if len(self.system_events) > 1000:
            self.system_events = self.system_events[-500:]
    
    def _detect_patterns(self) -> List[Pattern]:
        """Detect patterns in collected data."""
        all_patterns = []
        
        for metric_name, data in self.metric_history.items():
            if len(data) < 5:
                continue
            
            for detector_name, detector_func in self.pattern_detectors.items():
                try:
                    patterns = detector_func(metric_name, data)
                    all_patterns.extend(patterns)
                except Exception as e:
                    logger.error(f"Error in pattern detector {detector_name}: {e}")
        
        # Filter patterns by strength threshold
        strong_patterns = [p for p in all_patterns if p.strength >= self.pattern_threshold]
        
        # Update discovered patterns
        for pattern in strong_patterns:
            if pattern.pattern_id not in self.discovered_patterns:
                self.discovered_patterns[pattern.pattern_id] = pattern
                logger.info(f"Discovered new pattern: {pattern.description}")
        
        return strong_patterns
    
    def _generate_predictions(self) -> List[Prediction]:
        """Generate predictions for system metrics."""
        predictions = []
        
        for metric_name, data in self.metric_history.items():
            if len(data) < 3:
                continue
            
            for model_name, model_func in self.predictive_models.items():
                try:
                    prediction = model_func(metric_name, data, self.prediction_horizon)
                    if prediction:
                        predictions.append(prediction)
                        self.predictions[prediction.prediction_id] = prediction
                except Exception as e:
                    logger.error(f"Error in predictive model {model_name}: {e}")
        
        return predictions
    
    def _validate_predictions(self):
        """Validate previous predictions against actual values."""
        current_time = time.time()
        
        for prediction_id, prediction in list(self.predictions.items()):
            # Check if prediction time has passed
            prediction_time = prediction.timestamp + prediction.time_horizon
            
            if current_time >= prediction_time:
                # Find actual value at prediction time
                metric_name = prediction.target_metric
                
                if metric_name in self.metric_history:
                    # Find closest actual value
                    actual_data = [
                        (t, v) for t, v in self.metric_history[metric_name]
                        if abs(t - prediction_time) <= 30.0  # Within 30 seconds
                    ]
                    
                    if actual_data:
                        # Use closest value
                        closest_data = min(actual_data, key=lambda x: abs(x[0] - prediction_time))
                        actual_value = closest_data[1]
                        
                        # Calculate accuracy
                        error = abs(prediction.predicted_value - actual_value)
                        max_error = max(abs(prediction.predicted_value), abs(actual_value), 1.0)
                        accuracy = max(0.0, 1.0 - (error / max_error))
                        
                        # Update prediction
                        prediction.actual_value = actual_value
                        prediction.accuracy = accuracy
                        
                        # Update meta-knowledge
                        self._update_meta_knowledge("prediction_accuracy", accuracy)
                        
                        logger.info(f"Validated prediction {prediction_id}: accuracy={accuracy:.3f}")
                
                # Remove from active predictions
                del self.predictions[prediction_id]
    
    def _update_meta_knowledge(self, knowledge_type: str, value: float):
        """Update meta-learning knowledge."""
        if knowledge_type in self.meta_knowledge:
            # Exponential moving average
            alpha = 0.1
            self.meta_knowledge[knowledge_type] = (
                alpha * value + (1 - alpha) * self.meta_knowledge[knowledge_type]
            )
    
    def _adapt_learning_strategy(self):
        """Adapt learning strategy based on meta-knowledge."""
        exploration_effectiveness = self.meta_knowledge["exploration_effectiveness"]
        exploitation_efficiency = self.meta_knowledge["exploitation_efficiency"]
        
        # Decide on learning phase
        if exploration_effectiveness > 0.7 and self.current_phase != LearningPhase.EXPLORATION:
            self.current_phase = LearningPhase.EXPLORATION
            logger.info("Switched to exploration phase")
        elif exploitation_efficiency > 0.7 and self.current_phase != LearningPhase.EXPLOITATION:
            self.current_phase = LearningPhase.EXPLOITATION
            logger.info("Switched to exploitation phase")
        elif self.current_phase not in [LearningPhase.ADAPTATION, LearningPhase.VALIDATION]:
            self.current_phase = LearningPhase.ADAPTATION
            logger.info("Switched to adaptation phase")
    
    async def _learning_cycle(self):
        """Execute one learning cycle."""
        try:
            cycle_start = time.time()
            
            # Phase 1: Pattern Detection
            patterns = self._detect_patterns()
            
            # Phase 2: Prediction Generation
            predictions = self._generate_predictions()
            
            # Phase 3: Prediction Validation
            self._validate_predictions()
            
            # Phase 4: Meta-Learning
            self._adapt_learning_strategy()
            
            # Create learning event
            event = LearningEvent(
                event_id=f"learning_{int(cycle_start)}",
                learning_type=LearningType.PATTERN_RECOGNITION,
                phase=self.current_phase,
                data={
                    "patterns_detected": len(patterns),
                    "predictions_generated": len(predictions),
                    "total_patterns": len(self.discovered_patterns),
                    "active_predictions": len(self.predictions)
                },
                insights=[
                    f"Detected {len(patterns)} new patterns",
                    f"Generated {len(predictions)} predictions",
                    f"Current phase: {self.current_phase.value}"
                ],
                confidence=statistics.mean(self.meta_knowledge.values())
            )
            
            self.learning_events.append(event)
            
            # Keep events manageable
            if len(self.learning_events) > 200:
                self.learning_events = self.learning_events[-100:]
            
            cycle_duration = time.time() - cycle_start
            logger.info(f"Learning cycle completed in {cycle_duration:.2f}s: {len(patterns)} patterns, {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def _learning_loop(self):
        """Main learning loop."""
        logger.info("Continuous learning coordinator started")
        
        while self.is_running:
            try:
                await self._learning_cycle()
                await asyncio.sleep(self.learning_interval)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(self.learning_interval)
    
    async def start(self):
        """Start the continuous learning coordinator."""
        if self.is_running:
            logger.warning("Learning coordinator is already running")
            return
        
        self.is_running = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        logger.info("Continuous learning coordinator started")
    
    async def stop(self):
        """Stop the continuous learning coordinator."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Continuous learning coordinator stopped")
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get current learning insights and discoveries."""
        # Pattern insights
        pattern_types = {}
        for pattern in self.discovered_patterns.values():
            pattern_types[pattern.pattern_type] = pattern_types.get(pattern.pattern_type, 0) + 1
        
        # Prediction insights
        prediction_metrics = {}
        prediction_accuracies = []
        
        for event in self.learning_events:
            for prediction in self.predictions.values():
                if prediction.actual_value is not None:
                    prediction_accuracies.append(prediction.accuracy)
                    
                metric = prediction.target_metric
                prediction_metrics[metric] = prediction_metrics.get(metric, 0) + 1
        
        avg_prediction_accuracy = statistics.mean(prediction_accuracies) if prediction_accuracies else 0.0
        
        return {
            "current_phase": self.current_phase.value,
            "total_patterns": len(self.discovered_patterns),
            "pattern_distribution": pattern_types,
            "active_predictions": len(self.predictions),
            "prediction_metrics": prediction_metrics,
            "average_prediction_accuracy": avg_prediction_accuracy,
            "meta_knowledge": self.meta_knowledge.copy(),
            "learning_events": len(self.learning_events),
            "metrics_tracked": len(self.metric_history),
            "system_events": len(self.system_events)
        }
    
    def get_pattern_summary(self) -> List[Dict[str, Any]]:
        """Get summary of discovered patterns."""
        return [
            {
                "pattern_id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "description": pattern.description,
                "strength": pattern.strength,
                "occurrences": pattern.occurrences,
                "last_seen": pattern.last_seen,
                "parameters": pattern.parameters
            }
            for pattern in sorted(
                self.discovered_patterns.values(),
                key=lambda p: p.strength,
                reverse=True
            )
        ]
    
    def get_prediction_summary(self) -> List[Dict[str, Any]]:
        """Get summary of current predictions."""
        return [
            {
                "prediction_id": pred.prediction_id,
                "target_metric": pred.target_metric,
                "predicted_value": pred.predicted_value,
                "confidence": pred.confidence,
                "time_horizon": pred.time_horizon,
                "actual_value": pred.actual_value,
                "accuracy": pred.accuracy,
                "timestamp": pred.timestamp
            }
            for pred in sorted(
                self.predictions.values(),
                key=lambda p: p.confidence,
                reverse=True
            )
        ]
    
    def export_knowledge_base(self) -> Dict[str, Any]:
        """Export entire knowledge base for persistence."""
        return {
            "patterns": {
                pid: {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "description": p.description,
                    "strength": p.strength,
                    "occurrences": p.occurrences,
                    "last_seen": p.last_seen,
                    "parameters": p.parameters
                }
                for pid, p in self.discovered_patterns.items()
            },
            "meta_knowledge": self.meta_knowledge.copy(),
            "learning_events": [
                {
                    "event_id": e.event_id,
                    "learning_type": e.learning_type.value,
                    "phase": e.phase.value,
                    "data": e.data,
                    "insights": e.insights,
                    "confidence": e.confidence,
                    "timestamp": e.timestamp
                }
                for e in self.learning_events[-50:]  # Last 50 events
            ],
            "system_events": self.system_events[-100:],  # Last 100 events
            "current_phase": self.current_phase.value,
            "pattern_threshold": self.pattern_threshold,
            "prediction_horizon": self.prediction_horizon
        }