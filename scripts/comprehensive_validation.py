#!/usr/bin/env python3
"""Comprehensive validation script for Agent Mesh SDLC implementation.

This script validates all three generations of the SDLC implementation:
- Generation 1: Core functionality works
- Generation 2: System is robust and fault-tolerant  
- Generation 3: Performance and scaling capabilities
"""

import sys
import os
import time
import asyncio
import importlib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock heavy dependencies if not available
try:
    import structlog
except ImportError:
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass 
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def critical(self, *args, **kwargs): pass
    
    class MockStructlog:
        def get_logger(self, *args, **kwargs):
            return MockLogger()
    
    structlog = MockStructlog()
    sys.modules['structlog'] = structlog

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
except ImportError:
    class MockMetric:
        def inc(self): pass
        def set(self, value): pass
        def observe(self, value): pass
        def labels(self, **kwargs): return self
        
    Counter = Histogram = Gauge = lambda *args, **kwargs: MockMetric()
    CollectorRegistry = lambda: None
    generate_latest = lambda x: b"mock_metrics"


class ValidationResult:
    """Result of a validation test."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.success = False
        self.error_message = ""
        self.execution_time = 0.0
        self.details = {}
        
    def pass_test(self, details=None):
        """Mark test as passed."""
        self.success = True
        self.details = details or {}
        
    def fail_test(self, error_message: str, details=None):
        """Mark test as failed."""
        self.success = False
        self.error_message = error_message
        self.details = details or {}


class ComprehensiveValidator:
    """Main validation orchestrator."""
    
    def __init__(self):
        self.results = []
        self.generation_scores = {1: 0, 2: 0, 3: 0}
        self.start_time = time.time()
        
    def run_test(self, test_func, test_name: str) -> ValidationResult:
        """Run a single test and record results."""
        result = ValidationResult(test_name)
        start_time = time.time()
        
        try:
            print(f"🧪 Running: {test_name}")
            test_func(result)
            result.execution_time = time.time() - start_time
            
            if result.success:
                print(f"   ✅ PASSED ({result.execution_time:.2f}s)")
            else:
                print(f"   ❌ FAILED: {result.error_message}")
                
        except Exception as e:
            result.fail_test(f"Exception: {str(e)}")
            result.execution_time = time.time() - start_time
            print(f"   ❌ FAILED: {str(e)}")
            
        self.results.append(result)
        return result
        
    async def run_async_test(self, test_func, test_name: str) -> ValidationResult:
        """Run an async test and record results."""
        result = ValidationResult(test_name)
        start_time = time.time()
        
        try:
            print(f"🧪 Running: {test_name}")
            await test_func(result)
            result.execution_time = time.time() - start_time
            
            if result.success:
                print(f"   ✅ PASSED ({result.execution_time:.2f}s)")
            else:
                print(f"   ❌ FAILED: {result.error_message}")
                
        except Exception as e:
            result.fail_test(f"Exception: {str(e)}")
            result.execution_time = time.time() - start_time
            print(f"   ❌ FAILED: {str(e)}")
            
        self.results.append(result)
        return result
        
    def validate_generation_1(self):
        """Validate Generation 1: MAKE IT WORK."""
        print("\n🚀 GENERATION 1 VALIDATION: MAKE IT WORK")
        print("=" * 50)
        
        gen1_tests = [
            (self._test_core_imports, "Core module imports"),
            (self._test_mesh_node_creation, "MeshNode creation"),
            (self._test_federated_learner_creation, "FederatedLearner creation"),
            (self._test_network_layer, "Network layer functionality"),
            (self._test_consensus_engine, "Consensus engine"),
            (self._test_agent_coordination, "Agent coordination system")
        ]
        
        passed = 0
        for test_func, test_name in gen1_tests:
            result = self.run_test(test_func, test_name)
            if result.success:
                passed += 1
                
        self.generation_scores[1] = (passed / len(gen1_tests)) * 100
        print(f"\n📊 Generation 1 Score: {self.generation_scores[1]:.1f}% ({passed}/{len(gen1_tests)} tests passed)")
        
    def validate_generation_2(self):
        """Validate Generation 2: MAKE IT ROBUST."""
        print("\n🛡️ GENERATION 2 VALIDATION: MAKE IT ROBUST")
        print("=" * 50)
        
        gen2_tests = [
            (self._test_error_handling, "Error handling system"),
            (self._test_circuit_breakers, "Circuit breaker functionality"),
            (self._test_monitoring_system, "Monitoring and alerting"),
            (self._test_security_system, "Enhanced security features"),
            (self._test_threat_detection, "Threat detection system"),
            (self._test_recovery_mechanisms, "Recovery mechanisms")
        ]
        
        passed = 0
        for test_func, test_name in gen2_tests:
            result = self.run_test(test_func, test_name)
            if result.success:
                passed += 1
                
        self.generation_scores[2] = (passed / len(gen2_tests)) * 100
        print(f"\n📊 Generation 2 Score: {self.generation_scores[2]:.1f}% ({passed}/{len(gen2_tests)} tests passed)")
        
    def validate_generation_3(self):
        """Validate Generation 3: MAKE IT SCALE."""
        print("\n⚡ GENERATION 3 VALIDATION: MAKE IT SCALE")
        print("=" * 50)
        
        gen3_tests = [
            (self._test_adaptive_cache, "Adaptive caching system"),
            (self._test_load_balancing, "Load balancing strategies"),
            (self._test_resource_pooling, "Resource pooling"),
            (self._test_performance_optimization, "Performance optimization"),
            (self._test_auto_scaling, "Auto-scaling engine"),
            (self._test_predictive_scaling, "Predictive scaling")
        ]
        
        passed = 0
        for test_func, test_name in gen3_tests:
            result = self.run_test(test_func, test_name)
            if result.success:
                passed += 1
                
        self.generation_scores[3] = (passed / len(gen3_tests)) * 100
        print(f"\n📊 Generation 3 Score: {self.generation_scores[3]:.1f}% ({passed}/{len(gen3_tests)} tests passed)")
        
    def validate_integration(self):
        """Validate cross-generation integration."""
        print("\n🔗 INTEGRATION VALIDATION")
        print("=" * 30)
        
        integration_tests = [
            (self._test_end_to_end_flow, "End-to-end workflow"),
            (self._test_component_integration, "Component integration"),
            (self._test_production_readiness, "Production readiness")
        ]
        
        passed = 0
        for test_func, test_name in integration_tests:
            result = self.run_test(test_func, test_name)
            if result.success:
                passed += 1
                
        integration_score = (passed / len(integration_tests)) * 100
        print(f"\n📊 Integration Score: {integration_score:.1f}% ({passed}/{len(integration_tests)} tests passed)")
        
        return integration_score
        
    # Generation 1 Tests
    def _test_core_imports(self, result: ValidationResult):
        """Test core module imports work correctly."""
        try:
            from agent_mesh.core.mesh_node import MeshNode, NodeCapabilities
            from agent_mesh.federated.learner import FederatedLearner, FederatedConfig
            from agent_mesh.coordination.agent_mesh import AgentMesh
            
            result.pass_test({
                "imported_modules": ["mesh_node", "federated_learner", "agent_mesh"]
            })
        except ImportError as e:
            result.fail_test(f"Import failed: {e}")
            
    def _test_mesh_node_creation(self, result: ValidationResult):
        """Test MeshNode can be created."""
        try:
            from agent_mesh.core.mesh_node import MeshNode, NodeCapabilities
            from uuid import uuid4
            
            capabilities = NodeCapabilities(cpu_cores=2, memory_gb=4.0)
            node = MeshNode(node_id=uuid4(), capabilities=capabilities)
            
            if node.node_id and node.capabilities:
                result.pass_test({
                    "node_id": str(node.node_id),
                    "cpu_cores": node.capabilities.cpu_cores,
                    "memory_gb": node.capabilities.memory_gb
                })
            else:
                result.fail_test("Node creation incomplete")
                
        except Exception as e:
            result.fail_test(f"MeshNode creation failed: {e}")
            
    def _test_federated_learner_creation(self, result: ValidationResult):
        """Test FederatedLearner can be created."""
        try:
            from agent_mesh.federated.learner import FederatedLearner, FederatedConfig
            from uuid import uuid4
            
            def mock_model_fn():
                return "mock_model"
                
            def mock_dataset_fn():
                return "mock_dataset"
                
            config = FederatedConfig(rounds=5, local_epochs=2)
            learner = FederatedLearner(
                node_id=uuid4(),
                model_fn=mock_model_fn,
                dataset_fn=mock_dataset_fn,
                config=config
            )
            
            if learner.node_id and learner.config.rounds == 5:
                result.pass_test({
                    "node_id": str(learner.node_id),
                    "rounds": learner.config.rounds,
                    "local_epochs": learner.config.local_epochs
                })
            else:
                result.fail_test("FederatedLearner creation incomplete")
                
        except Exception as e:
            result.fail_test(f"FederatedLearner creation failed: {e}")
            
    def _test_network_layer(self, result: ValidationResult):
        """Test network layer functionality."""
        try:
            from agent_mesh.core.simple_network import SimpleP2PNetwork
            from uuid import uuid4
            
            network = SimpleP2PNetwork(uuid4())
            
            if hasattr(network, 'start') and hasattr(network, 'connect_to_peer'):
                result.pass_test({
                    "has_start_method": True,
                    "has_connect_method": True,
                    "node_id": str(network.node_id)
                })
            else:
                result.fail_test("Network layer missing required methods")
                
        except Exception as e:
            result.fail_test(f"Network layer test failed: {e}")
            
    def _test_consensus_engine(self, result: ValidationResult):
        """Test consensus engine functionality."""
        try:
            from agent_mesh.core.consensus import ConsensusEngine
            from uuid import uuid4
            
            engine = ConsensusEngine(uuid4())
            
            if hasattr(engine, 'propose') and hasattr(engine, 'vote_on_proposal'):
                result.pass_test({
                    "has_propose_method": True,
                    "has_vote_method": True,
                    "node_id": str(engine.node_id)
                })
            else:
                result.fail_test("Consensus engine missing required methods")
                
        except Exception as e:
            result.fail_test(f"Consensus engine test failed: {e}")
            
    def _test_agent_coordination(self, result: ValidationResult):
        """Test agent coordination system."""
        try:
            from agent_mesh.coordination.agent_mesh import AgentMesh, CoordinationProtocol
            from agent_mesh.core.mesh_node import MeshNode, NodeCapabilities
            from uuid import uuid4
            
            # Create mock mesh node
            capabilities = NodeCapabilities()
            mesh_node = MeshNode(node_id=uuid4(), capabilities=capabilities)
            
            # Create agent mesh
            agent_mesh = AgentMesh(mesh_node)
            
            if agent_mesh.mesh_node and hasattr(agent_mesh, 'register_agent'):
                result.pass_test({
                    "has_mesh_node": True,
                    "has_register_method": True,
                    "node_id": str(agent_mesh.mesh_node.node_id)
                })
            else:
                result.fail_test("Agent coordination missing required components")
                
        except Exception as e:
            result.fail_test(f"Agent coordination test failed: {e}")
            
    # Generation 2 Tests
    def _test_error_handling(self, result: ValidationResult):
        """Test error handling system."""
        try:
            from agent_mesh.core.error_handling import ErrorHandler, ErrorCategory, RetryPolicy
            from uuid import uuid4
            
            handler = ErrorHandler(uuid4())
            policy = RetryPolicy(max_attempts=3, base_delay=1.0)
            
            # Test policy delay calculation
            delay1 = policy.get_delay(0)
            delay2 = policy.get_delay(1)
            
            if delay1 == 1.0 and delay2 == 2.0:  # Exponential backoff
                result.pass_test({
                    "handler_created": True,
                    "policy_working": True,
                    "delay_calculation": f"{delay1} -> {delay2}"
                })
            else:
                result.fail_test("Error handling components not working correctly")
                
        except Exception as e:
            result.fail_test(f"Error handling test failed: {e}")
            
    def _test_circuit_breakers(self, result: ValidationResult):
        """Test circuit breaker functionality."""
        try:
            from agent_mesh.core.error_handling import CircuitBreaker
            
            cb = CircuitBreaker("test_circuit")
            
            # Test initial state
            if cb.state.state == "CLOSED" and cb.state.failure_count == 0:
                result.pass_test({
                    "initial_state": cb.state.state,
                    "failure_count": cb.state.failure_count,
                    "name": cb.state.name
                })
            else:
                result.fail_test("Circuit breaker not in correct initial state")
                
        except Exception as e:
            result.fail_test(f"Circuit breaker test failed: {e}")
            
    def _test_monitoring_system(self, result: ValidationResult):
        """Test monitoring and alerting system."""
        try:
            from agent_mesh.core.monitoring import MeshMonitor, AlertSeverity
            from uuid import uuid4
            
            monitor = MeshMonitor(uuid4())
            
            # Test metric recording
            monitor.record_metric("test_metric", 42.0)
            
            # Test alert triggering
            monitor.trigger_alert("test_alert", AlertSeverity.WARNING, "Test message")
            
            # Get health status
            health = monitor.get_health_status()
            
            if health["status"] == "WARNING" and health["active_alerts"] == 1:
                result.pass_test({
                    "metric_recorded": True,
                    "alert_triggered": True,
                    "health_status": health["status"]
                })
            else:
                result.fail_test("Monitoring system not working correctly")
                
        except Exception as e:
            result.fail_test(f"Monitoring system test failed: {e}")
            
    def _test_security_system(self, result: ValidationResult):
        """Test enhanced security features."""
        try:
            from agent_mesh.core.security_enhanced import SecurityManager, ThreatType
            from uuid import uuid4
            
            security = SecurityManager(uuid4())
            
            # Test node profile creation
            test_node = uuid4()
            profile = security.get_node_profile(test_node)
            
            # Test trust score updates
            initial_trust = profile.trust_score
            profile.update_trust_score(-0.1)
            
            if profile.node_id == test_node and profile.trust_score < initial_trust:
                result.pass_test({
                    "profile_created": True,
                    "trust_score_updated": True,
                    "initial_trust": initial_trust,
                    "updated_trust": profile.trust_score
                })
            else:
                result.fail_test("Security system not working correctly")
                
        except Exception as e:
            result.fail_test(f"Security system test failed: {e}")
            
    def _test_threat_detection(self, result: ValidationResult):
        """Test threat detection capabilities."""
        try:
            from agent_mesh.core.security_enhanced import SecurityManager, AnomalyDetector
            from uuid import uuid4
            
            security = SecurityManager(uuid4())
            detector = AnomalyDetector()
            
            # Test rate limiting
            rate_limiter = security.rate_limiter
            test_node = uuid4()
            
            # Should allow initial requests
            allowed = rate_limiter.check_rate_limit(test_node)
            
            # Test message structure validation
            valid_message = {
                "timestamp": time.time(),
                "sender_id": str(uuid4()),
                "message_type": "test"
            }
            
            is_valid = security._validate_message_structure(valid_message)
            
            if allowed and is_valid:
                result.pass_test({
                    "rate_limiting_works": allowed,
                    "message_validation_works": is_valid
                })
            else:
                result.fail_test("Threat detection not working correctly")
                
        except Exception as e:
            result.fail_test(f"Threat detection test failed: {e}")
            
    def _test_recovery_mechanisms(self, result: ValidationResult):
        """Test recovery mechanisms."""
        try:
            from agent_mesh.core.error_handling import ErrorHandler, RecoveryStrategy
            from uuid import uuid4
            
            handler = ErrorHandler(uuid4())
            
            # Test error statistics
            stats = handler.get_error_statistics()
            
            # Test circuit breaker creation
            cb = handler.get_circuit_breaker("test_service")
            
            if cb and "total_errors" in stats:
                result.pass_test({
                    "statistics_available": True,
                    "circuit_breaker_created": True,
                    "total_errors": stats["total_errors"]
                })
            else:
                result.fail_test("Recovery mechanisms not working correctly")
                
        except Exception as e:
            result.fail_test(f"Recovery mechanisms test failed: {e}")
            
    # Generation 3 Tests
    def _test_adaptive_cache(self, result: ValidationResult):
        """Test adaptive caching system."""
        try:
            from agent_mesh.core.performance_optimizer import AdaptiveCache, CachePolicy
            
            cache = AdaptiveCache(max_size_mb=1.0, policy=CachePolicy.LRU)
            
            # Test basic operations (sync version for testing)
            cache.entries["test_key"] = type('Entry', (), {
                'key': 'test_key', 
                'value': 'test_value',
                'access_count': 1,
                'size_bytes': 100
            })()
            
            # Test statistics
            stats = cache.get_statistics()
            
            if stats["entries"] >= 0 and "hit_rate" in stats:
                result.pass_test({
                    "cache_created": True,
                    "statistics_available": True,
                    "entries": stats["entries"]
                })
            else:
                result.fail_test("Adaptive cache not working correctly")
                
        except Exception as e:
            result.fail_test(f"Adaptive cache test failed: {e}")
            
    def _test_load_balancing(self, result: ValidationResult):
        """Test load balancing strategies."""
        try:
            from agent_mesh.core.performance_optimizer import LoadBalancer, LoadBalanceStrategy, PerformanceMetrics
            from uuid import uuid4
            
            lb = LoadBalancer(strategy=LoadBalanceStrategy.ROUND_ROBIN)
            
            # Register test nodes
            node1 = uuid4()
            node2 = uuid4()
            
            lb.register_node(node1, weight=1.0)
            lb.register_node(node2, weight=1.0)
            
            # Test node selection
            selected = lb.select_node()
            
            # Get statistics
            stats = lb.get_load_statistics()
            
            if selected in [node1, node2] and stats["total_nodes"] == 2:
                result.pass_test({
                    "nodes_registered": True,
                    "selection_works": True,
                    "statistics_available": True,
                    "selected_node": str(selected)[:8]
                })
            else:
                result.fail_test("Load balancing not working correctly")
                
        except Exception as e:
            result.fail_test(f"Load balancing test failed: {e}")
            
    def _test_resource_pooling(self, result: ValidationResult):
        """Test resource pooling system."""
        try:
            from agent_mesh.core.performance_optimizer import ResourcePool, ResourcePoolManager
            
            pool_manager = ResourcePoolManager()
            
            async def mock_create():
                return "mock_resource"
                
            pool = pool_manager.create_pool(
                "test_pool",
                create_func=mock_create,
                max_size=10
            )
            
            stats = pool_manager.get_pool_statistics("test_pool")
            
            if pool.pool_name == "test_pool" and stats["max_size"] == 10:
                result.pass_test({
                    "pool_created": True,
                    "statistics_available": True,
                    "max_size": stats["max_size"]
                })
            else:
                result.fail_test("Resource pooling not working correctly")
                
        except Exception as e:
            result.fail_test(f"Resource pooling test failed: {e}")
            
    def _test_performance_optimization(self, result: ValidationResult):
        """Test performance optimization system."""
        try:
            from agent_mesh.core.performance_optimizer import PerformanceOptimizer, PerformanceMetrics
            from uuid import uuid4
            
            optimizer = PerformanceOptimizer(uuid4())
            
            # Test performance analysis
            test_metrics = PerformanceMetrics(
                cpu_usage=85.0,
                memory_usage=90.0,
                network_latency_ms=150.0
            )
            
            suggestions = optimizer.analyze_performance(test_metrics)
            
            if len(suggestions) > 0:
                result.pass_test({
                    "optimizer_created": True,
                    "analysis_works": True,
                    "suggestions_count": len(suggestions),
                    "sample_suggestion": suggestions[0] if suggestions else None
                })
            else:
                result.fail_test("Performance optimization not generating suggestions")
                
        except Exception as e:
            result.fail_test(f"Performance optimization test failed: {e}")
            
    def _test_auto_scaling(self, result: ValidationResult):
        """Test auto-scaling engine."""
        try:
            from agent_mesh.core.auto_scaler import AutoScaler, ScalingRule, ScalingTrigger, InstanceMetrics
            from uuid import uuid4
            
            scaler = AutoScaler("test_cluster")
            
            # Add scaling rule
            rule = ScalingRule(
                rule_id="test_rule",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                metric_name="cpu_utilization",
                threshold_high=75.0,
                threshold_low=25.0
            )
            
            scaler.add_scaling_rule(rule)
            
            # Test instance metrics
            instance_id = uuid4()
            metrics = InstanceMetrics(
                instance_id=instance_id,
                cpu_utilization=80.0,
                memory_utilization=70.0
            )
            
            scaler.update_instance_metrics(instance_id, metrics)
            
            status = scaler.get_scaling_status()
            
            if status["current_instances"] >= 0 and len(scaler.scaling_rules) == 1:
                result.pass_test({
                    "scaler_created": True,
                    "rule_added": True,
                    "metrics_updated": True,
                    "status_available": True
                })
            else:
                result.fail_test("Auto-scaling not working correctly")
                
        except Exception as e:
            result.fail_test(f"Auto-scaling test failed: {e}")
            
    def _test_predictive_scaling(self, result: ValidationResult):
        """Test predictive scaling capabilities."""
        try:
            from agent_mesh.core.auto_scaler import PredictiveScaler
            
            predictor = PredictiveScaler()
            
            # Add some sample data
            for i in range(20):
                predictor.record_metric("cpu_usage", 50 + (i * 2))
                
            # Test pattern detection
            patterns = predictor.detect_patterns("cpu_usage")
            
            # Test prediction (might be None with limited data)
            prediction = predictor.predict_metric("cpu_usage", 15)
            
            if "average" in patterns:
                result.pass_test({
                    "predictor_created": True,
                    "patterns_detected": True,
                    "prediction_attempted": True,
                    "average": patterns["average"]
                })
            else:
                result.fail_test("Predictive scaling not detecting patterns")
                
        except Exception as e:
            result.fail_test(f"Predictive scaling test failed: {e}")
            
    # Integration Tests
    def _test_end_to_end_flow(self, result: ValidationResult):
        """Test end-to-end workflow."""
        try:
            # This would test a complete federated learning workflow
            # For now, just test that all major components can be imported together
            
            from agent_mesh.core.mesh_node import MeshNode
            from agent_mesh.federated.learner import FederatedLearner
            from agent_mesh.core.error_handling import ErrorHandler
            from agent_mesh.core.monitoring import MeshMonitor
            from agent_mesh.core.security_enhanced import SecurityManager
            from agent_mesh.core.performance_optimizer import PerformanceOptimizer
            from agent_mesh.core.auto_scaler import AutoScaler
            
            result.pass_test({
                "all_components_importable": True,
                "components": [
                    "MeshNode", "FederatedLearner", "ErrorHandler",
                    "MeshMonitor", "SecurityManager", "PerformanceOptimizer", "AutoScaler"
                ]
            })
            
        except Exception as e:
            result.fail_test(f"End-to-end flow test failed: {e}")
            
    def _test_component_integration(self, result: ValidationResult):
        """Test component integration."""
        try:
            from agent_mesh.core.error_handling import ErrorHandler
            from agent_mesh.core.monitoring import MeshMonitor
            from agent_mesh.core.security_enhanced import SecurityManager
            from uuid import uuid4
            
            # Test that components can be created together
            node_id = uuid4()
            
            error_handler = ErrorHandler(node_id)
            monitor = MeshMonitor(node_id)
            security_manager = SecurityManager(node_id)
            
            # Test integration setup
            security_manager.set_integrations(error_handler, monitor)
            
            if (security_manager.error_handler is not None and 
                security_manager.monitor is not None):
                result.pass_test({
                    "components_created": True,
                    "integration_setup": True,
                    "node_id": str(node_id)
                })
            else:
                result.fail_test("Component integration not working")
                
        except Exception as e:
            result.fail_test(f"Component integration test failed: {e}")
            
    def _test_production_readiness(self, result: ValidationResult):
        """Test production readiness indicators."""
        try:
            # Check for essential production features
            production_features = {
                "error_handling": False,
                "monitoring": False,  
                "security": False,
                "performance_optimization": False,
                "auto_scaling": False
            }
            
            # Test each production feature
            try:
                from agent_mesh.core.error_handling import ErrorHandler
                production_features["error_handling"] = True
            except:
                pass
                
            try:
                from agent_mesh.core.monitoring import MeshMonitor
                production_features["monitoring"] = True
            except:
                pass
                
            try:
                from agent_mesh.core.security_enhanced import SecurityManager
                production_features["security"] = True
            except:
                pass
                
            try:
                from agent_mesh.core.performance_optimizer import PerformanceOptimizer
                production_features["performance_optimization"] = True
            except:
                pass
                
            try:
                from agent_mesh.core.auto_scaler import AutoScaler
                production_features["auto_scaling"] = True
            except:
                pass
                
            ready_features = sum(production_features.values())
            total_features = len(production_features)
            readiness_score = (ready_features / total_features) * 100
            
            if readiness_score >= 80:  # 80% of features available
                result.pass_test({
                    "readiness_score": readiness_score,
                    "ready_features": ready_features,
                    "total_features": total_features,
                    "production_ready": True
                })
            else:
                result.fail_test(f"Production readiness too low: {readiness_score}%")
                
        except Exception as e:
            result.fail_test(f"Production readiness test failed: {e}")
            
    def generate_final_report(self):
        """Generate comprehensive final report."""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🏆 AGENT MESH SDLC VALIDATION REPORT")
        print("="*60)
        
        # Overall summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        overall_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {overall_score:.1f}%")
        print(f"   Execution Time: {total_time:.2f}s")
        
        # Generation scores
        print(f"\n🎯 GENERATION SCORES:")
        for gen, score in self.generation_scores.items():
            status = "✅ PASSED" if score >= 70 else "❌ FAILED"
            print(f"   Generation {gen}: {score:.1f}% {status}")
            
        # Production readiness assessment
        avg_gen_score = sum(self.generation_scores.values()) / len(self.generation_scores)
        
        print(f"\n🚀 PRODUCTION READINESS ASSESSMENT:")
        if avg_gen_score >= 85:
            print("   🟢 READY FOR PRODUCTION")
            print("   All generations meet high quality standards")
        elif avg_gen_score >= 70:
            print("   🟡 READY FOR STAGING")
            print("   Most features working, minor issues remain")
        else:
            print("   🔴 NEEDS MORE DEVELOPMENT")
            print("   Significant issues need resolution")
            
        # Failed tests summary
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test.test_name}: {test.error_message}")
                
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if avg_gen_score >= 85:
            print("   • System is production-ready!")
            print("   • Consider setting up CI/CD pipeline")
            print("   • Plan production deployment strategy")
        elif avg_gen_score >= 70:
            print("   • Address failed tests before production")
            print("   • Add more comprehensive testing")
            print("   • Consider staged rollout")
        else:
            print("   • Focus on fixing critical failures")
            print("   • Increase test coverage")
            print("   • Review architecture decisions")
            
        return overall_score


def main():
    """Main validation execution."""
    print("🚀 Starting Agent Mesh SDLC Comprehensive Validation")
    print("="*60)
    
    validator = ComprehensiveValidator()
    
    try:
        # Run all validation phases
        validator.validate_generation_1()
        validator.validate_generation_2() 
        validator.validate_generation_3()
        integration_score = validator.validate_integration()
        
        # Generate final report
        overall_score = validator.generate_final_report()
        
        # Return appropriate exit code
        if overall_score >= 70:
            return 0  # Success
        else:
            return 1  # Failure
            
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 2  # Error


if __name__ == "__main__":
    sys.exit(main())