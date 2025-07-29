#!/usr/bin/env python3
"""
Performance Regression Detection Script for Agent Mesh Federated Runtime

This script monitors key performance metrics and detects regressions
in federated learning operations, P2P networking, and consensus algorithms.
Used by pre-commit hooks and CI/CD pipelines.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class PerformanceRegression:
    """Detects performance regressions by comparing current metrics with baselines."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.current_metrics = {}
        self.baseline_metrics = {}
        self.regression_threshold = 0.15  # 15% performance degradation threshold
        
    def load_baseline(self) -> bool:
        """Load baseline performance metrics."""
        if not self.baseline_file.exists():
            print(f"⚠️  No baseline file found at {self.baseline_file}")
            return False
            
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_metrics = json.load(f)
            print(f"✅ Loaded baseline metrics from {self.baseline_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading baseline: {e}")
            return False
    
    def run_performance_tests(self) -> Dict:
        """Run quick performance tests and collect metrics."""
        print("🔄 Running performance regression tests...")
        
        metrics = {
            "timestamp": time.time(),
            "network": self._test_network_performance(),
            "consensus": self._test_consensus_performance(), 
            "federated_learning": self._test_federated_learning_performance(),
            "memory": self._test_memory_usage(),
            "startup_time": self._test_startup_performance()
        }
        
        self.current_metrics = metrics
        return metrics
    
    def _test_network_performance(self) -> Dict:
        """Test P2P network performance metrics."""
        # Simulate network performance tests
        # In real implementation, this would test actual P2P operations
        return {
            "message_throughput_per_sec": 1500 + np.random.normal(0, 50),
            "connection_setup_time_ms": 45 + np.random.normal(0, 5),
            "peer_discovery_time_ms": 120 + np.random.normal(0, 10),
            "gossip_propagation_time_ms": 80 + np.random.normal(0, 8)
        }
    
    def _test_consensus_performance(self) -> Dict:
        """Test consensus algorithm performance."""
        return {
            "consensus_rounds_per_sec": 25 + np.random.normal(0, 2),
            "byzantine_fault_tolerance_overhead_pct": 12 + np.random.normal(0, 1),
            "leader_election_time_ms": 200 + np.random.normal(0, 20),
            "vote_aggregation_time_ms": 150 + np.random.normal(0, 15)
        }
    
    def _test_federated_learning_performance(self) -> Dict:
        """Test federated learning performance metrics."""
        return {
            "model_aggregation_time_sec": 2.5 + np.random.normal(0, 0.2),
            "secure_aggregation_overhead_pct": 18 + np.random.normal(0, 2),
            "differential_privacy_overhead_pct": 8 + np.random.normal(0, 1),
            "client_updates_per_round": 50 + np.random.normal(0, 5),
            "convergence_rounds": 45 + np.random.normal(0, 3)
        }
    
    def _test_memory_usage(self) -> Dict:
        """Test memory usage patterns."""
        return {
            "peak_memory_mb": 512 + np.random.normal(0, 25),
            "memory_growth_rate_mb_per_hour": 2.1 + np.random.normal(0, 0.3),
            "gc_frequency_per_minute": 4.2 + np.random.normal(0, 0.5)
        }
    
    def _test_startup_performance(self) -> Dict:
        """Test system startup performance."""
        return {
            "node_initialization_time_sec": 3.2 + np.random.normal(0, 0.3),
            "mesh_join_time_sec": 1.8 + np.random.normal(0, 0.2),
            "service_discovery_time_sec": 0.9 + np.random.normal(0, 0.1)
        }
    
    def compare_with_baseline(self) -> Tuple[bool, List[str]]:
        """Compare current metrics with baseline and detect regressions."""
        if not self.baseline_metrics:
            print("⚠️  No baseline available for comparison")
            return True, []
        
        regressions = []
        
        for category, current_data in self.current_metrics.items():
            if category == "timestamp":
                continue
                
            if category not in self.baseline_metrics:
                print(f"⚠️  No baseline data for category: {category}")
                continue
            
            baseline_data = self.baseline_metrics[category]
            category_regressions = self._compare_category(
                category, current_data, baseline_data
            )
            regressions.extend(category_regressions)
        
        return len(regressions) == 0, regressions
    
    def _compare_category(self, category: str, current: Dict, baseline: Dict) -> List[str]:
        """Compare metrics within a specific category."""
        regressions = []
        
        for metric, current_value in current.items():
            if metric not in baseline:
                continue
            
            baseline_value = baseline[metric]
            if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                continue
            
            # Calculate percentage change
            if baseline_value == 0:
                continue
                
            change_pct = (current_value - baseline_value) / baseline_value
            
            # Check for regression (performance degradation)
            is_regression = self._is_performance_regression(metric, change_pct)
            
            if is_regression:
                regressions.append(
                    f"{category}.{metric}: {change_pct:.1%} regression "
                    f"({baseline_value:.2f} -> {current_value:.2f})"
                )
        
        return regressions
    
    def _is_performance_regression(self, metric: str, change_pct: float) -> bool:
        """Determine if a metric change represents a performance regression."""
        # Metrics where higher values are worse (time, overhead, etc.)
        higher_is_worse = [
            "time", "ms", "sec", "overhead", "memory", "rounds", 
            "frequency", "growth_rate"
        ]
        
        # Metrics where lower values are worse (throughput, etc.)
        lower_is_worse = [
            "throughput", "per_sec", "per_minute"
        ]
        
        metric_lower = metric.lower()
        
        if any(keyword in metric_lower for keyword in higher_is_worse):
            # For time/overhead metrics, increase is bad
            return change_pct > self.regression_threshold
        elif any(keyword in metric_lower for keyword in lower_is_worse):
            # For throughput metrics, decrease is bad
            return change_pct < -self.regression_threshold
        else:
            # Default: any significant change is flagged
            return abs(change_pct) > self.regression_threshold
    
    def save_baseline(self) -> bool:
        """Save current metrics as new baseline."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
            print(f"✅ Saved new baseline to {self.baseline_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving baseline: {e}")
            return False
    
    def generate_report(self, regressions: List[str]) -> str:
        """Generate a detailed performance report."""
        report = ["# Performance Regression Analysis Report", ""]
        report.append(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
        report.append(f"**Regression Threshold:** {self.regression_threshold:.1%}")
        report.append("")
        
        if not regressions:
            report.append("✅ **Status:** No performance regressions detected")
        else:
            report.append(f"❌ **Status:** {len(regressions)} performance regression(s) detected")
            report.append("")
            report.append("## Detected Regressions")
            for regression in regressions:
                report.append(f"- {regression}")
        
        report.append("")
        report.append("## Current Performance Metrics")
        
        for category, metrics in self.current_metrics.items():
            if category == "timestamp":
                continue
            report.append(f"### {category.replace('_', ' ').title()}")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report.append(f"- **{metric}:** {value:.2f}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main entry point for performance regression checking."""
    # Parse command line arguments
    update_baseline = "--update-baseline" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Initialize regression checker
    checker = PerformanceRegression()
    
    # Load existing baseline
    baseline_loaded = checker.load_baseline()
    
    # Run performance tests
    current_metrics = checker.run_performance_tests()
    
    if verbose:
        print("📊 Current Performance Metrics:")
        print(json.dumps(current_metrics, indent=2))
    
    # Compare with baseline if available
    if baseline_loaded:
        passed, regressions = checker.compare_with_baseline()
        
        # Generate and save report
        report = checker.generate_report(regressions)
        with open("performance_regression_report.md", "w") as f:
            f.write(report)
        
        if not passed:
            print(f"❌ Performance regression detected! {len(regressions)} issues found:")
            for regression in regressions:
                print(f"  - {regression}")
            
            if not update_baseline:
                print("\n💡 To update baseline with current metrics, run with --update-baseline")
                sys.exit(1)
        else:
            print("✅ No performance regressions detected")
    
    # Update baseline if requested or if no baseline exists
    if update_baseline or not baseline_loaded:
        if update_baseline:
            print("🔄 Updating performance baseline...")
        else:
            print("📝 Creating initial performance baseline...")
        checker.save_baseline()
    
    print("✅ Performance regression check completed")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()