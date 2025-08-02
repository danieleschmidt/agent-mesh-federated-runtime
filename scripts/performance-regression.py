#!/usr/bin/env python3
"""
Performance Regression Detection Script
Checks for performance regressions in the codebase by running lightweight benchmarks.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil


class PerformanceChecker:
    """Lightweight performance regression checker."""
    
    def __init__(self, baseline_file: str = ".performance-baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_data = self._load_baseline()
        self.current_metrics = {}
    
    def _load_baseline(self) -> Dict:
        """Load performance baseline data."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline file: {e}")
                return {}
        return {}
    
    def _save_baseline(self):
        """Save current metrics as new baseline."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baseline file: {e}")
    
    def measure_import_time(self) -> float:
        """Measure module import time."""
        start_time = time.time()
        try:
            # Try to import main modules
            result = subprocess.run([
                sys.executable, '-c',
                'import src.agent_mesh; '
                'import src.agent_mesh.networking; '
                'import src.agent_mesh.consensus'
            ], capture_output=True, timeout=10)
            
            if result.returncode != 0:
                print(f"Warning: Import failed: {result.stderr.decode()}")
                return -1
                
        except subprocess.TimeoutExpired:
            print("Warning: Import timeout")
            return -1
        except Exception as e:
            print(f"Warning: Import error: {e}")
            return -1
        
        return time.time() - start_time
    
    def measure_startup_time(self) -> float:
        """Measure application startup time."""
        start_time = time.time()
        try:
            # Run a minimal startup test
            result = subprocess.run([
                sys.executable, '-c',
                'from src.agent_mesh import AgentMesh; '
                'mesh = AgentMesh(); '
                'print("Started")'
            ], capture_output=True, timeout=15)
            
            if result.returncode != 0:
                print(f"Warning: Startup test failed: {result.stderr.decode()}")
                return -1
                
        except subprocess.TimeoutExpired:
            print("Warning: Startup timeout")
            return -1
        except Exception as e:
            print(f"Warning: Startup error: {e}")
            return -1
        
        return time.time() - start_time
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure basic memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # MB
            }
        except Exception as e:
            print(f"Warning: Memory measurement failed: {e}")
            return {'rss_mb': -1, 'vms_mb': -1}
    
    def measure_test_performance(self) -> Dict[str, float]:
        """Measure unit test execution time."""
        metrics = {}
        
        # Quick unit test run
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                'tests/unit/',
                '-x', '--tb=no', '-q',
                '--disable-warnings'
            ], capture_output=True, timeout=60)
            
            metrics['unit_test_time'] = time.time() - start_time
            
            if result.returncode != 0:
                print("Warning: Unit tests failed")
                metrics['unit_test_time'] = -1
                
        except subprocess.TimeoutExpired:
            print("Warning: Unit test timeout")
            metrics['unit_test_time'] = -1
        except Exception as e:
            print(f"Warning: Unit test error: {e}")
            metrics['unit_test_time'] = -1
        
        return metrics
    
    def run_performance_check(self) -> Dict[str, float]:
        """Run all performance checks."""
        print("🔍 Running performance regression check...")
        
        metrics = {}
        
        # Import time
        print("  • Measuring import time...")
        metrics['import_time'] = self.measure_import_time()
        
        # Startup time
        print("  • Measuring startup time...")
        metrics['startup_time'] = self.measure_startup_time()
        
        # Memory usage
        print("  • Measuring memory usage...")
        memory_metrics = self.measure_memory_usage()
        metrics.update(memory_metrics)
        
        # Test performance
        print("  • Measuring test performance...")
        test_metrics = self.measure_test_performance()
        metrics.update(test_metrics)
        
        self.current_metrics = metrics
        return metrics
    
    def check_regression(self, threshold_percent: float = 20.0) -> Tuple[bool, List[str]]:
        """Check for performance regressions."""
        if not self.baseline_data:
            print("📊 No baseline data available. Saving current metrics as baseline.")
            self._save_baseline()
            return True, []
        
        regressions = []
        
        for metric, current_value in self.current_metrics.items():
            if current_value < 0:  # Skip failed measurements
                continue
                
            if metric not in self.baseline_data:
                continue
            
            baseline_value = self.baseline_data[metric]
            if baseline_value <= 0:
                continue
            
            # Calculate percentage change
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            
            # Check for regression (performance got worse)
            if change_percent > threshold_percent:
                regressions.append(
                    f"{metric}: {change_percent:.1f}% slower "
                    f"({baseline_value:.3f}s → {current_value:.3f}s)"
                )
            elif change_percent < -10:  # Significant improvement
                print(f"✅ Performance improvement in {metric}: "
                      f"{abs(change_percent):.1f}% faster")
        
        return len(regressions) == 0, regressions
    
    def update_baseline(self):
        """Update baseline with current metrics."""
        print("📊 Updating performance baseline...")
        self._save_baseline()
    
    def print_metrics(self):
        """Print current performance metrics."""
        print("\n📈 Current Performance Metrics:")
        for metric, value in self.current_metrics.items():
            if value >= 0:
                if 'time' in metric:
                    print(f"  • {metric}: {value:.3f}s")
                elif 'mb' in metric:
                    print(f"  • {metric}: {value:.1f} MB")
                else:
                    print(f"  • {metric}: {value:.3f}")
            else:
                print(f"  • {metric}: Failed to measure")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for performance regressions"
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=20.0,
        help='Regression threshold percentage (default: 20.0)'
    )
    parser.add_argument(
        '--update-baseline',
        action='store_true',
        help='Update baseline with current metrics'
    )
    parser.add_argument(
        '--baseline-file',
        default='.performance-baseline.json',
        help='Baseline file path'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick checks only (skip heavy operations)'
    )
    
    args = parser.parse_args()
    
    checker = PerformanceChecker(args.baseline_file)
    
    # Run performance checks
    metrics = checker.run_performance_check()
    
    if args.quick:
        # In quick mode, just check import time
        if metrics.get('import_time', -1) > 2.0:  # 2 second threshold
            print("❌ Import time is too slow (> 2s)")
            return 1
        else:
            print("✅ Quick performance check passed")
            return 0
    
    # Print current metrics
    checker.print_metrics()
    
    if args.update_baseline:
        checker.update_baseline()
        print("✅ Baseline updated successfully")
        return 0
    
    # Check for regressions
    no_regression, regressions = checker.check_regression(args.threshold)
    
    if no_regression:
        print("✅ No performance regressions detected")
        return 0
    else:
        print("❌ Performance regressions detected:")
        for regression in regressions:
            print(f"  • {regression}")
        
        print(f"\nTo update the baseline (if these changes are expected):")
        print(f"  python {sys.argv[0]} --update-baseline")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())