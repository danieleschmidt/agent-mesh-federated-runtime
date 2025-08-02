#!/usr/bin/env python3
"""
Integration Health Check Script
Validates the complete Terragon SDLC integration and reports on system health.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml


class HealthCheckResult:
    """Represents the result of a health check."""
    
    def __init__(self, name: str, status: str, message: str, details: Dict = None):
        self.name = name
        self.status = status  # 'pass', 'warn', 'fail'
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()


class IntegrationHealthChecker:
    """Comprehensive health checker for SDLC integration."""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path.cwd()
        self.results = []
        
    def add_result(self, name: str, status: str, message: str, details: Dict = None):
        """Add a health check result."""
        result = HealthCheckResult(name, status, message, details)
        self.results.append(result)
        
        # Print immediate feedback
        status_emoji = {
            'pass': '✅',
            'warn': '⚠️',
            'fail': '❌'
        }
        print(f"{status_emoji.get(status, '❓')} {name}: {message}")
    
    async def check_development_environment(self):
        """Check development environment setup."""
        print("\n🔧 Checking Development Environment...")
        
        # Check Python version
        try:
            result = subprocess.run([sys.executable, '--version'], 
                                  capture_output=True, text=True)
            python_version = result.stdout.strip()
            
            if '3.11' in python_version:
                self.add_result("Python Version", "pass", python_version)
            else:
                self.add_result("Python Version", "warn", 
                              f"Expected Python 3.11, found {python_version}")
        except Exception as e:
            self.add_result("Python Version", "fail", f"Cannot check Python version: {e}")
        
        # Check required packages
        required_packages = ['pytest', 'black', 'flake8', 'mypy', 'bandit']
        for package in required_packages:
            try:
                subprocess.run([sys.executable, '-c', f'import {package}'], 
                             check=True, capture_output=True)
                self.add_result(f"Package {package}", "pass", "Installed")
            except subprocess.CalledProcessError:
                self.add_result(f"Package {package}", "fail", "Not installed")
        
        # Check Docker availability
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.add_result("Docker", "pass", result.stdout.strip())
            else:
                self.add_result("Docker", "fail", "Docker command failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.add_result("Docker", "fail", "Docker not available")
        
        # Check pre-commit installation
        if (self.repo_path / '.pre-commit-config.yaml').exists():
            try:
                result = subprocess.run(['pre-commit', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    self.add_result("Pre-commit", "pass", "Installed and configured")
                else:
                    self.add_result("Pre-commit", "warn", "Configured but not installed")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.add_result("Pre-commit", "warn", "Configuration exists but pre-commit not installed")
        else:
            self.add_result("Pre-commit", "fail", "No pre-commit configuration found")
    
    async def check_configuration_files(self):
        """Check SDLC configuration files."""
        print("\n📄 Checking Configuration Files...")
        
        # Check SDLC configuration
        sdlc_config = self.repo_path / '.terragon' / 'sdlc-config.yaml'
        if sdlc_config.exists():
            try:
                with open(sdlc_config) as f:
                    config = yaml.safe_load(f)
                
                required_sections = ['metadata', 'development', 'quality_gates', 
                                   'testing', 'security', 'monitoring']
                missing_sections = [s for s in required_sections if s not in config]
                
                if not missing_sections:
                    self.add_result("SDLC Configuration", "pass", "All required sections present")
                else:
                    self.add_result("SDLC Configuration", "warn", 
                                  f"Missing sections: {missing_sections}")
            except Exception as e:
                self.add_result("SDLC Configuration", "fail", f"Invalid configuration: {e}")
        else:
            self.add_result("SDLC Configuration", "fail", "No SDLC configuration found")
        
        # Check Docker configuration
        dockerfile = self.repo_path / 'Dockerfile'
        if dockerfile.exists():
            try:
                with open(dockerfile) as f:
                    content = f.read()
                
                required_instructions = ['FROM', 'WORKDIR', 'COPY', 'RUN']
                missing = [inst for inst in required_instructions 
                          if not any(line.strip().startswith(inst) for line in content.split('\n'))]
                
                if not missing:
                    self.add_result("Dockerfile", "pass", "Contains required instructions")
                else:
                    self.add_result("Dockerfile", "warn", f"Missing instructions: {missing}")
            except Exception as e:
                self.add_result("Dockerfile", "fail", f"Cannot read Dockerfile: {e}")
        else:
            self.add_result("Dockerfile", "fail", "No Dockerfile found")
        
        # Check GitHub Actions workflows
        workflows_dir = self.repo_path / '.github' / 'workflows'
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
            if workflow_files:
                self.add_result("GitHub Workflows", "pass", 
                              f"Found {len(workflow_files)} workflow files")
            else:
                self.add_result("GitHub Workflows", "warn", "No workflow files found")
        else:
            self.add_result("GitHub Workflows", "fail", "No workflows directory found")
    
    async def check_testing_infrastructure(self):
        """Check testing infrastructure."""
        print("\n🧪 Checking Testing Infrastructure...")
        
        # Check test directories
        test_dirs = ['tests/unit', 'tests/integration', 'tests/e2e']
        for test_dir in test_dirs:
            test_path = self.repo_path / test_dir
            if test_path.exists():
                test_files = list(test_path.glob('test_*.py'))
                if test_files:
                    self.add_result(f"Tests {test_dir}", "pass", 
                                  f"Found {len(test_files)} test files")
                else:
                    self.add_result(f"Tests {test_dir}", "warn", "Directory exists but no test files")
            else:
                self.add_result(f"Tests {test_dir}", "warn", f"Directory {test_dir} not found")
        
        # Check pytest configuration
        pytest_configs = ['pytest.ini', 'pyproject.toml', 'setup.cfg']
        found_config = False
        for config_file in pytest_configs:
            if (self.repo_path / config_file).exists():
                found_config = True
                self.add_result("Pytest Configuration", "pass", f"Found in {config_file}")
                break
        
        if not found_config:
            self.add_result("Pytest Configuration", "warn", "No pytest configuration found")
        
        # Test pytest execution
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/', '--collect-only', '-q'
            ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
            
            if result.returncode == 0:
                # Count collected tests
                lines = result.stdout.split('\n')
                collected_line = [line for line in lines if 'collected' in line and 'items' in line]
                if collected_line:
                    self.add_result("Test Collection", "pass", collected_line[0].strip())
                else:
                    self.add_result("Test Collection", "pass", "Tests collected successfully")
            else:
                self.add_result("Test Collection", "fail", f"Collection failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.add_result("Test Collection", "fail", "Test collection timed out")
        except Exception as e:
            self.add_result("Test Collection", "fail", f"Cannot run pytest: {e}")
    
    async def check_quality_gates(self):
        """Check quality gates and code quality tools."""
        print("\n🚦 Checking Quality Gates...")
        
        # Check code formatting (Black)
        try:
            result = subprocess.run([
                sys.executable, '-m', 'black', 
                '--check', '--diff', 'src/'
            ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
            
            if result.returncode == 0:
                self.add_result("Code Formatting", "pass", "Code is properly formatted")
            else:
                self.add_result("Code Formatting", "warn", "Code formatting issues found")
        except Exception as e:
            self.add_result("Code Formatting", "fail", f"Cannot check formatting: {e}")
        
        # Check linting (Flake8)
        try:
            result = subprocess.run([
                sys.executable, '-m', 'flake8', 
                'src/', '--count', '--statistics'
            ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
            
            if result.returncode == 0:
                self.add_result("Code Linting", "pass", "No linting issues")
            else:
                error_count = len(result.stdout.split('\n')) if result.stdout else 0
                self.add_result("Code Linting", "warn", f"Found {error_count} linting issues")
        except Exception as e:
            self.add_result("Code Linting", "fail", f"Cannot run linting: {e}")
        
        # Check security scanning (Bandit)
        try:
            result = subprocess.run([
                sys.executable, '-m', 'bandit', 
                '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
            
            if result.returncode == 0:
                self.add_result("Security Scanning", "pass", "No security issues found")
            else:
                try:
                    bandit_output = json.loads(result.stdout)
                    issue_count = len(bandit_output.get('results', []))
                    self.add_result("Security Scanning", "warn", 
                                  f"Found {issue_count} potential security issues")
                except json.JSONDecodeError:
                    self.add_result("Security Scanning", "warn", "Security issues found")
        except Exception as e:
            self.add_result("Security Scanning", "fail", f"Cannot run security scan: {e}")
    
    async def check_monitoring_setup(self):
        """Check monitoring and observability setup."""
        print("\n📊 Checking Monitoring Setup...")
        
        # Check monitoring configuration files
        monitoring_configs = [
            'monitoring/prometheus.yml',
            'monitoring/alertmanager.yml',
            'monitoring/grafana-dashboards.json'
        ]
        
        for config_file in monitoring_configs:
            config_path = self.repo_path / config_file
            if config_path.exists():
                self.add_result(f"Monitoring Config", "pass", f"{config_file} exists")
            else:
                self.add_result(f"Monitoring Config", "warn", f"{config_file} not found")
        
        # Check if Prometheus metrics endpoint is accessible (if running)
        try:
            response = requests.get('http://localhost:9090/metrics', timeout=5)
            if response.status_code == 200:
                self.add_result("Prometheus Metrics", "pass", "Metrics endpoint accessible")
            else:
                self.add_result("Prometheus Metrics", "warn", 
                              f"Metrics endpoint returned {response.status_code}")
        except requests.RequestException:
            self.add_result("Prometheus Metrics", "warn", 
                          "Prometheus not running (expected in development)")
        
        # Check logging configuration
        logging_configs = ['monitoring/logging/fluentd.conf', 'logging.yaml', 'logging.json']
        found_logging = False
        for config_file in logging_configs:
            if (self.repo_path / config_file).exists():
                found_logging = True
                self.add_result("Logging Configuration", "pass", f"Found {config_file}")
                break
        
        if not found_logging:
            self.add_result("Logging Configuration", "warn", "No logging configuration found")
    
    async def check_automation_scripts(self):
        """Check automation scripts and tools."""
        print("\n🤖 Checking Automation Scripts...")
        
        # Check required scripts
        required_scripts = [
            'scripts/repository-automation.py',
            'scripts/metrics-collector.py',
            'scripts/validate-sdlc-config.py',
            'scripts/performance-regression.py'
        ]
        
        for script in required_scripts:
            script_path = self.repo_path / script
            if script_path.exists():
                if script_path.is_file() and script_path.stat().st_mode & 0o111:
                    self.add_result(f"Script {script}", "pass", "Exists and executable")
                else:
                    self.add_result(f"Script {script}", "warn", "Exists but not executable")
            else:
                self.add_result(f"Script {script}", "fail", "Script not found")
        
        # Test script execution (dry run)
        test_scripts = [
            ('scripts/validate-sdlc-config.py', ['--help']),
            ('scripts/performance-regression.py', ['--help']),
        ]
        
        for script, args in test_scripts:
            script_path = self.repo_path / script
            if script_path.exists():
                try:
                    result = subprocess.run([
                        sys.executable, str(script_path)
                    ] + args, capture_output=True, text=True, timeout=10, cwd=self.repo_path)
                    
                    if result.returncode == 0:
                        self.add_result(f"Script Test {script}", "pass", "Executes successfully")
                    else:
                        self.add_result(f"Script Test {script}", "warn", 
                                      f"Returned code {result.returncode}")
                except Exception as e:
                    self.add_result(f"Script Test {script}", "fail", f"Execution failed: {e}")
    
    async def check_security_integration(self):
        """Check security integration."""
        print("\n🔒 Checking Security Integration...")
        
        # Check secrets detection baseline
        secrets_baseline = self.repo_path / '.secrets.baseline'
        if secrets_baseline.exists():
            self.add_result("Secrets Baseline", "pass", "Secrets detection baseline exists")
        else:
            self.add_result("Secrets Baseline", "warn", "No secrets detection baseline")
        
        # Check security policy files
        security_files = [
            'SECURITY.md',
            '.github/SECURITY.md',
            'docs/security/SECURITY_POLICY.md'
        ]
        
        found_security_policy = False
        for security_file in security_files:
            if (self.repo_path / security_file).exists():
                found_security_policy = True
                self.add_result("Security Policy", "pass", f"Found {security_file}")
                break
        
        if not found_security_policy:
            self.add_result("Security Policy", "warn", "No security policy documentation found")
        
        # Check dependency scanning
        try:
            result = subprocess.run([
                sys.executable, '-m', 'safety', 'check', '--json'
            ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
            
            if result.returncode == 0:
                safety_output = result.stdout.strip()
                if safety_output == '[]':
                    self.add_result("Dependency Security", "pass", "No known vulnerabilities")
                else:
                    vulnerabilities = json.loads(safety_output)
                    self.add_result("Dependency Security", "warn", 
                                  f"Found {len(vulnerabilities)} vulnerabilities")
            else:
                self.add_result("Dependency Security", "warn", "Safety check failed")
        except Exception as e:
            self.add_result("Dependency Security", "fail", f"Cannot run safety check: {e}")
    
    async def check_performance_setup(self):
        """Check performance monitoring and benchmarking setup."""
        print("\n⚡ Checking Performance Setup...")
        
        # Check performance test directory
        perf_test_dir = self.repo_path / 'tests' / 'performance'
        if perf_test_dir.exists():
            perf_files = list(perf_test_dir.glob('test_*.py'))
            if perf_files:
                self.add_result("Performance Tests", "pass", 
                              f"Found {len(perf_files)} performance test files")
            else:
                self.add_result("Performance Tests", "warn", 
                              "Performance test directory exists but no test files")
        else:
            self.add_result("Performance Tests", "warn", "No performance test directory")
        
        # Check performance baseline
        baseline_file = self.repo_path / '.performance-baseline.json'
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline = json.load(f)
                self.add_result("Performance Baseline", "pass", 
                              f"Baseline with {len(baseline)} metrics")
            except Exception as e:
                self.add_result("Performance Baseline", "warn", 
                              f"Baseline file exists but invalid: {e}")
        else:
            self.add_result("Performance Baseline", "warn", "No performance baseline found")
        
        # Check performance regression script
        perf_script = self.repo_path / 'scripts' / 'performance-regression.py'
        if perf_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(perf_script), '--quick'
                ], capture_output=True, text=True, timeout=30, cwd=self.repo_path)
                
                if result.returncode == 0:
                    self.add_result("Performance Check", "pass", "Performance check passed")
                else:
                    self.add_result("Performance Check", "warn", "Performance check issues")
            except Exception as e:
                self.add_result("Performance Check", "fail", f"Cannot run performance check: {e}")
    
    async def run_comprehensive_health_check(self):
        """Run all health checks."""
        print("🏥 Starting Comprehensive SDLC Integration Health Check...")
        print(f"Repository: {self.repo_path}")
        print("=" * 60)
        
        # Run all health checks
        await self.check_development_environment()
        await self.check_configuration_files()
        await self.check_testing_infrastructure()
        await self.check_quality_gates()
        await self.check_monitoring_setup()
        await self.check_automation_scripts()
        await self.check_security_integration()
        await self.check_performance_setup()
    
    def generate_report(self) -> Dict:
        """Generate health check report."""
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == 'pass'])
        warnings = len([r for r in self.results if r.status == 'warn'])
        failed = len([r for r in self.results if r.status == 'fail'])
        
        report = {
            'timestamp': time.time(),
            'repository': str(self.repo_path),
            'summary': {
                'total_checks': total_checks,
                'passed': passed,
                'warnings': warnings,
                'failed': failed,
                'health_score': round((passed / total_checks) * 100, 1) if total_checks > 0 else 0
            },
            'results': [
                {
                    'name': r.name,
                    'status': r.status,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp
                }
                for r in self.results
            ]
        }
        
        return report
    
    def print_summary(self):
        """Print health check summary."""
        report = self.generate_report()
        summary = report['summary']
        
        print("\n" + "=" * 60)
        print("🏥 SDLC Integration Health Check Summary")
        print("=" * 60)
        
        print(f"📊 Total Checks: {summary['total_checks']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"🎯 Health Score: {summary['health_score']}%")
        
        # Health status
        if summary['health_score'] >= 90:
            print("🟢 Overall Status: EXCELLENT")
        elif summary['health_score'] >= 80:
            print("🟡 Overall Status: GOOD")
        elif summary['health_score'] >= 70:
            print("🟠 Overall Status: FAIR")
        else:
            print("🔴 Overall Status: NEEDS ATTENTION")
        
        # Recommendations
        if summary['failed'] > 0:
            print(f"\n🔧 Recommendations:")
            failed_results = [r for r in self.results if r.status == 'fail']
            for result in failed_results[:5]:  # Show top 5 critical issues
                print(f"  • Fix: {result.name} - {result.message}")
        
        print("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive SDLC Integration Health Check"
    )
    parser.add_argument(
        '--repo-path',
        type=Path,
        default=Path.cwd(),
        help='Repository path to check'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output JSON report file'
    )
    parser.add_argument(
        '--component',
        choices=[
            'development', 'configuration', 'testing', 'quality',
            'monitoring', 'automation', 'security', 'performance'
        ],
        help='Check specific component only'
    )
    
    args = parser.parse_args()
    
    checker = IntegrationHealthChecker(args.repo_path)
    
    if args.component:
        # Run specific component check
        component_map = {
            'development': checker.check_development_environment,
            'configuration': checker.check_configuration_files,
            'testing': checker.check_testing_infrastructure,
            'quality': checker.check_quality_gates,
            'monitoring': checker.check_monitoring_setup,
            'automation': checker.check_automation_scripts,
            'security': checker.check_security_integration,
            'performance': checker.check_performance_setup
        }
        
        if args.component in component_map:
            await component_map[args.component]()
        else:
            print(f"Unknown component: {args.component}")
            return 1
    else:
        # Run comprehensive health check
        await checker.run_comprehensive_health_check()
    
    # Generate and save report
    report = checker.generate_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output}")
    
    # Print summary
    checker.print_summary()
    
    # Return exit code based on health score
    health_score = report['summary']['health_score']
    if health_score >= 80:
        return 0
    elif health_score >= 60:
        return 1
    else:
        return 2


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))