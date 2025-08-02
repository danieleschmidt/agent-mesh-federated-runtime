#!/usr/bin/env python3
"""
SDLC Configuration Validation Script
Validates Terragon SDLC configuration files for correctness and completeness.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class SDLCConfigValidator:
    """Validates SDLC configuration files."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_file(self, config_path: Path) -> bool:
        """Validate a single SDLC configuration file."""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            if not config:
                self.errors.append(f"Empty configuration file: {config_path}")
                return False
            
            # Validate required sections
            self._validate_required_sections(config, config_path)
            
            # Validate metadata
            if 'metadata' in config:
                self._validate_metadata(config['metadata'], config_path)
            
            # Validate quality gates
            if 'quality_gates' in config:
                self._validate_quality_gates(config['quality_gates'], config_path)
            
            # Validate testing configuration
            if 'testing' in config:
                self._validate_testing_config(config['testing'], config_path)
            
            # Validate security configuration
            if 'security' in config:
                self._validate_security_config(config['security'], config_path)
            
            # Validate monitoring configuration
            if 'monitoring' in config:
                self._validate_monitoring_config(config['monitoring'], config_path)
            
            return len(self.errors) == 0
            
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in {config_path}: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Error validating {config_path}: {e}")
            return False
    
    def _validate_required_sections(self, config: Dict[str, Any], config_path: Path):
        """Validate required configuration sections."""
        required_sections = [
            'metadata',
            'development',
            'quality_gates',
            'testing',
            'security',
            'monitoring'
        ]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(
                    f"Missing required section '{section}' in {config_path}"
                )
    
    def _validate_metadata(self, metadata: Dict[str, Any], config_path: Path):
        """Validate metadata section."""
        required_fields = ['name', 'version', 'terragon_sdlc_version', 'team']
        
        for field in required_fields:
            if field not in metadata:
                self.errors.append(
                    f"Missing required metadata field '{field}' in {config_path}"
                )
        
        # Validate version format
        if 'version' in metadata:
            version = metadata['version']
            if not isinstance(version, str) or not version.count('.') >= 1:
                self.errors.append(
                    f"Invalid version format '{version}' in {config_path}. "
                    "Expected semantic versioning (e.g., '1.0.0')"
                )
    
    def _validate_quality_gates(self, quality_gates: Dict[str, Any], config_path: Path):
        """Validate quality gates configuration."""
        if 'pre_commit' in quality_gates:
            pre_commit = quality_gates['pre_commit']
            if 'enabled' in pre_commit and pre_commit['enabled']:
                if 'hooks' not in pre_commit:
                    self.warnings.append(
                        f"Pre-commit enabled but no hooks specified in {config_path}"
                    )
        
        if 'pull_request' in quality_gates:
            pr_config = quality_gates['pull_request']
            if 'thresholds' in pr_config:
                thresholds = pr_config['thresholds']
                
                # Validate test coverage threshold
                if 'test_coverage' in thresholds:
                    coverage = thresholds['test_coverage']
                    if not isinstance(coverage, int) or not 0 <= coverage <= 100:
                        self.errors.append(
                            f"Invalid test coverage threshold '{coverage}' in {config_path}. "
                            "Must be an integer between 0 and 100"
                        )
    
    def _validate_testing_config(self, testing: Dict[str, Any], config_path: Path):
        """Validate testing configuration."""
        if 'strategy' in testing:
            strategy = testing['strategy']
            
            test_types = ['unit_tests', 'integration_tests', 'e2e_tests']
            for test_type in test_types:
                if test_type in strategy:
                    test_config = strategy[test_type]
                    
                    # Validate coverage target
                    if 'coverage_target' in test_config:
                        coverage = test_config['coverage_target']
                        if not isinstance(coverage, int) or not 0 <= coverage <= 100:
                            self.errors.append(
                                f"Invalid coverage target '{coverage}' for {test_type} "
                                f"in {config_path}"
                            )
                    
                    # Validate timeout
                    if 'timeout' in test_config:
                        timeout = test_config['timeout']
                        if not isinstance(timeout, int) or timeout <= 0:
                            self.errors.append(
                                f"Invalid timeout '{timeout}' for {test_type} "
                                f"in {config_path}"
                            )
    
    def _validate_security_config(self, security: Dict[str, Any], config_path: Path):
        """Validate security configuration."""
        security_sections = ['sast', 'dependency_scanning', 'container_scanning']
        
        for section in security_sections:
            if section in security:
                sec_config = security[section]
                
                # Validate severity threshold
                if 'severity_threshold' in sec_config:
                    threshold = sec_config['severity_threshold']
                    valid_thresholds = ['low', 'medium', 'high', 'critical']
                    
                    if threshold not in valid_thresholds:
                        self.errors.append(
                            f"Invalid severity threshold '{threshold}' for {section} "
                            f"in {config_path}. Must be one of: {valid_thresholds}"
                        )
    
    def _validate_monitoring_config(self, monitoring: Dict[str, Any], config_path: Path):
        """Validate monitoring configuration."""
        if 'metrics' in monitoring:
            metrics = monitoring['metrics']
            
            if 'prometheus' in metrics:
                prom_config = metrics['prometheus']
                
                # Validate port
                if 'port' in prom_config:
                    port = prom_config['port']
                    if not isinstance(port, int) or not 1024 <= port <= 65535:
                        self.errors.append(
                            f"Invalid Prometheus port '{port}' in {config_path}. "
                            "Must be an integer between 1024 and 65535"
                        )
            
            # Validate custom metrics
            if 'custom_metrics' in metrics:
                custom_metrics = metrics['custom_metrics']
                if isinstance(custom_metrics, list):
                    for i, metric in enumerate(custom_metrics):
                        required_fields = ['name', 'type', 'description']
                        for field in required_fields:
                            if field not in metric:
                                self.errors.append(
                                    f"Missing field '{field}' in custom metric {i} "
                                    f"in {config_path}"
                                )
                        
                        # Validate metric type
                        if 'type' in metric:
                            metric_type = metric['type']
                            valid_types = ['counter', 'gauge', 'histogram', 'summary']
                            if metric_type not in valid_types:
                                self.errors.append(
                                    f"Invalid metric type '{metric_type}' in custom metric {i} "
                                    f"in {config_path}. Must be one of: {valid_types}"
                                )
    
    def print_results(self):
        """Print validation results."""
        if self.errors:
            print("❌ SDLC Configuration Validation FAILED")
            print("\nErrors:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        if not self.errors and not self.warnings:
            print("✅ SDLC Configuration validation passed")
        elif not self.errors:
            print("✅ SDLC Configuration validation passed with warnings")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Terragon SDLC configuration files"
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Configuration files to validate'
    )
    parser.add_argument(
        '--directory',
        default='.terragon',
        help='Directory containing SDLC configuration files'
    )
    
    args = parser.parse_args()
    
    validator = SDLCConfigValidator()
    
    if args.files:
        # Validate specific files
        config_files = [Path(f) for f in args.files]
    else:
        # Validate all YAML files in the directory
        config_dir = Path(args.directory)
        if not config_dir.exists():
            print(f"❌ Configuration directory '{config_dir}' does not exist")
            return 1
        
        config_files = list(config_dir.glob('*.yaml')) + list(config_dir.glob('*.yml'))
        
        if not config_files:
            print(f"❌ No configuration files found in '{config_dir}'")
            return 1
    
    success = True
    for config_file in config_files:
        if not config_file.exists():
            print(f"❌ Configuration file '{config_file}' does not exist")
            success = False
            continue
        
        file_success = validator.validate_file(config_file)
        success = success and file_success
    
    validator.print_results()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())