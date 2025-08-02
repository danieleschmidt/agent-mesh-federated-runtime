#!/usr/bin/env python3
"""
Comprehensive metrics collection system for Agent Mesh Federated Runtime.
Collects metrics from various sources and updates project metrics dashboard.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import subprocess
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self.metrics_data = {}
        self.collectors = {}
        
        # Initialize collectors
        self._initialize_collectors()
    
    def _load_config(self) -> Dict:
        """Load metrics configuration."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _initialize_collectors(self):
        """Initialize metric collectors based on configuration."""
        integrations = self.config.get('automation', {}).get('integrations', {})
        
        if integrations.get('github', {}).get('enabled'):
            self.collectors['github'] = GitHubMetricsCollector()
        
        if integrations.get('sonarcloud', {}).get('enabled'):
            self.collectors['sonarcloud'] = SonarCloudCollector(
                integrations['sonarcloud'].get('project_key')
            )
        
        if integrations.get('prometheus', {}).get('enabled'):
            self.collectors['prometheus'] = PrometheusCollector(
                integrations['prometheus'].get('endpoint')
            )
        
        if integrations.get('sentry', {}).get('enabled'):
            self.collectors['sentry'] = SentryCollector(
                integrations['sentry'].get('project')
            )
    
    async def collect_all_metrics(self) -> Dict:
        """Collect metrics from all enabled sources."""
        logger.info("Starting comprehensive metrics collection...")
        
        collected_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'collection_duration': 0,
            'sources': {}
        }
        
        start_time = time.time()
        
        # Collect from each source
        for name, collector in self.collectors.items():
            try:
                logger.info(f"Collecting metrics from {name}...")
                source_start = time.time()
                
                metrics = await collector.collect()
                collected_metrics['sources'][name] = {
                    'metrics': metrics,
                    'collection_time': time.time() - source_start,
                    'status': 'success'
                }
                
                logger.info(f"Successfully collected {len(metrics)} metrics from {name}")
                
            except Exception as e:
                logger.error(f"Failed to collect metrics from {name}: {e}")
                collected_metrics['sources'][name] = {
                    'metrics': {},
                    'collection_time': time.time() - source_start,
                    'status': 'error',
                    'error': str(e)
                }
        
        collected_metrics['collection_duration'] = time.time() - start_time
        
        # Aggregate and analyze metrics
        aggregated = self._aggregate_metrics(collected_metrics)
        
        # Update project metrics file
        await self._update_project_metrics(aggregated)
        
        logger.info(f"Metrics collection completed in {collected_metrics['collection_duration']:.2f}s")
        return collected_metrics
    
    def _aggregate_metrics(self, collected: Dict) -> Dict:
        """Aggregate metrics from different sources."""
        aggregated = {
            'code_quality': {},
            'security': {},
            'performance': {},
            'reliability': {},
            'development': {},
            'business': {}
        }
        
        # Aggregate GitHub metrics
        if 'github' in collected['sources'] and collected['sources']['github']['status'] == 'success':
            github_metrics = collected['sources']['github']['metrics']
            
            aggregated['development'].update({
                'commits_per_week': github_metrics.get('commits_last_week', 0),
                'pull_requests_per_week': github_metrics.get('prs_last_week', 0),
                'active_contributors': github_metrics.get('contributors_count', 0),
                'open_issues': github_metrics.get('open_issues', 0),
                'closed_issues': github_metrics.get('closed_issues_last_week', 0)
            })
        
        # Aggregate SonarCloud metrics
        if 'sonarcloud' in collected['sources'] and collected['sources']['sonarcloud']['status'] == 'success':
            sonar_metrics = collected['sources']['sonarcloud']['metrics']
            
            aggregated['code_quality'].update({
                'coverage': sonar_metrics.get('coverage', 0),
                'duplication': sonar_metrics.get('duplicated_lines_density', 0),
                'maintainability_rating': sonar_metrics.get('sqale_rating', 'E'),
                'reliability_rating': sonar_metrics.get('reliability_rating', 'E'),
                'security_rating': sonar_metrics.get('security_rating', 'E'),
                'technical_debt': sonar_metrics.get('sqale_index', 0)
            })
            
            aggregated['security'].update({
                'vulnerabilities': sonar_metrics.get('vulnerabilities', 0),
                'security_hotspots': sonar_metrics.get('security_hotspots', 0)
            })
        
        # Aggregate Prometheus metrics
        if 'prometheus' in collected['sources'] and collected['sources']['prometheus']['status'] == 'success':
            prom_metrics = collected['sources']['prometheus']['metrics']
            
            aggregated['performance'].update({
                'response_time_p95': prom_metrics.get('http_request_duration_p95', 0),
                'throughput_rps': prom_metrics.get('http_requests_per_second', 0),
                'error_rate': prom_metrics.get('http_error_rate', 0)
            })
            
            aggregated['reliability'].update({
                'uptime': prom_metrics.get('uptime_percentage', 0),
                'availability': prom_metrics.get('availability', 0)
            })
        
        return aggregated
    
    async def _update_project_metrics(self, aggregated: Dict):
        """Update the project metrics file with new data."""
        try:
            # Load current metrics
            current_metrics = self.config.copy()
            
            # Update measurements with new data
            for category, metrics in aggregated.items():
                if category in current_metrics.get('metrics', {}):
                    measurements = current_metrics['metrics'][category].get('measurements', {})
                    
                    for metric_name, value in metrics.items():
                        if metric_name in measurements:
                            if isinstance(measurements[metric_name], dict):
                                measurements[metric_name]['current'] = value
                                measurements[metric_name]['last_updated'] = datetime.utcnow().isoformat()
                            else:
                                measurements[metric_name] = value
            
            # Add collection metadata
            current_metrics['last_collection'] = {
                'timestamp': datetime.utcnow().isoformat(),
                'sources_collected': list(self.collectors.keys()),
                'collection_success': True
            }
            
            # Save updated metrics
            with open(self.config_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            
            logger.info(f"Updated project metrics file: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to update project metrics: {e}")
    
    async def generate_report(self, format_type: str = 'json') -> str:
        """Generate metrics report in specified format."""
        if format_type == 'json':
            return json.dumps(self.config, indent=2)
        elif format_type == 'markdown':
            return self._generate_markdown_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown metrics report."""
        report = []
        report.append("# Agent Mesh Project Metrics Report")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        metrics = self.config.get('metrics', {})
        
        for category, data in metrics.items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append(f"{data.get('description', '')}")
            report.append("")
            
            measurements = data.get('measurements', {})
            if measurements:
                report.append("| Metric | Current | Target | Trend |")
                report.append("|--------|---------|--------|--------|")
                
                for metric, values in measurements.items():
                    if isinstance(values, dict):
                        current = values.get('current', 'N/A')
                        target = values.get('target', 'N/A')
                        trend = values.get('trend', 'N/A')
                        report.append(f"| {metric.replace('_', ' ').title()} | {current} | {target} | {trend} |")
                
                report.append("")
        
        return "\n".join(report)


class GitHubMetricsCollector:
    """Collect metrics from GitHub API."""
    
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        self.repo = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/agent-mesh-federated-runtime')
        self.api_url = 'https://api.github.com'
    
    async def collect(self) -> Dict:
        """Collect GitHub metrics."""
        headers = {'Authorization': f'token {self.token}'} if self.token else {}
        
        metrics = {}
        
        try:
            # Repository info
            repo_data = await self._api_request(f'/repos/{self.repo}', headers)
            metrics.update({
                'stars': repo_data.get('stargazers_count', 0),
                'forks': repo_data.get('forks_count', 0),
                'open_issues': repo_data.get('open_issues_count', 0),
                'watchers': repo_data.get('subscribers_count', 0)
            })
            
            # Commits (last week)
            since_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            commits_data = await self._api_request(f'/repos/{self.repo}/commits?since={since_date}', headers)
            metrics['commits_last_week'] = len(commits_data) if isinstance(commits_data, list) else 0
            
            # Pull requests (last week)
            prs_data = await self._api_request(f'/repos/{self.repo}/pulls?state=all&since={since_date}', headers)
            metrics['prs_last_week'] = len(prs_data) if isinstance(prs_data, list) else 0
            
            # Contributors
            contributors_data = await self._api_request(f'/repos/{self.repo}/contributors', headers)
            metrics['contributors_count'] = len(contributors_data) if isinstance(contributors_data, list) else 0
            
            # Issues closed last week
            issues_data = await self._api_request(f'/repos/{self.repo}/issues?state=closed&since={since_date}', headers)
            metrics['closed_issues_last_week'] = len(issues_data) if isinstance(issues_data, list) else 0
            
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _api_request(self, endpoint: str, headers: Dict) -> Dict:
        """Make GitHub API request."""
        url = f"{self.api_url}{endpoint}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()


class SonarCloudCollector:
    """Collect metrics from SonarCloud API."""
    
    def __init__(self, project_key: str):
        self.project_key = project_key
        self.token = os.getenv('SONAR_TOKEN')
        self.api_url = 'https://sonarcloud.io/api'
    
    async def collect(self) -> Dict:
        """Collect SonarCloud metrics."""
        if not self.token or not self.project_key:
            return {'error': 'SonarCloud token or project key not configured'}
        
        headers = {'Authorization': f'Bearer {self.token}'}
        metrics = {}
        
        try:
            # Project measures
            measures_response = await self._api_request(
                f'/measures/component?component={self.project_key}&metricKeys=coverage,duplicated_lines_density,sqale_rating,reliability_rating,security_rating,vulnerabilities,security_hotspots,sqale_index',
                headers
            )
            
            component = measures_response.get('component', {})
            measures = component.get('measures', [])
            
            for measure in measures:
                metric_key = measure.get('metric')
                value = measure.get('value', 0)
                
                # Convert value based on metric type
                if metric_key in ['coverage', 'duplicated_lines_density']:
                    metrics[metric_key] = float(value)
                elif metric_key in ['sqale_rating', 'reliability_rating', 'security_rating']:
                    metrics[metric_key] = value
                else:
                    metrics[metric_key] = int(float(value))
            
        except Exception as e:
            logger.error(f"SonarCloud API error: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _api_request(self, endpoint: str, headers: Dict) -> Dict:
        """Make SonarCloud API request."""
        url = f"{self.api_url}{endpoint}"
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()


class PrometheusCollector:
    """Collect metrics from Prometheus."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint or 'http://localhost:9090'
    
    async def collect(self) -> Dict:
        """Collect Prometheus metrics."""
        metrics = {}
        
        try:
            # HTTP request duration (95th percentile)
            p95_response = await self._query('histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))')
            metrics['http_request_duration_p95'] = self._extract_value(p95_response)
            
            # Requests per second
            rps_response = await self._query('rate(http_requests_total[5m])')
            metrics['http_requests_per_second'] = self._extract_value(rps_response)
            
            # Error rate
            error_response = await self._query('rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])')
            metrics['http_error_rate'] = self._extract_value(error_response) * 100
            
            # Uptime
            uptime_response = await self._query('up')
            metrics['uptime_percentage'] = self._extract_value(uptime_response) * 100
            
            # Application-specific metrics
            # Consensus metrics
            consensus_latency = await self._query('agent_mesh_consensus_duration_seconds_p95')
            metrics['consensus_latency_p95'] = self._extract_value(consensus_latency) * 1000  # Convert to ms
            
            # P2P network metrics
            p2p_latency = await self._query('agent_mesh_p2p_latency_seconds_p95')
            metrics['p2p_latency_p95'] = self._extract_value(p2p_latency) * 1000
            
            # Federated learning metrics
            fl_accuracy = await self._query('agent_mesh_federated_model_accuracy')
            metrics['federated_learning_accuracy'] = self._extract_value(fl_accuracy)
            
        except Exception as e:
            logger.error(f"Prometheus query error: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _query(self, query: str) -> Dict:
        """Execute Prometheus query."""
        url = f"{self.endpoint}/api/v1/query"
        params = {'query': query}
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    
    def _extract_value(self, response: Dict) -> float:
        """Extract numeric value from Prometheus response."""
        try:
            data = response.get('data', {})
            result = data.get('result', [])
            if result and len(result) > 0:
                value = result[0].get('value', [None, '0'])
                return float(value[1])
            return 0.0
        except (ValueError, TypeError, IndexError):
            return 0.0


class SentryCollector:
    """Collect metrics from Sentry."""
    
    def __init__(self, project: str):
        self.project = project
        self.token = os.getenv('SENTRY_AUTH_TOKEN')
        self.org = os.getenv('SENTRY_ORG')
        self.api_url = 'https://sentry.io/api/0'
    
    async def collect(self) -> Dict:
        """Collect Sentry metrics."""
        if not self.token or not self.project or not self.org:
            return {'error': 'Sentry configuration incomplete'}
        
        headers = {'Authorization': f'Bearer {self.token}'}
        metrics = {}
        
        try:
            # Project stats
            stats_response = await self._api_request(
                f'/projects/{self.org}/{self.project}/stats/',
                headers,
                params={'resolution': '1d', 'since': (datetime.utcnow() - timedelta(days=7)).timestamp()}
            )
            
            # Extract error counts and performance data
            if isinstance(stats_response, list):
                total_events = sum(point[1] for point in stats_response)
                metrics['error_events_last_week'] = total_events
            
            # Issues
            issues_response = await self._api_request(
                f'/projects/{self.org}/{self.project}/issues/',
                headers,
                params={'statsPeriod': '7d'}
            )
            
            if isinstance(issues_response, list):
                metrics['open_issues'] = len([issue for issue in issues_response if issue.get('status') == 'unresolved'])
                metrics['resolved_issues'] = len([issue for issue in issues_response if issue.get('status') == 'resolved'])
            
        except Exception as e:
            logger.error(f"Sentry API error: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    async def _api_request(self, endpoint: str, headers: Dict, params: Dict = None) -> Dict:
        """Make Sentry API request."""
        url = f"{self.api_url}{endpoint}"
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--output', help='Output file for collected metrics')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize collector
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    collector = MetricsCollector(config_path)
    
    try:
        # Collect metrics
        results = await collector.collect_all_metrics()
        
        # Generate report
        report = await collector.generate_report(args.format)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report written to {args.output}")
        else:
            print(report)
        
        # Summary
        success_count = sum(1 for source in results['sources'].values() 
                          if source['status'] == 'success')
        total_sources = len(results['sources'])
        
        logger.info(f"Collection completed: {success_count}/{total_sources} sources successful")
        
        if success_count < total_sources:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())