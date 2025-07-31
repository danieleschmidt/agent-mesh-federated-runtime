#!/usr/bin/env python3
"""Advanced health check for Agent Mesh Federated Runtime."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
import click
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health check for mesh nodes."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        self.results = {}

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load health check configuration."""
        default_config = {
            "endpoints": {
                "api": "http://localhost:8000",
                "metrics": "http://localhost:9090",
                "p2p": "http://localhost:4001"
            },
            "thresholds": {
                "response_time_ms": 1000,
                "cpu_usage_percent": 80,
                "memory_usage_percent": 85,
                "peer_count_min": 1
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config

    async def check_api_health(self) -> Dict:
        """Check API endpoint health."""
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['endpoints']['api']}/health") as resp:
                    response_time = (time.time() - start_time) * 1000
                    status = await resp.json()
                    
                    return {
                        "status": "healthy" if resp.status == 200 else "unhealthy",
                        "response_time_ms": response_time,
                        "details": status
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_metrics(self) -> Dict:
        """Check Prometheus metrics."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.config['endpoints']['metrics']}/metrics") as resp:
                    metrics_text = await resp.text()
                    
                    # Parse key metrics
                    metrics = {}
                    for family in text_string_to_metric_families(metrics_text):
                        if family.name in ["cpu_usage", "memory_usage", "peer_count"]:
                            for sample in family.samples:
                                metrics[sample.name] = sample.value
                    
                    # Check thresholds
                    issues = []
                    cpu = metrics.get("cpu_usage", 0)
                    memory = metrics.get("memory_usage", 0)
                    peers = metrics.get("peer_count", 0)
                    
                    if cpu > self.config["thresholds"]["cpu_usage_percent"]:
                        issues.append(f"High CPU usage: {cpu}%")
                    if memory > self.config["thresholds"]["memory_usage_percent"]:
                        issues.append(f"High memory usage: {memory}%")
                    if peers < self.config["thresholds"]["peer_count_min"]:
                        issues.append(f"Low peer count: {peers}")
                    
                    return {
                        "status": "healthy" if not issues else "warning",
                        "metrics": metrics,
                        "issues": issues
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def run_all_checks(self) -> Dict:
        """Run all health checks."""
        logger.info("Starting comprehensive health check...")
        
        checks = {
            "api": self.check_api_health(),
            "metrics": self.check_metrics(),
        }
        
        # Run checks concurrently
        results = {}
        for name, check_coro in checks.items():
            try:
                results[name] = await check_coro
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        
        # Overall health status
        overall_status = "healthy"
        for check_result in results.values():
            if check_result["status"] == "unhealthy":
                overall_status = "unhealthy"
                break
            elif check_result["status"] == "warning" and overall_status == "healthy":
                overall_status = "warning"
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "checks": results
        }


@click.command()
@click.option("--config", type=click.Path(exists=True), help="Health check configuration file")
@click.option("--output", type=click.Path(), help="Output file for results")
@click.option("--format", type=click.Choice(["json", "text"]), default="text", help="Output format")
def main(config: Optional[str], output: Optional[str], format: str):
    """Run comprehensive health check."""
    checker = HealthChecker(Path(config) if config else None)
    
    async def run():
        results = await checker.run_all_checks()
        
        if format == "json":
            output_text = json.dumps(results, indent=2)
        else:
            output_text = format_text_output(results)
        
        if output:
            with open(output, "w") as f:
                f.write(output_text)
        else:
            print(output_text)
        
        # Exit with error code if unhealthy
        if results["overall_status"] == "unhealthy":
            exit(1)
    
    asyncio.run(run())


def format_text_output(results: Dict) -> str:
    """Format results as human-readable text."""
    lines = [
        f"Health Check Results - {results['overall_status'].upper()}",
        "=" * 50,
        ""
    ]
    
    for check_name, check_result in results["checks"].items():
        lines.append(f"{check_name.upper()}: {check_result['status']}")
        if "error" in check_result:
            lines.append(f"  Error: {check_result['error']}")
        if "issues" in check_result:
            for issue in check_result["issues"]:
                lines.append(f"  Issue: {issue}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()