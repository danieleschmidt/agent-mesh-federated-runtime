#!/usr/bin/env python3
"""Performance optimization and profiling for Agent Mesh."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import psutil
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """Analyze and optimize mesh performance."""

    def __init__(self):
        self.metrics_history = []
        self.recommendations = []

    def analyze_system_resources(self) -> Dict:
        """Analyze current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
                "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "usage_percent": (disk.used / disk.total) * 100
            }
        }

    def analyze_network_performance(self) -> Dict:
        """Analyze network performance metrics."""
        net_io = psutil.net_io_counters()
        net_connections = len(psutil.net_connections())
        
        return {
            "io": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errors_in": net_io.errin,
                "errors_out": net_io.errout
            },
            "connections": net_connections
        }

    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # CPU recommendations
        if analysis["system"]["cpu"]["usage_percent"] > 80:
            recommendations.append(
                "HIGH_CPU: Consider reducing local training epochs or implementing model compression"
            )
        
        # Memory recommendations
        if analysis["system"]["memory"]["usage_percent"] > 85:
            recommendations.append(
                "HIGH_MEMORY: Consider batch size reduction or gradient accumulation"
            )
        
        # Disk recommendations
        if analysis["system"]["disk"]["usage_percent"] > 90:
            recommendations.append(
                "LOW_DISK: Clean up old model checkpoints and logs"
            )
        
        # Network recommendations
        if analysis["network"]["io"]["errors_in"] > 0 or analysis["network"]["io"]["errors_out"] > 0:
            recommendations.append(
                "NETWORK_ERRORS: Check network configuration and P2P connectivity"
            )
        
        return recommendations

    def profile_training_performance(self) -> Dict:
        """Profile federated learning training performance."""
        # This would integrate with actual training metrics
        # For now, return mock data structure
        return {
            "training_metrics": {
                "rounds_completed": 0,
                "avg_round_time_s": 0,
                "avg_local_training_time_s": 0,
                "avg_aggregation_time_s": 0,
                "model_accuracy": 0,
                "convergence_rate": 0
            },
            "bottlenecks": []
        }

    async def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive performance analysis."""
        logger.info("Starting performance analysis...")
        
        analysis = {
            "timestamp": time.time(),
            "system": self.analyze_system_resources(),
            "network": self.analyze_network_performance(),
            "training": self.profile_training_performance()
        }
        
        recommendations = self.generate_recommendations(analysis)
        
        return {
            "analysis": analysis,
            "recommendations": recommendations,
            "performance_score": self.calculate_performance_score(analysis)
        }

    def calculate_performance_score(self, analysis: Dict) -> Dict:
        """Calculate overall performance score (0-100)."""
        scores = {
            "cpu": max(0, 100 - analysis["system"]["cpu"]["usage_percent"]),
            "memory": max(0, 100 - analysis["system"]["memory"]["usage_percent"]),
            "disk": max(0, 100 - analysis["system"]["disk"]["usage_percent"]),
            "network": 90 if analysis["network"]["io"]["errors_in"] == 0 else 50
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        return {
            "overall": round(overall_score, 1),
            "breakdown": scores,
            "grade": self.score_to_grade(overall_score)
        }

    def score_to_grade(self, score: float) -> str:
        """Convert performance score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


@click.command()
@click.option("--output", type=click.Path(), help="Output file for analysis")
@click.option("--format", type=click.Choice(["json", "text"]), default="text")
@click.option("--continuous", is_flag=True, help="Run continuous monitoring")
@click.option("--interval", default=60, help="Monitoring interval in seconds")
def main(output: Optional[str], format: str, continuous: bool, interval: int):
    """Run performance optimization analysis."""
    optimizer = PerformanceOptimizer()
    
    async def run_analysis():
        while True:
            results = await optimizer.run_comprehensive_analysis()
            
            if format == "json":
                output_text = json.dumps(results, indent=2)
            else:
                output_text = format_text_output(results)
            
            if output:
                with open(output, "w") as f:
                    f.write(output_text)
            else:
                print(output_text)
            
            if not continuous:
                break
            
            await asyncio.sleep(interval)
    
    asyncio.run(run_analysis())


def format_text_output(results: Dict) -> str:
    """Format results as human-readable text."""
    lines = [
        "Performance Analysis Report",
        "=" * 50,
        f"Overall Score: {results['performance_score']['overall']}/100 (Grade: {results['performance_score']['grade']})",
        "",
        "System Resources:",
        f"  CPU Usage: {results['analysis']['system']['cpu']['usage_percent']:.1f}%",
        f"  Memory Usage: {results['analysis']['system']['memory']['usage_percent']:.1f}%",
        f"  Disk Usage: {results['analysis']['system']['disk']['usage_percent']:.1f}%",
        "",
        "Recommendations:"
    ]
    
    for rec in results["recommendations"]:
        lines.append(f"  • {rec}")
    
    if not results["recommendations"]:
        lines.append("  • No optimization recommendations at this time")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()