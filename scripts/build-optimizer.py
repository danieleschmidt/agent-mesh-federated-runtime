#!/usr/bin/env python3
"""
Build optimization script for Agent Mesh Federated Runtime.
Optimizes Docker builds, manages build cache, and provides build analytics.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class BuildOptimizer:
    """Optimizes Docker builds and provides build analytics."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_cache_dir = project_root / ".build-cache"
        self.build_cache_dir.mkdir(exist_ok=True)
        self.build_stats_file = self.build_cache_dir / "build-stats.json"
        self.docker_registry = os.getenv("DOCKER_REGISTRY", "ghcr.io/your-org")
        self.project_name = "agent-mesh-federated-runtime"
        
    def load_build_stats(self) -> Dict:
        """Load build statistics from cache."""
        if self.build_stats_file.exists():
            with open(self.build_stats_file) as f:
                return json.load(f)
        return {"builds": [], "cache_metrics": {}}
    
    def save_build_stats(self, stats: Dict):
        """Save build statistics to cache."""
        with open(self.build_stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def get_image_size(self, image_tag: str) -> Optional[int]:
        """Get Docker image size in bytes."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_tag, "--format", "{{.Size}}"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None
    
    def get_layer_count(self, image_tag: str) -> Optional[int]:
        """Get number of layers in Docker image."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_tag, "--format", "{{len .RootFS.Layers}}"],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            return None
    
    def build_image(self, stage: str = "production", tags: List[str] = None, 
                   use_cache: bool = True, build_args: Dict[str, str] = None) -> Dict:
        """Build Docker image with optimization."""
        start_time = time.time()
        
        if tags is None:
            tags = [f"{self.docker_registry}/{self.project_name}:{stage}"]
        
        if build_args is None:
            build_args = {}
        
        # Build Docker command
        cmd = ["docker", "build"]
        
        # Add build arguments
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])
        
        # Add cache options
        if use_cache:
            # Use registry cache
            cache_from_tag = f"{self.docker_registry}/{self.project_name}:cache-{stage}"
            cmd.extend(["--cache-from", cache_from_tag])
        
        # Add target stage
        cmd.extend(["--target", stage])
        
        # Add tags
        for tag in tags:
            cmd.extend(["-t", tag])
        
        # Add context
        cmd.append(str(self.project_root))
        
        print(f"Building image for stage '{stage}'...")
        print(f"Command: {' '.join(cmd)}")
        
        # Execute build
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            build_time = time.time() - start_time
            
            # Collect build metrics
            primary_tag = tags[0]
            image_size = self.get_image_size(primary_tag)
            layer_count = self.get_layer_count(primary_tag)
            
            build_info = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "tags": tags,
                "build_time": build_time,
                "image_size": image_size,
                "layer_count": layer_count,
                "success": True,
                "use_cache": use_cache,
                "build_args": build_args
            }
            
            print(f"✅ Build completed in {build_time:.2f}s")
            if image_size:
                print(f"📦 Image size: {image_size / (1024**2):.1f} MB")
            if layer_count:
                print(f"🍰 Layers: {layer_count}")
                
            return build_info
            
        except subprocess.CalledProcessError as e:
            build_time = time.time() - start_time
            build_info = {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "tags": tags,
                "build_time": build_time,
                "success": False,
                "error": e.stderr,
                "use_cache": use_cache,
                "build_args": build_args
            }
            
            print(f"❌ Build failed after {build_time:.2f}s")
            print(f"Error: {e.stderr}")
            
            return build_info
    
    def build_multi_stage(self, stages: List[str], use_cache: bool = True) -> List[Dict]:
        """Build multiple Docker stages."""
        build_results = []
        
        for stage in stages:
            print(f"\n{'='*60}")
            print(f"Building stage: {stage}")
            print(f"{'='*60}")
            
            result = self.build_image(stage=stage, use_cache=use_cache)
            build_results.append(result)
            
            if not result["success"]:
                print(f"❌ Build failed for stage '{stage}', stopping...")
                break
        
        return build_results
    
    def build_multi_arch(self, stage: str = "production", platforms: List[str] = None) -> Dict:
        """Build multi-architecture images."""
        if platforms is None:
            platforms = ["linux/amd64", "linux/arm64"]
        
        start_time = time.time()
        platform_str = ",".join(platforms)
        
        tag = f"{self.docker_registry}/{self.project_name}:latest"
        
        cmd = [
            "docker", "buildx", "build",
            "--platform", platform_str,
            "--target", stage,
            "-t", tag,
            "--push",
            str(self.project_root)
        ]
        
        print(f"Building multi-arch image for platforms: {platform_str}")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            build_time = time.time() - start_time
            
            build_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "multi-arch",
                "stage": stage,
                "platforms": platforms,
                "tag": tag,
                "build_time": build_time,
                "success": True
            }
            
            print(f"✅ Multi-arch build completed in {build_time:.2f}s")
            return build_info
            
        except subprocess.CalledProcessError as e:
            build_time = time.time() - start_time
            build_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "multi-arch",
                "stage": stage,
                "platforms": platforms,
                "tag": tag,
                "build_time": build_time,
                "success": False,
                "error": e.stderr
            }
            
            print(f"❌ Multi-arch build failed after {build_time:.2f}s")
            print(f"Error: {e.stderr}")
            
            return build_info
    
    def optimize_dockerfile(self) -> List[str]:
        """Analyze Dockerfile and suggest optimizations."""
        dockerfile_path = self.project_root / "Dockerfile"
        if not dockerfile_path.exists():
            return ["Dockerfile not found"]
        
        suggestions = []
        
        with open(dockerfile_path) as f:
            lines = f.readlines()
        
        # Check for optimization opportunities
        has_multi_stage = any("FROM" in line and "as" in line for line in lines)
        if not has_multi_stage:
            suggestions.append("Consider using multi-stage builds to reduce image size")
        
        # Check for .dockerignore
        dockerignore_path = self.project_root / ".dockerignore"
        if not dockerignore_path.exists():
            suggestions.append("Add .dockerignore file to exclude unnecessary files")
        
        # Check for cache optimization
        run_commands = [line for line in lines if line.strip().startswith("RUN")]
        if len(run_commands) > 10:
            suggestions.append("Consider combining RUN commands to reduce layers")
        
        # Check for package manager cache cleanup
        cache_cleanup_patterns = ["rm -rf /var/lib/apt/lists/*", "apt-get clean", "yum clean all"]
        has_cleanup = any(
            any(pattern in line for pattern in cache_cleanup_patterns)
            for line in lines
        )
        if not has_cleanup:
            suggestions.append("Add package manager cache cleanup to reduce image size")
        
        return suggestions
    
    def analyze_build_history(self) -> Dict:
        """Analyze build history and provide insights."""
        stats = self.load_build_stats()
        builds = stats.get("builds", [])
        
        if not builds:
            return {"message": "No build history available"}
        
        # Calculate statistics
        successful_builds = [b for b in builds if b.get("success")]
        failed_builds = [b for b in builds if not b.get("success")]
        
        if successful_builds:
            avg_build_time = sum(b["build_time"] for b in successful_builds) / len(successful_builds)
            avg_image_size = sum(b.get("image_size", 0) for b in successful_builds if b.get("image_size")) / \
                           len([b for b in successful_builds if b.get("image_size")])
        else:
            avg_build_time = 0
            avg_image_size = 0
        
        analysis = {
            "total_builds": len(builds),
            "successful_builds": len(successful_builds),
            "failed_builds": len(failed_builds),
            "success_rate": len(successful_builds) / len(builds) * 100 if builds else 0,
            "average_build_time": avg_build_time,
            "average_image_size_mb": avg_image_size / (1024**2) if avg_image_size else 0,
            "recent_builds": builds[-5:] if builds else []
        }
        
        return analysis
    
    def clean_build_cache(self):
        """Clean Docker build cache."""
        print("Cleaning Docker build cache...")
        
        try:
            # Remove unused build cache
            subprocess.run(["docker", "builder", "prune", "-f"], check=True)
            
            # Remove dangling images
            subprocess.run(["docker", "image", "prune", "-f"], check=True)
            
            print("✅ Build cache cleaned successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clean build cache: {e}")
    
    def push_cache_image(self, stage: str):
        """Push cache image to registry."""
        local_tag = f"{self.docker_registry}/{self.project_name}:{stage}"
        cache_tag = f"{self.docker_registry}/{self.project_name}:cache-{stage}"
        
        try:
            # Tag image for cache
            subprocess.run(["docker", "tag", local_tag, cache_tag], check=True)
            
            # Push cache image
            subprocess.run(["docker", "push", cache_tag], check=True)
            
            print(f"✅ Cache image pushed: {cache_tag}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to push cache image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Docker build optimizer for Agent Mesh")
    parser.add_argument("--stage", default="production", help="Build stage")
    parser.add_argument("--stages", nargs="+", help="Multiple stages to build")
    parser.add_argument("--multi-arch", action="store_true", help="Build multi-architecture images")
    parser.add_argument("--platforms", nargs="+", default=["linux/amd64", "linux/arm64"],
                       help="Platforms for multi-arch build")
    parser.add_argument("--no-cache", action="store_true", help="Disable build cache")
    parser.add_argument("--analyze", action="store_true", help="Analyze build history")
    parser.add_argument("--optimize", action="store_true", help="Show optimization suggestions")
    parser.add_argument("--clean", action="store_true", help="Clean build cache")
    parser.add_argument("--push-cache", action="store_true", help="Push cache images")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    optimizer = BuildOptimizer(project_root)
    
    # Load existing stats
    stats = optimizer.load_build_stats()
    
    if args.clean:
        optimizer.clean_build_cache()
        return
    
    if args.analyze:
        analysis = optimizer.analyze_build_history()
        print("\n📊 Build History Analysis")
        print("=" * 40)
        for key, value in analysis.items():
            if key != "recent_builds":
                print(f"{key.replace('_', ' ').title()}: {value}")
        
        if analysis.get("recent_builds"):
            print("\nRecent Builds:")
            for build in analysis["recent_builds"]:
                status = "✅" if build.get("success") else "❌"
                print(f"  {status} {build.get('stage', 'unknown')} - {build.get('build_time', 0):.1f}s")
        return
    
    if args.optimize:
        suggestions = optimizer.optimize_dockerfile()
        print("\n🔧 Dockerfile Optimization Suggestions")
        print("=" * 40)
        for suggestion in suggestions:
            print(f"• {suggestion}")
        return
    
    # Build images
    build_results = []
    
    if args.multi_arch:
        result = optimizer.build_multi_arch(
            stage=args.stage,
            platforms=args.platforms
        )
        build_results.append(result)
    
    elif args.stages:
        results = optimizer.build_multi_stage(
            stages=args.stages,
            use_cache=not args.no_cache
        )
        build_results.extend(results)
    
    else:
        result = optimizer.build_image(
            stage=args.stage,
            use_cache=not args.no_cache
        )
        build_results.append(result)
    
    # Save build results
    stats["builds"].extend(build_results)
    optimizer.save_build_stats(stats)
    
    # Push cache images if requested
    if args.push_cache:
        for result in build_results:
            if result.get("success") and result.get("stage"):
                optimizer.push_cache_image(result["stage"])
    
    # Summary
    successful = sum(1 for r in build_results if r.get("success"))
    total = len(build_results)
    
    print(f"\n📋 Build Summary: {successful}/{total} successful")
    
    if successful < total:
        sys.exit(1)


if __name__ == "__main__":
    main()