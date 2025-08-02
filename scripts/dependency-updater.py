#!/usr/bin/env python3
"""Intelligent dependency management and security updates."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import click
import requests
from packaging import version

logger = logging.getLogger(__name__)


class DependencyUpdater:
    """Intelligent dependency update manager."""

    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.pyproject_path = repo_path / "pyproject.toml"
        self.package_json_path = repo_path / "package.json"
        self.security_advisories = []

    def check_pypi_security_advisories(self, package_name: str) -> List[Dict]:
        """Check PyPI security advisories for a package."""
        try:
            # Use safety database or similar service
            result = subprocess.run(
                ["safety", "check", "--json", "--package", package_name],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.warning(f"Could not check security advisories for {package_name}: {e}")
        return []

    def get_latest_version(self, package_name: str, package_type: str = "pypi") -> Optional[str]:
        """Get latest version of a package."""
        try:
            if package_type == "pypi":
                response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
                if response.status_code == 200:
                    data = response.json()
                    return data["info"]["version"]
            elif package_type == "npm":
                result = subprocess.run(
                    ["npm", "view", package_name, "version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get latest version for {package_name}: {e}")
        return None

    def analyze_python_dependencies(self) -> Dict:
        """Analyze Python dependencies from pyproject.toml."""
        if not self.pyproject_path.exists():
            return {"dependencies": [], "dev_dependencies": []}

        try:
            import tomli
            with open(self.pyproject_path, "rb") as f:
                pyproject = tomli.load(f)
        except ImportError:
            logger.error("tomli not available, cannot parse pyproject.toml")
            return {"dependencies": [], "dev_dependencies": []}

        analysis = {
            "dependencies": [],
            "dev_dependencies": []
        }

        # Analyze main dependencies
        deps = pyproject.get("project", {}).get("dependencies", [])
        for dep in deps:
            dep_info = self.analyze_dependency(dep, "production")
            analysis["dependencies"].append(dep_info)

        # Analyze dev dependencies
        optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
        dev_deps = optional_deps.get("dev", [])
        for dep in dev_deps:
            dep_info = self.analyze_dependency(dep, "development")
            analysis["dev_dependencies"].append(dep_info)

        return analysis

    def analyze_dependency(self, dep_spec: str, dep_type: str) -> Dict:
        """Analyze a single dependency specification."""
        # Parse dependency specification (e.g., "requests>=2.25.0")
        package_name = dep_spec.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("~")[0].strip()
        
        latest_version = self.get_latest_version(package_name)
        security_issues = self.check_pypi_security_advisories(package_name)
        
        return {
            "name": package_name,
            "current_spec": dep_spec,
            "latest_version": latest_version,
            "security_issues": security_issues,
            "type": dep_type,
            "update_recommended": len(security_issues) > 0 or self.should_update(dep_spec, latest_version)
        }

    def should_update(self, current_spec: str, latest_version: Optional[str]) -> bool:
        """Determine if a dependency should be updated."""
        if not latest_version:
            return False
        
        # Extract version from spec (simplified)
        try:
            if ">="in current_spec:
                current_version = current_spec.split(">=")[1].strip()
            elif "==" in current_spec:
                current_version = current_spec.split("==")[1].strip()
            else:
                return False  # Cannot determine current version
            
            return version.parse(latest_version) > version.parse(current_version)
        except Exception:
            return False

    def generate_update_plan(self) -> Dict:
        """Generate comprehensive update plan."""
        python_analysis = self.analyze_python_dependencies()
        
        update_plan = {
            "high_priority": [],  # Security updates
            "medium_priority": [],  # Major version updates
            "low_priority": [],  # Minor/patch updates
            "no_update": []
        }
        
        all_deps = python_analysis["dependencies"] + python_analysis["dev_dependencies"]
        
        for dep in all_deps:
            if dep["security_issues"]:
                update_plan["high_priority"].append(dep)
            elif dep["update_recommended"]:
                # Categorize by version change type (simplified)
                update_plan["medium_priority"].append(dep)
            else:
                update_plan["no_update"].append(dep)
        
        return update_plan

    def apply_updates(self, plan: Dict, priority_level: str = "high_priority") -> bool:
        """Apply dependency updates based on plan."""
        updates_to_apply = plan.get(priority_level, [])
        
        if not updates_to_apply:
            logger.info(f"No {priority_level} updates to apply")
            return True
        
        logger.info(f"Applying {len(updates_to_apply)} {priority_level} updates...")
        
        for dep in updates_to_apply:
            try:
                if dep["latest_version"]:
                    cmd = ["pip", "install", f"{dep['name']}=={dep['latest_version']}"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Updated {dep['name']} to {dep['latest_version']}")
                    else:
                        logger.error(f"Failed to update {dep['name']}: {result.stderr}")
                        return False
            except Exception as e:
                logger.error(f"Error updating {dep['name']}: {e}")
                return False
        
        return True


@click.command()
@click.option("--check-only", is_flag=True, help="Only check for updates, don't apply")
@click.option("--security-only", is_flag=True, help="Only apply security updates")
@click.option("--output", type=click.Path(), help="Output file for analysis")
@click.option("--format", type=click.Choice(["json", "text"]), default="text")
def main(check_only: bool, security_only: bool, output: Optional[str], format: str):
    """Intelligent dependency management."""
    updater = DependencyUpdater()
    
    # Generate update plan
    plan = updater.generate_update_plan()
    
    if format == "json":
        output_text = json.dumps(plan, indent=2)
    else:
        output_text = format_text_output(plan)
    
    if output:
        with open(output, "w") as f:
            f.write(output_text)
    else:
        print(output_text)
    
    # Apply updates if not check-only
    if not check_only:
        if security_only:
            success = updater.apply_updates(plan, "high_priority")
        else:
            # Apply high and medium priority updates
            success = (updater.apply_updates(plan, "high_priority") and 
                      updater.apply_updates(plan, "medium_priority"))
        
        if not success:
            sys.exit(1)


def format_text_output(plan: Dict) -> str:
    """Format update plan as human-readable text."""
    lines = [
        "Dependency Update Plan",
        "=" * 50,
        ""
    ]
    
    for priority, deps in plan.items():
        if deps:
            lines.append(f"{priority.replace('_', ' ').title()}: ({len(deps)} packages)")
            for dep in deps:
                security_note = " [SECURITY]" if dep["security_issues"] else ""
                lines.append(f"  • {dep['name']}: {dep['current_spec']} → {dep['latest_version']}{security_note}")
            lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()