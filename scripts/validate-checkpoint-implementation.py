#!/usr/bin/env python3
"""
Terragon Checkpointed SDLC Validation Script

This script validates that all checkpoints of the Terragon SDLC implementation
are properly configured and functional across the repository.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CheckpointValidator:
    """Validates Terragon SDLC checkpoint implementation"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.validation_results = {}
        
    def run_all_validations(self) -> Dict[str, any]:
        """Run all checkpoint validations"""
        print("🚀 Starting Terragon Checkpointed SDLC Validation")
        print("=" * 60)
        
        checkpoints = [
            ("CHECKPOINT 1", self.validate_checkpoint_1_foundation),
            ("CHECKPOINT 2", self.validate_checkpoint_2_devenv),
            ("CHECKPOINT 3", self.validate_checkpoint_3_testing),
            ("CHECKPOINT 4", self.validate_checkpoint_4_build),
            ("CHECKPOINT 5", self.validate_checkpoint_5_monitoring),
            ("CHECKPOINT 6", self.validate_checkpoint_6_workflows),
            ("CHECKPOINT 7", self.validate_checkpoint_7_metrics),
            ("CHECKPOINT 8", self.validate_checkpoint_8_integration),
        ]
        
        for name, validator in checkpoints:
            print(f"\n📋 Validating {name}...")
            try:
                result = validator()
                self.validation_results[name] = result
                status = "✅ PASS" if result["status"] == "pass" else "❌ FAIL"
                print(f"{status} {name}: {result.get('summary', 'Complete')}")
                
                if result["status"] == "fail" and result.get("missing"):
                    for missing in result["missing"][:3]:  # Show first 3
                        print(f"  ⚠️  Missing: {missing}")
                        
            except Exception as e:
                self.validation_results[name] = {
                    "status": "error", 
                    "error": str(e)
                }
                print(f"❌ ERROR {name}: {e}")
        
        self._print_summary()
        return self.validation_results
        
    def validate_checkpoint_1_foundation(self) -> Dict[str, any]:
        """Validate project foundation and documentation files"""
        required_files = [
            "README.md", "ARCHITECTURE.md", "PROJECT_CHARTER.md",
            "LICENSE", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", 
            "SECURITY.md", "CHANGELOG.md", "CODEOWNERS",
            "docs/ROADMAP.md", "docs/adr/000-adr-template.md",
            "docs/guides/user/getting-started.md"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        # Check documentation quality
        readme_path = self.repo_root / "README.md"
        if readme_path.exists():
            content = readme_path.read_text()
            has_features = "Key Features" in content
            has_quickstart = "Quick Start" in content
            has_terragon = "Terragon SDLC" in content
        else:
            has_features = has_quickstart = has_terragon = False
            
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Foundation files: {len(required_files) - len(missing)}/{len(required_files)}",
            "quality_checks": {
                "readme_features": has_features,
                "readme_quickstart": has_quickstart,
                "terragon_integration": has_terragon
            }
        }
        
    def validate_checkpoint_2_devenv(self) -> Dict[str, any]:
        """Validate development environment and tooling"""
        required_files = [
            "package.json", "requirements.txt", "pyproject.toml",
            "Dockerfile", "docker-compose.yml", ".gitignore",
            ".editorconfig", "setup.cfg", "tox.ini", "pytest.ini"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        # Check package.json scripts
        package_json_path = self.repo_root / "package.json"
        has_scripts = False
        if package_json_path.exists():
            try:
                with open(package_json_path) as f:
                    package_data = json.load(f)
                    scripts = package_data.get("scripts", {})
                    required_scripts = ["test", "lint", "build"]
                    has_scripts = all(script in scripts for script in required_scripts)
            except Exception:
                has_scripts = False
                
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"DevEnv files: {len(required_files) - len(missing)}/{len(required_files)}",
            "has_required_scripts": has_scripts
        }
        
    def validate_checkpoint_3_testing(self) -> Dict[str, any]:
        """Validate testing infrastructure"""
        required_dirs = ["tests/unit", "tests/integration", "tests/e2e", "tests/fixtures"]
        required_files = [
            "tests/conftest.py", "pytest.ini", "codecov.yml",
            "tests/unit/test_example.py", "tests/integration/test_network_integration.py"
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not (self.repo_root / dir_path).exists():
                missing.append(f"{dir_path}/")
                
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Testing infrastructure: {len(required_dirs + required_files) - len(missing)}/{len(required_dirs + required_files)}"
        }
        
    def validate_checkpoint_4_build(self) -> Dict[str, any]:
        """Validate build and containerization"""
        required_files = [
            "Dockerfile", "docker-compose.yml", ".dockerignore",
            "Makefile", "scripts/validate-docker-build.sh"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        # Check Dockerfile multi-stage
        dockerfile_path = self.repo_root / "Dockerfile"
        has_multistage = False
        if dockerfile_path.exists():
            content = dockerfile_path.read_text()
            has_multistage = "FROM" in content and "AS" in content
            
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Build files: {len(required_files) - len(missing)}/{len(required_files)}",
            "dockerfile_multistage": has_multistage
        }
        
    def validate_checkpoint_5_monitoring(self) -> Dict[str, any]:
        """Validate monitoring and observability setup"""
        required_files = [
            "monitoring/prometheus.yml", "monitoring/rules/agent_mesh.yml",
            "scripts/health-check.py", "scripts/integration-health-check.py"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Monitoring files: {len(required_files) - len(missing)}/{len(required_files)}"
        }
        
    def validate_checkpoint_6_workflows(self) -> Dict[str, any]:
        """Validate workflow documentation and templates"""
        required_files = [
            "docs/workflows/examples/ci.yml",
            "docs/workflows/examples/security-scan.yml",
            "docs/workflows/examples/advanced-security.yml",
            "docs/workflows/GITHUB_ACTIONS_SETUP.md",
            "docs/workflows/MANUAL_SETUP.md"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Workflow docs: {len(required_files) - len(missing)}/{len(required_files)}"
        }
        
    def validate_checkpoint_7_metrics(self) -> Dict[str, any]:
        """Validate metrics and automation setup"""
        required_files = [
            "scripts/dependency-updater.py", "scripts/performance-optimizer.py",
            "scripts/security-scan.sh", "scripts/environment-manager.py",
            "renovate.json", "sonar-project.properties"
        ]
        
        missing = []
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                missing.append(file_path)
                
        return {
            "status": "pass" if not missing else "fail",
            "missing": missing,
            "summary": f"Metrics/automation: {len(required_files) - len(missing)}/{len(required_files)}"
        }
        
    def validate_checkpoint_8_integration(self) -> Dict[str, any]:
        """Validate integration and final configuration"""
        # This checkpoint requires manual setup
        setup_file = self.repo_root / "docs/SETUP_REQUIRED.md"
        has_setup_guide = setup_file.exists()
        
        # Check if GitHub workflows exist (manual setup indicator)
        github_workflows_dir = self.repo_root / ".github/workflows"
        workflows_exist = github_workflows_dir.exists() and list(github_workflows_dir.glob("*.yml"))
        
        status = "manual_required" if not workflows_exist else "pass"
        
        return {
            "status": status,
            "summary": "Requires manual GitHub setup" if not workflows_exist else "Integration complete",
            "has_setup_guide": has_setup_guide,
            "workflows_configured": workflows_exist
        }
        
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("📊 TERRAGON CHECKPOINTED SDLC VALIDATION SUMMARY")
        print("=" * 60)
        
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results.values() 
                    if r.get("status") == "pass")
        manual = sum(1 for r in self.validation_results.values() 
                    if r.get("status") == "manual_required")
        failed = total - passed - manual
        
        print(f"✅ Passed: {passed}/{total}")
        print(f"⚠️  Manual Required: {manual}/{total}")
        print(f"❌ Failed: {failed}/{total}")
        
        if manual > 0:
            print(f"\n📋 Manual Setup Required:")
            print(f"   See docs/SETUP_REQUIRED.md for GitHub Actions setup")
            
        overall_status = "🚀 READY" if failed == 0 else "⚠️  NEEDS ATTENTION"
        print(f"\nOverall Status: {overall_status}")
        
        return {
            "total": total,
            "passed": passed,
            "manual_required": manual,
            "failed": failed,
            "overall_status": overall_status
        }


def main():
    """Main validation entry point"""
    validator = CheckpointValidator()
    results = validator.run_all_validations()
    
    # Exit with appropriate code
    failed = sum(1 for r in results.values() if r.get("status") == "fail")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()