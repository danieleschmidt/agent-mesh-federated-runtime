#!/usr/bin/env python3
"""Development environment setup script for Agent Mesh Federated Runtime."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command with error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version meets requirements."""
    version = sys.version_info
    if version < (3, 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install Python dependencies."""
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements-dev.txt", "Installing development dependencies"),
        ("pip install -e .", "Installing package in editable mode"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True


def setup_pre_commit():
    """Install and configure pre-commit hooks."""
    commands = [
        ("pre-commit install", "Installing pre-commit hooks"),
        ("pre-commit install --hook-type commit-msg", "Installing commit-msg hooks"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        "logs",
        "data",
        "models",
        "checkpoints", 
        "configs/local",
        "test_results",
        "coverage_reports",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def setup_git_config():
    """Configure git settings for the project."""
    commands = [
        ("git config core.autocrlf false", "Setting git autocrlf"),
        ("git config pull.rebase false", "Setting git pull strategy"),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)  # Don't fail on git config errors
    
    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Agent Mesh Federated Runtime development environment...\n")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Run setup steps
    steps = [
        (create_directories, "Creating project directories"),
        (install_dependencies, "Installing dependencies"),
        (setup_pre_commit, "Setting up pre-commit hooks"),
        (setup_git_config, "Configuring git settings"),
    ]
    
    failed_steps = []
    for step_func, step_name in steps:
        print(f"\n📋 {step_name}...")
        if not step_func():
            failed_steps.append(step_name)
    
    print("\n" + "="*60)
    if failed_steps:
        print("❌ Setup completed with errors:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease address the errors above before continuing.")
        sys.exit(1)
    else:
        print("✅ Development environment setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Copy .env.example to .env and configure")
        print("   2. Run 'pytest tests/' to verify installation")
        print("   3. Run 'pre-commit run --all-files' to test hooks")
        print("   4. Start development with 'npm run dev'")


if __name__ == "__main__":
    main()