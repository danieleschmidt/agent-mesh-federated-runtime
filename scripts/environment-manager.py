#!/usr/bin/env python3
"""Advanced environment management for Agent Mesh."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import click
import yaml

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manage development and deployment environments."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("env-config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """Load environment configuration."""
        default_config = {
            "environments": {
                "development": {
                    "python_version": "3.11",
                    "node_version": "18",
                    "dependencies": ["dev", "test"],
                    "services": ["redis", "postgres"],
                    "env_vars": {
                        "DEBUG": "true",
                        "LOG_LEVEL": "DEBUG"
                    }
                },
                "staging": {
                    "python_version": "3.11",
                    "node_version": "18",
                    "dependencies": ["all"],
                    "services": ["redis", "postgres", "prometheus"],
                    "env_vars": {
                        "DEBUG": "false",
                        "LOG_LEVEL": "INFO"
                    }
                },
                "production": {
                    "python_version": "3.11",
                    "node_version": "18",
                    "dependencies": ["monitoring", "ml"],
                    "services": ["redis", "postgres", "prometheus", "grafana"],
                    "env_vars": {
                        "DEBUG": "false",
                        "LOG_LEVEL": "WARNING"
                    }
                }
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config

    def setup_python_environment(self, env_name: str) -> bool:
        """Set up Python virtual environment."""
        env_config = self.config["environments"].get(env_name)
        if not env_config:
            logger.error(f"Environment {env_name} not found in config")
            return False
        
        python_version = env_config["python_version"]
        venv_path = Path(f".venv-{env_name}")
        
        try:
            # Create virtual environment
            subprocess.run([
                f"python{python_version}", "-m", "venv", str(venv_path)
            ], check=True)
            
            # Install dependencies
            pip_path = venv_path / "bin" / "pip"
            if not pip_path.exists():  # Windows
                pip_path = venv_path / "Scripts" / "pip.exe"
            
            # Upgrade pip
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            # Install project dependencies
            deps = env_config["dependencies"]
            for dep_group in deps:
                if dep_group == "all":
                    subprocess.run([str(pip_path), "install", "-e", ".[all]"], check=True)
                else:
                    subprocess.run([str(pip_path), "install", "-e", f".[{dep_group}]"], check=True)
            
            logger.info(f"Python environment {env_name} set up successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up Python environment: {e}")
            return False

    def setup_node_environment(self, env_name: str) -> bool:
        """Set up Node.js environment."""
        env_config = self.config["environments"].get(env_name)
        if not env_config:
            return False
        
        node_version = env_config["node_version"]
        
        try:
            # Use nvm if available
            subprocess.run([
                "bash", "-c", 
                f"source ~/.nvm/nvm.sh && nvm install {node_version} && nvm use {node_version}"
            ], check=True)
            
            # Install npm dependencies
            subprocess.run(["npm", "install"], check=True)
            
            # Install dashboard dependencies
            dashboard_path = Path("src/web/dashboard")
            if dashboard_path.exists():
                subprocess.run(["npm", "install"], cwd=dashboard_path, check=True)
            
            logger.info(f"Node.js environment {env_name} set up successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up Node.js environment: {e}")
            return False

    def setup_services(self, env_name: str) -> bool:
        """Set up required services using Docker Compose."""
        env_config = self.config["environments"].get(env_name)
        if not env_config:
            return False
        
        services = env_config["services"]
        
        # Generate docker-compose file for services
        compose_config = {
            "version": "3.8",
            "services": {}
        }
        
        for service in services:
            if service == "redis":
                compose_config["services"]["redis"] = {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"]
                }
            elif service == "postgres":
                compose_config["services"]["postgres"] = {
                    "image": "postgres:15-alpine",
                    "ports": ["5432:5432"],
                    "environment": {
                        "POSTGRES_DB": "agentmesh",
                        "POSTGRES_USER": "agentmesh",
                        "POSTGRES_PASSWORD": "development"
                    },
                    "volumes": ["postgres_data:/var/lib/postgresql/data"]
                }
            elif service == "prometheus":
                compose_config["services"]["prometheus"] = {
                    "image": "prom/prometheus:latest",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml",
                        "prometheus_data:/prometheus"
                    ]
                }
            elif service == "grafana":
                compose_config["services"]["grafana"] = {
                    "image": "grafana/grafana:latest",
                    "ports": ["3000:3000"],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "admin"
                    },
                    "volumes": ["grafana_data:/var/lib/grafana"]
                }
        
        # Add volumes
        compose_config["volumes"] = {}
        for service in services:
            if service in ["redis", "postgres", "prometheus", "grafana"]:
                compose_config["volumes"][f"{service}_data"] = {}
        
        # Write docker-compose file
        compose_file = Path(f"docker-compose.{env_name}.yml")
        with open(compose_file, "w") as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        try:
            # Start services
            subprocess.run([
                "docker-compose", "-f", str(compose_file), "up", "-d"
            ], check=True)
            
            logger.info(f"Services for {env_name} started successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start services: {e}")
            return False

    def create_env_file(self, env_name: str) -> bool:
        """Create environment variables file."""
        env_config = self.config["environments"].get(env_name)
        if not env_config:
            return False
        
        env_vars = env_config.get("env_vars", {})
        
        # Add default environment variables
        default_vars = {
            "ENVIRONMENT": env_name,
            "PYTHONPATH": "src",
            "DATABASE_URL": "postgresql://agentmesh:development@localhost:5432/agentmesh",
            "REDIS_URL": "redis://localhost:6379/0",
            "PROMETHEUS_URL": "http://localhost:9090",
            "GRAFANA_URL": "http://localhost:3000"
        }
        
        env_vars.update(default_vars)
        
        env_file = Path(f".env.{env_name}")
        with open(env_file, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Environment file {env_file} created successfully")
        return True

    def validate_environment(self, env_name: str) -> Dict:
        """Validate environment setup."""
        validation_results = {
            "python_env": False,
            "node_env": False,
            "services": {},
            "env_file": False
        }
        
        # Check Python environment
        venv_path = Path(f".venv-{env_name}")
        if venv_path.exists():
            validation_results["python_env"] = True
        
        # Check Node.js environment
        if Path("node_modules").exists():
            validation_results["node_env"] = True
        
        # Check services
        env_config = self.config["environments"].get(env_name, {})
        services = env_config.get("services", [])
        
        for service in services:
            try:
                if service == "redis":
                    result = subprocess.run(
                        ["redis-cli", "ping"], 
                        capture_output=True, timeout=5
                    )
                    validation_results["services"][service] = result.returncode == 0
                elif service == "postgres":
                    result = subprocess.run([
                        "pg_isready", "-h", "localhost", "-p", "5432"
                    ], capture_output=True, timeout=5)
                    validation_results["services"][service] = result.returncode == 0
                else:
                    # For other services, just check if container is running
                    result = subprocess.run([
                        "docker", "ps", "--filter", f"name={service}", "--format", "{{.Names}}"
                    ], capture_output=True, text=True)
                    validation_results["services"][service] = service in result.stdout
            except Exception:
                validation_results["services"][service] = False
        
        # Check environment file
        env_file = Path(f".env.{env_name}")
        validation_results["env_file"] = env_file.exists()
        
        return validation_results

    def setup_environment(self, env_name: str) -> bool:
        """Set up complete environment."""
        logger.info(f"Setting up {env_name} environment...")
        
        success = True
        
        # Set up Python environment
        if not self.setup_python_environment(env_name):
            success = False
        
        # Set up Node.js environment
        if not self.setup_node_environment(env_name):
            logger.warning("Node.js environment setup failed, continuing...")
        
        # Set up services
        if not self.setup_services(env_name):
            success = False
        
        # Create environment file
        if not self.create_env_file(env_name):
            success = False
        
        if success:
            logger.info(f"Environment {env_name} set up successfully")
            
            # Print activation instructions
            print(f"\nTo activate the {env_name} environment:")
            print(f"  source .venv-{env_name}/bin/activate  # Linux/macOS")
            print(f"  .venv-{env_name}\\Scripts\\activate     # Windows")
            print(f"  source .env.{env_name}                # Load environment variables")
        
        return success


@click.command()
@click.argument("action", type=click.Choice(["setup", "validate", "teardown"]))
@click.argument("environment", type=click.Choice(["development", "staging", "production"]))
@click.option("--config", type=click.Path(exists=True), help="Custom configuration file")
def main(action: str, environment: str, config: Optional[str]):
    """Manage development and deployment environments."""
    manager = EnvironmentManager(Path(config) if config else None)
    
    if action == "setup":
        success = manager.setup_environment(environment)
        if not success:
            exit(1)
    elif action == "validate":
        results = manager.validate_environment(environment)
        
        print(f"\nValidation results for {environment}:")
        print(f"  Python environment: {'✓' if results['python_env'] else '✗'}")
        print(f"  Node.js environment: {'✓' if results['node_env'] else '✗'}")
        print(f"  Environment file: {'✓' if results['env_file'] else '✗'}")
        print("  Services:")
        for service, status in results["services"].items():
            print(f"    {service}: {'✓' if status else '✗'}")
        
        # Exit with error if validation fails
        all_good = (results["python_env"] and results["env_file"] and 
                   all(results["services"].values()))
        if not all_good:
            exit(1)
    elif action == "teardown":
        # Stop services
        compose_file = Path(f"docker-compose.{environment}.yml")
        if compose_file.exists():
            subprocess.run([
                "docker-compose", "-f", str(compose_file), "down", "-v"
            ])
            compose_file.unlink()
        
        # Remove virtual environment
        venv_path = Path(f".venv-{environment}")
        if venv_path.exists():
            import shutil
            shutil.rmtree(venv_path)
        
        # Remove environment file
        env_file = Path(f".env.{environment}")
        if env_file.exists():
            env_file.unlink()
        
        print(f"Environment {environment} torn down successfully")


if __name__ == "__main__":
    main()