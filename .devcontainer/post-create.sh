#!/bin/bash

# Agent Mesh Federated Runtime - Development Environment Setup
set -e

echo "🚀 Setting up Agent Mesh development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev curl wget git

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pytest-benchmark \
    pre-commit \
    safety \
    bandit \
    sphinx \
    sphinx-rtd-theme \
    jupyter \
    notebook

# Install libp2p dependencies (if needed for Python bindings)
echo "🌐 Installing P2P networking dependencies..."
sudo apt-get install -y protobuf-compiler

# Install Rust (for performance-critical components)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Install Go (for Protocol Buffers and some tooling)
if ! command -v go &> /dev/null; then
    wget -q https://go.dev/dl/go1.21.0.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi

# Install Docker Compose
echo "🐳 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install kubectl (for Kubernetes development)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install additional monitoring tools
echo "📊 Installing monitoring tools..."
pip install prometheus-client grafana-api

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
fi

# Create necessary directories
mkdir -p \
    tests/{unit,integration,e2e,performance} \
    docs/{api,guides,examples} \
    scripts/{deployment,monitoring,testing} \
    configs/{development,staging,production} \
    .github/{workflows,ISSUE_TEMPLATE,PULL_REQUEST_TEMPLATE}

# Install project in development mode if setup.py exists
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "📋 Installing project in development mode..."
    pip install -e ".[dev]" 2>/dev/null || echo "setup.py/pyproject.toml not ready yet"
fi

# Setup environment file
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "📄 Created .env from .env.example"
fi

# Install Node.js dependencies for any frontend components
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Setup Git hooks for better development workflow
echo "🔗 Setting up Git configuration..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Display success message
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  • make test          - Run test suite"
echo "  • make lint          - Run code quality checks"  
echo "  • make docs          - Build documentation"
echo "  • make dev           - Start development server"
echo "  • docker-compose up  - Start local services"
echo ""
echo "📚 Open docs/DEVELOPMENT.md for detailed development guide"