# Agent Mesh Development Container
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode

# Set up Python environment
RUN pip install --upgrade pip setuptools wheel

# Install development tools
RUN pip install \
    pre-commit \
    black \
    isort \
    flake8 \
    mypy \
    bandit \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pytest-benchmark \
    rich \
    click

# Switch to non-root user
USER vscode

# Set working directory
WORKDIR /workspaces/agent-mesh-federated-runtime

# Default command
CMD ["bash"]