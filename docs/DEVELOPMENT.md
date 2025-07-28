# Development Environment Setup

Quick setup guide for Agent Mesh Federated Runtime development.

## Prerequisites

- Python 3.9+ with asyncio support
- Node.js 18+ and npm 8+ for dashboard components  
- Docker and Docker Compose for testing
- Git with LFS support

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd agent-mesh-federated-runtime

# Install dependencies
pip install -e ".[dev]"
cd src/web/dashboard && npm install && cd ../../..

# Setup pre-commit hooks
pre-commit install

# Verify installation
npm run test:unit
npm run lint
npm run dev
```

## Configuration

```bash
# Copy environment template
cp .env.example .env
vim .env  # Edit as needed
```

## Testing

```bash
npm run test          # All tests
npm run test:unit     # Unit tests only
npm run test:e2e      # End-to-end tests
```

For detailed instructions, see [CONTRIBUTING.md](../CONTRIBUTING.md).