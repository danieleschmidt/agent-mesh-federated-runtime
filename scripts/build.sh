#!/bin/bash
set -e

echo "🚀 Building Agent Mesh Docker image..."

# Build image
docker build -t terragonlabs/agent-mesh:latest .

# Tag with version
VERSION=$(python3 -c "import json; print(json.load(open('configs/production.json'))['app']['version'])")
docker tag terragonlabs/agent-mesh:latest terragonlabs/agent-mesh:$VERSION

echo "✅ Build completed"
echo "   Image: terragonlabs/agent-mesh:latest"
echo "   Tagged: terragonlabs/agent-mesh:$VERSION"
