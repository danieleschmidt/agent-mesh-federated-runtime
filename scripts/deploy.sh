#!/bin/bash
set -e

ENVIRONMENT=${1:-development}
echo "🚀 Deploying Agent Mesh to $ENVIRONMENT..."

if [ "$ENVIRONMENT" = "development" ]; then
    echo "📝 Starting development deployment..."
    docker-compose -f docker-compose.dev.yml up -d
    echo "✅ Development deployment completed"
    echo "   Agent Mesh: http://localhost:8080"
    echo "   Monitoring: http://localhost:3000"

elif [ "$ENVIRONMENT" = "production" ]; then
    echo "📝 Starting production deployment..."
    
    # Check if Docker Swarm is initialized
    if ! docker info | grep -q "Swarm: active"; then
        echo "Initializing Docker Swarm..."
        docker swarm init
    fi
    
    echo "✅ Production deployment ready"
    echo "   Run: docker stack deploy -c docker-compose.prod.yml agent-mesh"
    
else
    echo "❌ Unknown environment: $ENVIRONMENT"
    echo "Usage: $0 [development|production]"
    exit 1
fi
