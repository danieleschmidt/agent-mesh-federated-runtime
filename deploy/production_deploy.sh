#!/bin/bash
set -e

echo "🚀 TERRAGON SDLC - PRODUCTION DEPLOYMENT SCRIPT"
echo "================================================="

# Configuration
DOCKER_IMAGE="agent-mesh:latest"
COMPOSE_FILE="deploy/docker-compose.prod.yml"
K8S_MANIFEST="deploy/kubernetes.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose found"
    
    # Check if running as root or in docker group
    if ! docker info &> /dev/null; then
        log_error "Cannot connect to Docker daemon. Are you in the docker group?"
        exit 1
    fi
    log_success "Docker daemon accessible"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    # Navigate to project root
    cd "$(dirname "$0")/.."
    
    # Build image
    docker build -t $DOCKER_IMAGE -f deploy/Dockerfile . || {
        log_error "Docker build failed"
        exit 1
    }
    
    log_success "Docker image built: $DOCKER_IMAGE"
}

# Run quality checks
run_quality_checks() {
    log_info "Running quality checks on built image..."
    
    # Test image can start
    docker run --rm -d --name agent-mesh-test \
        -e NODE_ROLE=test \
        -e LISTEN_PORT=4001 \
        $DOCKER_IMAGE sleep 10 || {
        log_error "Image startup test failed"
        exit 1
    }
    
    # Check if container is running
    sleep 2
    if docker ps | grep -q agent-mesh-test; then
        log_success "Image startup test passed"
        docker stop agent-mesh-test &> /dev/null
    else
        log_error "Container failed to start"
        docker logs agent-mesh-test 2>/dev/null || true
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    # Stop any existing deployment
    docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true
    
    # Create required directories
    mkdir -p data logs
    
    # Deploy
    docker-compose -f $COMPOSE_FILE up -d || {
        log_error "Docker Compose deployment failed"
        exit 1
    }
    
    log_success "Docker Compose deployment started"
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    local services=(
        "agent-mesh-bootstrap:8080"
        "agent-mesh-prometheus:9090"
        "agent-mesh-grafana:3000"
    )
    
    for service in "${services[@]}"; do
        local name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        if docker-compose -f $COMPOSE_FILE exec -T $name curl -f http://localhost:$port/health &>/dev/null || \
           docker-compose -f $COMPOSE_FILE exec -T $name curl -f http://localhost:$port/ &>/dev/null; then
            log_success "$name service is healthy"
        else
            log_warning "$name service may not be ready yet"
        fi
    done
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        return 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        return 1
    fi
    
    # Apply manifests
    kubectl apply -f $K8S_MANIFEST || {
        log_error "Kubernetes deployment failed"
        return 1
    }
    
    log_success "Kubernetes deployment applied"
    
    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/agent-mesh-bootstrap -n agent-mesh --timeout=300s
    kubectl rollout status deployment/agent-mesh-workers -n agent-mesh --timeout=300s
    
    log_success "Kubernetes deployment completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status"
    echo "=================="
    
    if docker-compose -f $COMPOSE_FILE ps 2>/dev/null; then
        echo -e "\n${BLUE}Docker Compose Services:${NC}"
        docker-compose -f $COMPOSE_FILE ps
        
        echo -e "\n${BLUE}Service URLs:${NC}"
        echo "• Agent Mesh Bootstrap: http://localhost:8080/health"
        echo "• Prometheus Metrics: http://localhost:9100"
        echo "• Grafana Dashboard: http://localhost:3000 (admin/admin)"
        echo "• Load Balancer: http://localhost:80"
    fi
    
    if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        echo -e "\n${BLUE}Kubernetes Pods:${NC}"
        kubectl get pods -n agent-mesh 2>/dev/null || echo "No Kubernetes deployment found"
        
        echo -e "\n${BLUE}Kubernetes Services:${NC}"
        kubectl get services -n agent-mesh 2>/dev/null || echo "No Kubernetes services found"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop Docker Compose
    docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true
    
    # Remove test containers
    docker rm -f agent-mesh-test 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    local deployment_type=${1:-"docker"}
    
    echo "Deployment Type: $deployment_type"
    echo "Docker Image: $DOCKER_IMAGE"
    echo "Timestamp: $(date)"
    echo "================================================="
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    build_image
    run_quality_checks
    
    case $deployment_type in
        "docker")
            deploy_docker_compose
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        "both")
            deploy_docker_compose
            deploy_kubernetes
            ;;
        *)
            log_error "Unknown deployment type: $deployment_type"
            log_info "Available types: docker, kubernetes (k8s), both"
            exit 1
            ;;
    esac
    
    show_status
    
    log_success "🎉 PRODUCTION DEPLOYMENT COMPLETED!"
    log_info "Monitor the deployment and check service health endpoints"
}

# Script usage
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [deployment_type]"
        echo ""
        echo "Deployment types:"
        echo "  docker     - Deploy using Docker Compose (default)"
        echo "  kubernetes - Deploy to Kubernetes cluster"  
        echo "  k8s        - Same as kubernetes"
        echo "  both       - Deploy to both Docker Compose and Kubernetes"
        echo ""
        echo "Examples:"
        echo "  $0                 # Deploy with Docker Compose"
        echo "  $0 docker          # Deploy with Docker Compose"
        echo "  $0 kubernetes      # Deploy to Kubernetes"
        echo "  $0 both            # Deploy to both"
        exit 0
    fi
    
    main "$@"
fi