#!/bin/bash

# Agent Mesh Federated Runtime - Global Deployment Script
# Production-ready autonomous deployment with comprehensive health checks

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="agent-mesh-federated-runtime"
readonly NAMESPACE="agent-mesh"
readonly DOCKER_IMAGE="ghcr.io/danieleschmidt/agent-mesh-federated-runtime"

# Environment settings
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOY_REGION="${DEPLOY_REGION:-us-west-2}"
KUBERNETES_CONTEXT="${KUBERNETES_CONTEXT:-}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
SKIP_TESTS="${SKIP_TESTS:-false}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Helper functions
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("kubectl" "docker" "helm")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_error "Please install the missing dependencies and try again."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

validate_kubernetes_context() {
    log_info "Validating Kubernetes context..."
    
    if [ -z "$KUBERNETES_CONTEXT" ]; then
        KUBERNETES_CONTEXT=$(kubectl config current-context)
        log_info "Using current context: $KUBERNETES_CONTEXT"
    else
        kubectl config use-context "$KUBERNETES_CONTEXT" || {
            log_error "Failed to switch to context: $KUBERNETES_CONTEXT"
            exit 1
        }
    fi
    
    # Verify cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Kubernetes cluster is accessible"
}

build_and_push_image() {
    log_info "Building Docker image..."
    
    # Build multi-architecture image
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        --tag "${DOCKER_IMAGE}:${DOCKER_TAG}" \
        --tag "${DOCKER_IMAGE}:latest" \
        --push \
        --target production \
        . || {
        log_error "Failed to build and push Docker image"
        exit 1
    }
    
    log_success "Docker image built and pushed successfully"
}

run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log_info "Running comprehensive test suite..."
    
    # Build test image
    docker buildx build \
        --platform linux/amd64 \
        --tag "${DOCKER_IMAGE}:test" \
        --target testing \
        --load \
        .
    
    # Run tests in container
    docker run --rm \
        -v "$(pwd)/test-results:/app/test-results" \
        "${DOCKER_IMAGE}:test" \
        python -m pytest tests/ -v --junitxml=/app/test-results/junit.xml || {
        log_error "Tests failed"
        exit 1
    }
    
    log_success "All tests passed"
}

create_namespace() {
    log_info "Creating/updating namespace..."
    
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    name: ${NAMESPACE}
    environment: ${ENVIRONMENT}
    istio-injection: enabled
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-mesh
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: agent-mesh
    app.kubernetes.io/component: serviceaccount
EOF
    
    log_success "Namespace and ServiceAccount created"
}

deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret agent-mesh-secrets -n "$NAMESPACE" &> /dev/null; then
        log_info "Secrets already exist, skipping creation"
        return
    fi
    
    # Generate random secrets if not provided
    local db_password=${DB_PASSWORD:-$(openssl rand -base64 32)}
    local jwt_secret=${JWT_SECRET:-$(openssl rand -base64 64)}
    local encryption_key=${ENCRYPTION_KEY:-$(openssl rand -base64 32)}
    
    kubectl create secret generic agent-mesh-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=db-password="$db_password" \
        --from-literal=jwt-secret="$jwt_secret" \
        --from-literal=encryption-key="$encryption_key" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Secrets deployed successfully"
}

deploy_configmap() {
    log_info "Deploying configuration..."
    
    kubectl apply -f configs/global-deployment.yaml
    
    log_success "Configuration deployed successfully"
}

deploy_application() {
    log_info "Deploying Agent Mesh application..."
    
    # Update deployment with current image tag
    envsubst < k8s/deployment.yaml | kubectl apply -f -
    
    # Apply other Kubernetes manifests
    kubectl apply -f k8s/
    
    log_success "Application deployed successfully"
}

deploy_monitoring() {
    if [ "$ENABLE_MONITORING" != "true" ]; then
        log_info "Monitoring disabled, skipping deployment"
        return
    fi
    
    log_info "Deploying monitoring stack..."
    
    # Add Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --values monitoring/prometheus-values.yaml \
        --wait
    
    # Deploy custom monitoring rules
    kubectl apply -f monitoring/rules/
    
    log_success "Monitoring stack deployed successfully"
}

wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for StatefulSet to be ready
    kubectl rollout status statefulset/agent-mesh -n "$NAMESPACE" --timeout=600s
    
    # Wait for all pods to be ready
    kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=agent-mesh -n "$NAMESPACE" --timeout=300s
    
    log_success "Deployment is ready"
}

run_health_checks() {
    log_info "Running health checks..."
    
    # Get service endpoint
    local service_ip
    service_ip=$(kubectl get svc agent-mesh-global -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$service_ip" ]; then
        service_ip=$(kubectl get svc agent-mesh-global -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    fi
    
    if [ -z "$service_ip" ]; then
        log_warning "Load balancer IP not available yet, using port-forward for health check"
        kubectl port-forward svc/agent-mesh-global 8000:8000 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        service_ip="localhost"
    fi
    
    # Health check endpoints
    local endpoints=("/health" "/ready" "/metrics")
    
    for endpoint in "${endpoints[@]}"; do
        log_info "Checking endpoint: $endpoint"
        
        if curl -f -s "http://${service_ip}:8000${endpoint}" > /dev/null; then
            log_success "Endpoint $endpoint is healthy"
        else
            log_error "Endpoint $endpoint is not responding"
            if [ -n "${port_forward_pid:-}" ]; then
                kill $port_forward_pid 2>/dev/null || true
            fi
            exit 1
        fi
    done
    
    if [ -n "${port_forward_pid:-}" ]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
    
    log_success "All health checks passed"
}

run_integration_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping integration tests as requested"
        return
    fi
    
    log_info "Running integration tests..."
    
    # Run integration test pod
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: agent-mesh-integration-test
  namespace: ${NAMESPACE}
spec:
  template:
    spec:
      serviceAccountName: agent-mesh
      containers:
      - name: integration-test
        image: ${DOCKER_IMAGE}:${DOCKER_TAG}
        command: ["python", "-m", "pytest", "tests/integration/", "-v"]
        env:
        - name: MESH_API_URL
          value: "http://agent-mesh-global:8000"
      restartPolicy: Never
  backoffLimit: 3
EOF
    
    # Wait for integration tests to complete
    kubectl wait --for=condition=complete job/agent-mesh-integration-test -n "$NAMESPACE" --timeout=600s
    
    # Check test results
    if kubectl get job agent-mesh-integration-test -n "$NAMESPACE" -o jsonpath='{.status.succeeded}' | grep -q "1"; then
        log_success "Integration tests passed"
    else
        log_error "Integration tests failed"
        kubectl logs job/agent-mesh-integration-test -n "$NAMESPACE"
        exit 1
    fi
    
    # Cleanup test job
    kubectl delete job agent-mesh-integration-test -n "$NAMESPACE"
}

cleanup_old_resources() {
    log_info "Cleaning up old resources..."
    
    # Remove old ReplicaSets
    kubectl delete rs -l app.kubernetes.io/name=agent-mesh -n "$NAMESPACE" --cascade=orphan || true
    
    # Remove old ConfigMaps (keep current ones)
    kubectl get configmap -n "$NAMESPACE" -l app.kubernetes.io/name=agent-mesh --sort-by=.metadata.creationTimestamp | head -n -3 | awk '{print $1}' | xargs -r kubectl delete configmap -n "$NAMESPACE" || true
    
    log_success "Old resources cleaned up"
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "========================"
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: $NAMESPACE"
    echo "Image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
    echo "Context: $KUBERNETES_CONTEXT"
    echo ""
    
    log_info "Service Information:"
    kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/name=agent-mesh
    echo ""
    
    log_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=agent-mesh
    echo ""
    
    log_info "Access URLs:"
    local service_ip
    service_ip=$(kubectl get svc agent-mesh-global -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ -z "$service_ip" ]; then
        service_ip=$(kubectl get svc agent-mesh-global -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")
    fi
    
    echo "API: http://${service_ip}:8000"
    echo "gRPC: ${service_ip}:5001"
    echo "P2P: ${service_ip}:4001"
    echo "Metrics: http://${service_ip}:9090/metrics"
    echo ""
    
    if [ "$ENABLE_MONITORING" = "true" ]; then
        log_info "Monitoring URLs:"
        echo "Prometheus: kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090"
        echo "Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    fi
}

main() {
    log_info "Starting Agent Mesh Federated Runtime deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $DEPLOY_REGION"
    
    # Pre-deployment checks
    check_dependencies
    validate_kubernetes_context
    
    # Build and test
    build_and_push_image
    run_tests
    
    # Deploy infrastructure
    create_namespace
    deploy_secrets
    deploy_configmap
    
    # Deploy application
    deploy_application
    deploy_monitoring
    
    # Post-deployment validation
    wait_for_deployment
    run_health_checks
    run_integration_tests
    
    # Cleanup and info
    cleanup_old_resources
    show_deployment_info
    
    log_success "🎉 Agent Mesh Federated Runtime deployed successfully!"
    log_success "🚀 System is ready for federated learning and multi-agent coordination!"
}

# Handle script interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --environment|-e)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --region|-r)
            DEPLOY_REGION="$2"
            shift 2
            ;;
        --context|-c)
            KUBERNETES_CONTEXT="$2"
            shift 2
            ;;
        --tag|-t)
            DOCKER_TAG="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --no-monitoring)
            ENABLE_MONITORING="false"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --environment, -e   Deployment environment (default: production)"
            echo "  --region, -r        AWS region (default: us-west-2)"
            echo "  --context, -c       Kubernetes context to use"
            echo "  --tag, -t           Docker image tag (default: latest)"
            echo "  --skip-tests        Skip test execution"
            echo "  --no-monitoring     Disable monitoring deployment"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    production|staging|development|edge)
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT"
        log_error "Valid environments: production, staging, development, edge"
        exit 1
        ;;
esac

# Export variables for envsubst
export DOCKER_IMAGE DOCKER_TAG NAMESPACE ENVIRONMENT DEPLOY_REGION

# Run main deployment
main "$@"