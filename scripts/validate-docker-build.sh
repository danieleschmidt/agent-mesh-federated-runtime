#!/bin/bash
# Docker Build Validation Script
# Validates Docker build configuration and performs test builds

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}🐳 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Validate Dockerfile
validate_dockerfile() {
    local dockerfile="${1:-Dockerfile}"
    
    print_status "Validating $dockerfile..."
    
    if [[ ! -f "$dockerfile" ]]; then
        print_error "Dockerfile '$dockerfile' not found"
        return 1
    fi
    
    # Check for required instructions
    local required_instructions=("FROM" "WORKDIR" "COPY" "RUN")
    for instruction in "${required_instructions[@]}"; do
        if ! grep -q "^$instruction" "$dockerfile"; then
            print_warning "$dockerfile missing recommended instruction: $instruction"
        fi
    done
    
    # Check for security best practices
    if grep -q "^USER root" "$dockerfile"; then
        print_warning "$dockerfile runs as root user (security risk)"
    fi
    
    if grep -q "ADD http" "$dockerfile"; then
        print_warning "$dockerfile uses ADD with URL (prefer COPY or RUN + curl)"
    fi
    
    # Check for multi-stage build
    local from_count=$(grep -c "^FROM" "$dockerfile")
    if [[ $from_count -gt 1 ]]; then
        print_success "$dockerfile uses multi-stage build"
    else
        print_warning "$dockerfile could benefit from multi-stage build"
    fi
    
    print_success "$dockerfile validation completed"
}

# Validate .dockerignore
validate_dockerignore() {
    print_status "Validating .dockerignore..."
    
    if [[ ! -f ".dockerignore" ]]; then
        print_warning ".dockerignore not found - build context may be large"
        return 0
    fi
    
    # Check for common patterns
    local recommended_patterns=(
        "*.pyc"
        "__pycache__"
        ".git"
        ".pytest_cache"
        "node_modules"
        "*.log"
    )
    
    for pattern in "${recommended_patterns[@]}"; do
        if ! grep -q "$pattern" .dockerignore; then
            print_warning ".dockerignore missing recommended pattern: $pattern"
        fi
    done
    
    print_success ".dockerignore validation completed"
}

# Validate docker-compose files
validate_docker_compose() {
    print_status "Validating docker-compose files..."
    
    local compose_files=(
        "docker-compose.yml"
        "docker-compose.yaml"
        "docker-compose.dev.yml"
        "docker-compose.prod.yml"
        "docker-compose.override.yml"
    )
    
    local found_compose=false
    
    for compose_file in "${compose_files[@]}"; do
        if [[ -f "$compose_file" ]]; then
            found_compose=true
            print_status "Validating $compose_file..."
            
            # Basic YAML validation
            if command -v yq >/dev/null 2>&1; then
                if yq eval '.' "$compose_file" >/dev/null 2>&1; then
                    print_success "$compose_file is valid YAML"
                else
                    print_error "$compose_file contains invalid YAML"
                    return 1
                fi
            elif python3 -c "import yaml" 2>/dev/null; then
                if python3 -c "import yaml; yaml.safe_load(open('$compose_file'))" 2>/dev/null; then
                    print_success "$compose_file is valid YAML"
                else
                    print_error "$compose_file contains invalid YAML"
                    return 1
                fi
            fi
            
            # Check for version specification
            if grep -q "^version:" "$compose_file"; then
                print_success "$compose_file specifies compose version"
            else
                print_warning "$compose_file missing version specification"
            fi
            
            # Check for health checks
            if grep -q "healthcheck:" "$compose_file"; then
                print_success "$compose_file includes health checks"
            else
                print_warning "$compose_file could benefit from health checks"
            fi
        fi
    done
    
    if [[ "$found_compose" == false ]]; then
        print_warning "No docker-compose files found"
    fi
}

# Test Docker build
test_docker_build() {
    local dockerfile="${1:-Dockerfile}"
    local build_args="${2:-}"
    
    print_status "Testing Docker build with $dockerfile..."
    
    # Generate a unique tag for testing
    local test_tag="agent-mesh-test:$(date +%s)"
    
    # Attempt to build
    if docker build $build_args -t "$test_tag" -f "$dockerfile" . >/dev/null 2>&1; then
        print_success "Docker build test successful"
        
        # Clean up test image
        docker rmi "$test_tag" >/dev/null 2>&1 || true
        
        return 0
    else
        print_error "Docker build test failed"
        
        # Show detailed error
        print_status "Attempting build with verbose output..."
        docker build $build_args -t "$test_tag" -f "$dockerfile" .
        
        return 1
    fi
}

# Check Docker daemon availability
check_docker_daemon() {
    print_status "Checking Docker daemon availability..."
    
    if ! docker info >/dev/null 2>&1; then
        print_warning "Docker daemon not available - skipping build tests"
        return 1
    fi
    
    print_success "Docker daemon is available"
    return 0
}

# Check build context size
check_build_context_size() {
    print_status "Checking build context size..."
    
    # Calculate size of build context
    local context_size=$(du -sh . 2>/dev/null | cut -f1)
    print_status "Build context size: $context_size"
    
    # Get size in MB for comparison
    local size_mb=$(du -sm . 2>/dev/null | cut -f1)
    
    if [[ $size_mb -gt 500 ]]; then
        print_warning "Build context is large (${context_size}). Consider improving .dockerignore"
    elif [[ $size_mb -gt 100 ]]; then
        print_status "Build context size is moderate (${context_size})"
    else
        print_success "Build context size is good (${context_size})"
    fi
}

# Check for Hadolint (Dockerfile linter)
run_hadolint() {
    local dockerfile="${1:-Dockerfile}"
    
    if command -v hadolint >/dev/null 2>&1; then
        print_status "Running Hadolint on $dockerfile..."
        
        if hadolint "$dockerfile"; then
            print_success "Hadolint passed"
        else
            print_warning "Hadolint found issues (not blocking)"
        fi
    else
        print_warning "Hadolint not installed - skipping Dockerfile linting"
    fi
}

# Main function
main() {
    local dockerfile="${1:-Dockerfile}"
    local test_build="${2:-true}"
    local exit_code=0
    
    print_status "Starting Docker build validation..."
    
    # Validate Dockerfile
    if ! validate_dockerfile "$dockerfile"; then
        exit_code=1
    fi
    
    # Validate .dockerignore
    validate_dockerignore
    
    # Validate docker-compose files
    if ! validate_docker_compose; then
        exit_code=1
    fi
    
    # Check build context size
    check_build_context_size
    
    # Run Hadolint if available
    run_hadolint "$dockerfile"
    
    # Test actual build if Docker is available and requested
    if [[ "$test_build" == "true" ]] && check_docker_daemon; then
        if ! test_docker_build "$dockerfile"; then
            exit_code=1
        fi
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Docker build validation completed successfully"
    else
        print_error "Docker build validation failed"
    fi
    
    return $exit_code
}

# Handle command line arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse arguments
    dockerfile="Dockerfile"
    test_build="true"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--file)
                dockerfile="$2"
                shift 2
                ;;
            --no-build-test)
                test_build="false"
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -f, --file DOCKERFILE    Dockerfile to validate (default: Dockerfile)"
                echo "  --no-build-test         Skip actual build test"
                echo "  -h, --help              Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    main "$dockerfile" "$test_build"
fi