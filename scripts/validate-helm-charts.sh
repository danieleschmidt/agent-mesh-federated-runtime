#!/bin/bash
# Helm Chart Validation Script
# Validates Helm charts for correctness, security, and best practices

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}⎈ $1${NC}"
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

# Check if Helm is installed
check_helm_installation() {
    if ! command -v helm >/dev/null 2>&1; then
        print_error "Helm is not installed"
        return 1
    fi
    
    local helm_version=$(helm version --short --client 2>/dev/null || helm version --short 2>/dev/null)
    print_success "Helm is installed: $helm_version"
    return 0
}

# Find Helm charts in the repository
find_helm_charts() {
    local charts=()
    
    # Look for Chart.yaml files
    while IFS= read -r -d '' chart_file; do
        local chart_dir=$(dirname "$chart_file")
        charts+=("$chart_dir")
    done < <(find . -name "Chart.yaml" -type f -print0 2>/dev/null)
    
    printf '%s\n' "${charts[@]}"
}

# Validate Chart.yaml
validate_chart_yaml() {
    local chart_dir="$1"
    local chart_yaml="$chart_dir/Chart.yaml"
    
    print_status "Validating $chart_yaml..."
    
    if [[ ! -f "$chart_yaml" ]]; then
        print_error "Chart.yaml not found in $chart_dir"
        return 1
    fi
    
    # Check required fields
    local required_fields=("name" "version" "apiVersion")
    local missing_fields=()
    
    for field in "${required_fields[@]}"; do
        if ! grep -q "^$field:" "$chart_yaml"; then
            missing_fields+=("$field")
        fi
    done
    
    if [[ ${#missing_fields[@]} -gt 0 ]]; then
        print_error "Missing required fields in $chart_yaml: ${missing_fields[*]}"
        return 1
    fi
    
    # Check API version
    local api_version=$(grep "^apiVersion:" "$chart_yaml" | cut -d' ' -f2)
    if [[ "$api_version" != "v2" ]]; then
        print_warning "$chart_yaml uses API version $api_version (v2 recommended)"
    fi
    
    # Check for description
    if ! grep -q "^description:" "$chart_yaml"; then
        print_warning "$chart_yaml missing description field"
    fi
    
    # Check for maintainers
    if ! grep -q "^maintainers:" "$chart_yaml"; then
        print_warning "$chart_yaml missing maintainers field"
    fi
    
    print_success "$chart_yaml validation passed"
    return 0
}

# Validate values.yaml
validate_values_yaml() {
    local chart_dir="$1"
    local values_yaml="$chart_dir/values.yaml"
    
    print_status "Validating $values_yaml..."
    
    if [[ ! -f "$values_yaml" ]]; then
        print_warning "values.yaml not found in $chart_dir"
        return 0
    fi
    
    # Check YAML syntax
    if command -v yq >/dev/null 2>&1; then
        if ! yq eval '.' "$values_yaml" >/dev/null 2>&1; then
            print_error "$values_yaml contains invalid YAML"
            return 1
        fi
    elif python3 -c "import yaml" 2>/dev/null; then
        if ! python3 -c "import yaml; yaml.safe_load(open('$values_yaml'))" 2>/dev/null; then
            print_error "$values_yaml contains invalid YAML"
            return 1
        fi
    fi
    
    # Check for security-related configurations
    local security_checks=(
        "securityContext"
        "resources.limits"
        "resources.requests"
        "readinessProbe"
        "livenessProbe"
    )
    
    for check in "${security_checks[@]}"; do
        if grep -q "$check" "$values_yaml"; then
            print_success "$values_yaml includes $check configuration"
        else
            print_warning "$values_yaml missing recommended $check configuration"
        fi
    done
    
    print_success "$values_yaml validation completed"
    return 0
}

# Validate templates directory
validate_templates() {
    local chart_dir="$1"
    local templates_dir="$chart_dir/templates"
    
    print_status "Validating templates in $templates_dir..."
    
    if [[ ! -d "$templates_dir" ]]; then
        print_error "Templates directory not found in $chart_dir"
        return 1
    fi
    
    # Check for required templates
    local required_templates=("deployment.yaml" "service.yaml")
    local found_templates=()
    
    for template in "${required_templates[@]}"; do
        if [[ -f "$templates_dir/$template" ]]; then
            found_templates+=("$template")
        else
            print_warning "Recommended template $template not found in $templates_dir"
        fi
    done
    
    # Check for NOTES.txt
    if [[ -f "$templates_dir/NOTES.txt" ]]; then
        print_success "$templates_dir includes NOTES.txt"
    else
        print_warning "$templates_dir missing NOTES.txt"
    fi
    
    # Count template files
    local template_count=$(find "$templates_dir" -name "*.yaml" -o -name "*.yml" | wc -l)
    print_status "Found $template_count template files in $templates_dir"
    
    return 0
}

# Run Helm lint
run_helm_lint() {
    local chart_dir="$1"
    
    print_status "Running helm lint on $chart_dir..."
    
    if helm lint "$chart_dir" >/dev/null 2>&1; then
        print_success "helm lint passed for $chart_dir"
        return 0
    else
        print_error "helm lint failed for $chart_dir"
        
        # Show detailed output
        print_status "Detailed lint output:"
        helm lint "$chart_dir"
        
        return 1
    fi
}

# Test template rendering
test_template_rendering() {
    local chart_dir="$1"
    
    print_status "Testing template rendering for $chart_dir..."
    
    if helm template test-release "$chart_dir" >/dev/null 2>&1; then
        print_success "Template rendering test passed for $chart_dir"
        return 0
    else
        print_error "Template rendering test failed for $chart_dir"
        
        # Show detailed output
        print_status "Detailed template rendering output:"
        helm template test-release "$chart_dir"
        
        return 1
    fi
}

# Check for security best practices
check_security_practices() {
    local chart_dir="$1"
    local templates_dir="$chart_dir/templates"
    
    print_status "Checking security best practices in $chart_dir..."
    
    local security_issues=0
    
    # Check for runAsRoot
    if grep -r "runAsUser: 0" "$templates_dir" >/dev/null 2>&1; then
        print_warning "Found containers running as root (runAsUser: 0)"
        ((security_issues++))
    fi
    
    # Check for privileged containers
    if grep -r "privileged: true" "$templates_dir" >/dev/null 2>&1; then
        print_warning "Found privileged containers"
        ((security_issues++))
    fi
    
    # Check for missing resource limits
    if ! grep -r "resources:" "$templates_dir" >/dev/null 2>&1; then
        print_warning "No resource limits found in templates"
        ((security_issues++))
    fi
    
    # Check for missing security context
    if ! grep -r "securityContext:" "$templates_dir" >/dev/null 2>&1; then
        print_warning "No security context found in templates"
        ((security_issues++))
    fi
    
    # Check for missing health checks
    if ! grep -r "livenessProbe:" "$templates_dir" >/dev/null 2>&1; then
        print_warning "No liveness probes found in templates"
        ((security_issues++))
    fi
    
    if ! grep -r "readinessProbe:" "$templates_dir" >/dev/null 2>&1; then
        print_warning "No readiness probes found in templates"
        ((security_issues++))
    fi
    
    if [[ $security_issues -eq 0 ]]; then
        print_success "Security best practices check passed"
    else
        print_warning "Found $security_issues potential security issues"
    fi
    
    return 0
}

# Validate dependencies
validate_dependencies() {
    local chart_dir="$1"
    local chart_yaml="$chart_dir/Chart.yaml"
    
    if grep -q "^dependencies:" "$chart_yaml"; then
        print_status "Validating chart dependencies for $chart_dir..."
        
        if helm dependency list "$chart_dir" >/dev/null 2>&1; then
            print_success "Dependencies validation passed"
            
            # Check if dependencies are up to date
            if helm dependency build "$chart_dir" --dry-run >/dev/null 2>&1; then
                print_success "Dependencies are up to date"
            else
                print_warning "Dependencies may need to be updated"
            fi
        else
            print_error "Dependencies validation failed"
            return 1
        fi
    else
        print_status "No dependencies found in $chart_dir"
    fi
    
    return 0
}

# Validate a single chart
validate_chart() {
    local chart_dir="$1"
    local validation_errors=0
    
    print_status "Validating Helm chart: $chart_dir"
    
    # Validate Chart.yaml
    if ! validate_chart_yaml "$chart_dir"; then
        ((validation_errors++))
    fi
    
    # Validate values.yaml
    if ! validate_values_yaml "$chart_dir"; then
        ((validation_errors++))
    fi
    
    # Validate templates
    if ! validate_templates "$chart_dir"; then
        ((validation_errors++))
    fi
    
    # Run Helm lint
    if ! run_helm_lint "$chart_dir"; then
        ((validation_errors++))
    fi
    
    # Test template rendering
    if ! test_template_rendering "$chart_dir"; then
        ((validation_errors++))
    fi
    
    # Check security practices
    check_security_practices "$chart_dir"
    
    # Validate dependencies
    if ! validate_dependencies "$chart_dir"; then
        ((validation_errors++))
    fi
    
    if [[ $validation_errors -eq 0 ]]; then
        print_success "Chart validation passed: $chart_dir"
    else
        print_error "Chart validation failed: $chart_dir ($validation_errors errors)"
    fi
    
    return $validation_errors
}

# Main function
main() {
    local exit_code=0
    
    print_status "Starting Helm chart validation..."
    
    # Check Helm installation
    if ! check_helm_installation; then
        return 1
    fi
    
    # Find all Helm charts
    mapfile -t charts < <(find_helm_charts)
    
    if [[ ${#charts[@]} -eq 0 ]]; then
        print_warning "No Helm charts found in the repository"
        return 0
    fi
    
    print_status "Found ${#charts[@]} Helm chart(s): ${charts[*]}"
    
    # Validate each chart
    for chart in "${charts[@]}"; do
        if ! validate_chart "$chart"; then
            exit_code=1
        fi
        echo  # Add spacing between charts
    done
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "All Helm charts validation completed successfully"
    else
        print_error "Helm charts validation failed"
    fi
    
    return $exit_code
}

# Handle command line arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Validates all Helm charts in the repository"
                echo ""
                echo "Options:"
                echo "  -h, --help              Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    main
fi