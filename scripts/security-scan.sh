#!/bin/bash
# Security scanning script for Agent Mesh Federated Runtime
# This script runs comprehensive security checks across the codebase

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECURITY_DIR="$PROJECT_ROOT/security"

# Create security directory if it doesn't exist
mkdir -p "$SECURITY_DIR"

echo -e "${GREEN}🔒 Starting comprehensive security scan...${NC}"

# Function to run a security tool and capture output
run_security_tool() {
    local tool_name="$1"
    local command="$2"
    local output_file="$3"
    
    echo -e "${YELLOW}Running $tool_name...${NC}"
    
    if eval "$command" > "$output_file" 2>&1; then
        echo -e "${GREEN}✓ $tool_name completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $tool_name found issues${NC}"
        return 1
    fi
}

# Python security scanning with Bandit
echo -e "${YELLOW}🐍 Python Security Analysis${NC}"
run_security_tool "Bandit" \
    "bandit -r src/ -f json" \
    "$SECURITY_DIR/bandit-report.json"

# Dependency vulnerability scanning with Safety  
run_security_tool "Safety" \
    "safety check --json" \
    "$SECURITY_DIR/safety-report.json"

# Container security scanning (if Dockerfile exists)
if [[ -f "$PROJECT_ROOT/Dockerfile" ]]; then
    echo -e "${YELLOW}🐳 Container Security Analysis${NC}"
    
    # Build image for scanning
    docker build -t agent-mesh-security-scan:latest "$PROJECT_ROOT" > /dev/null 2>&1
    
    # Trivy container scan
    if command -v trivy >/dev/null 2>&1; then
        run_security_tool "Trivy Container Scan" \
            "trivy image --format json agent-mesh-security-scan:latest" \
            "$SECURITY_DIR/trivy-container-report.json"
    fi
    
    # Clean up
    docker rmi agent-mesh-security-scan:latest > /dev/null 2>&1 || true
fi

# Secret detection
echo -e "${YELLOW}🔐 Secret Detection${NC}"
run_security_tool "Detect Secrets" \
    "detect-secrets scan --baseline .secrets.baseline --force-use-all-plugins" \
    "$SECURITY_DIR/secrets-scan.json"

# YAML security scanning
echo -e "${YELLOW}📄 Configuration Security Analysis${NC}"
find "$PROJECT_ROOT" -name "*.yml" -o -name "*.yaml" | grep -v node_modules | while read -r file; do
    if [[ -f "$file" ]]; then
        echo "Scanning: $file"
        yamllint "$file" >> "$SECURITY_DIR/yaml-security.log" 2>&1 || true
    fi
done

# License compliance check
echo -e "${YELLOW}📋 License Compliance Check${NC}"
if command -v pip-licenses >/dev/null 2>&1; then
    pip-licenses --format=json --output-file="$SECURITY_DIR/license-report.json" || true
fi

# Generate SBOM (Software Bill of Materials)
echo -e "${YELLOW}📊 Generating Software Bill of Materials${NC}"
if command -v syft >/dev/null 2>&1; then
    syft "$PROJECT_ROOT" -o spdx-json > "$SECURITY_DIR/sbom.spdx.json" 2>/dev/null || true
fi

# Network security analysis for exposed services
echo -e "${YELLOW}🌐 Network Security Analysis${NC}"
if [[ -f "$PROJECT_ROOT/docker-compose.yml" ]]; then
    echo "Analyzing exposed ports in docker-compose.yml..."
    grep -n "ports:" "$PROJECT_ROOT/docker-compose.yml" > "$SECURITY_DIR/exposed-ports.txt" 2>/dev/null || true
fi

# Compliance checks
echo -e "${YELLOW}✅ Compliance Validation${NC}"
{
    echo "# Security Compliance Report - $(date)"
    echo ""
    echo "## Files Scanned"
    find "$PROJECT_ROOT" -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) | wc -l
    echo ""
    echo "## Security Tools Run"
    echo "- Bandit (Python security)"
    echo "- Safety (Dependency vulnerabilities)"
    echo "- Detect-secrets (Secret detection)"
    echo "- YAML Lint (Configuration security)"
    if command -v trivy >/dev/null 2>&1; then
        echo "- Trivy (Container security)"
    fi
    if command -v syft >/dev/null 2>&1; then
        echo "- Syft (SBOM generation)"
    fi
} > "$SECURITY_DIR/compliance-report.md"

# Summary report
echo -e "${GREEN}📊 Security Scan Summary${NC}"
echo "Reports generated in: $SECURITY_DIR"
echo "Files created:"
ls -la "$SECURITY_DIR/"

# Check for critical issues
critical_issues=0

# Check Bandit results
if [[ -f "$SECURITY_DIR/bandit-report.json" ]]; then
    high_severity=$(jq '.results[] | select(.issue_severity == "HIGH") | length' "$SECURITY_DIR/bandit-report.json" 2>/dev/null | wc -l)
    if [[ "$high_severity" -gt 0 ]]; then
        echo -e "${RED}⚠️  Found $high_severity high-severity security issues${NC}"
        critical_issues=$((critical_issues + high_severity))
    fi
fi

# Check Safety results
if [[ -f "$SECURITY_DIR/safety-report.json" ]]; then
    vulnerabilities=$(jq '.vulnerabilities | length' "$SECURITY_DIR/safety-report.json" 2>/dev/null || echo "0")
    if [[ "$vulnerabilities" -gt 0 ]]; then
        echo -e "${RED}⚠️  Found $vulnerabilities dependency vulnerabilities${NC}"
        critical_issues=$((critical_issues + vulnerabilities))
    fi
fi

# Exit with appropriate code
if [[ "$critical_issues" -gt 0 ]]; then
    echo -e "${RED}❌ Security scan completed with $critical_issues critical issues${NC}"
    echo "Review the reports in $SECURITY_DIR for details"
    exit 1
else
    echo -e "${GREEN}✅ Security scan completed successfully with no critical issues${NC}"
    exit 0
fi