# CI/CD Workflows and Automation

This directory contains comprehensive CI/CD workflow documentation and templates for the Agent Mesh Federated Runtime project.

## Directory Structure

```
docs/workflows/
├── README.md                    # This file
├── ci-cd-strategy.md           # Overall CI/CD strategy and architecture
├── deployment-guide.md         # Deployment procedures and best practices
├── security-workflows.md       # Security scanning and compliance workflows
└── examples/                   # Workflow template files
    ├── ci.yml                  # Continuous Integration workflow
    ├── cd.yml                  # Continuous Deployment workflow
    ├── security-scan.yml       # Security scanning workflow
    ├── dependency-update.yml   # Automated dependency updates
    ├── performance-test.yml    # Performance testing workflow
    └── release.yml             # Release automation workflow
```

## Quick Start

Due to GitHub App permission limitations, workflows cannot be created automatically. Repository maintainers must manually copy workflow files from `examples/` to `.github/workflows/` after reviewing and customizing them.

### Manual Setup Required

1. **Copy Workflow Files**:
   ```bash
   cp docs/workflows/examples/*.yml .github/workflows/
   ```

2. **Configure Secrets**:
   - `GITHUB_TOKEN`: Automatically provided by GitHub
   - `DOCKER_REGISTRY_TOKEN`: For container image publishing
   - `SECURITY_SCAN_TOKEN`: For security scanning services
   - `DEPLOYMENT_KEY`: For production deployments

3. **Configure Branch Protection**:
   - Require PR reviews before merging
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators in restrictions

4. **Configure Environments**:
   - `development`: Automatic deployment on feature branches
   - `staging`: Manual approval required
   - `production`: Manual approval + additional reviewers

## Workflow Overview

### Continuous Integration (CI)

Triggered on: Pull requests, pushes to main/develop branches

**Pipeline Steps**:
1. **Code Quality**
   - Linting (flake8, mypy, eslint)
   - Code formatting validation (black, prettier)
   - Security scanning (bandit, safety)
   - Dependency vulnerability checks

2. **Testing**
   - Unit tests with coverage reporting
   - Integration tests with real services
   - End-to-end tests with Docker Compose
   - Performance regression tests

3. **Build Validation**
   - Python package building
   - Docker image building
   - Documentation generation
   - API documentation validation

4. **Security & Compliance**
   - Container image scanning
   - SBOM (Software Bill of Materials) generation
   - License compliance checking
   - Secrets detection

### Continuous Deployment (CD)

Triggered on: Successful CI on main/develop branches, manual deployment triggers

**Pipeline Steps**:
1. **Build & Package**
   - Build production Docker images
   - Sign container images
   - Push to container registry
   - Generate deployment artifacts

2. **Deploy to Staging**
   - Deploy to staging environment
   - Run smoke tests
   - Performance validation
   - Security validation

3. **Production Deployment** (Manual Approval)
   - Blue-green deployment strategy
   - Health checks and monitoring
   - Rollback capability
   - Post-deployment validation

### Security Workflows

**Scheduled Security Scans**:
- Daily dependency vulnerability scans
- Weekly container image scans
- Monthly penetration testing
- Quarterly security audit

**Incident Response**:
- Automated security alert triage
- Emergency patching workflow
- Security incident documentation
- Post-incident review process

## Workflow Configuration

### Environment Variables

```yaml
# Common environment variables used across workflows
env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'
  DOCKER_BUILDKIT: '1'
  REGISTRY: ghcr.io
  IMAGE_NAME: agent-mesh-federated-runtime
  COVERAGE_THRESHOLD: '90'
  SECURITY_SCAN_THRESHOLD: 'medium'
```

### Matrix Testing

```yaml
# Python version matrix
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]
    include:
      - python-version: '3.11'
        os: ubuntu-latest
        coverage: true
```

### Caching Strategy

```yaml
# Dependency caching
- uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pre-commit
      ~/.npm
      ~/.docker/buildx-cache
    key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements*.txt', '**/package*.json') }}
```

## Deployment Strategies

### Development Environment
- **Trigger**: Push to feature branches
- **Strategy**: Direct deployment
- **Validation**: Basic smoke tests
- **Rollback**: Automatic on failure

### Staging Environment
- **Trigger**: Push to develop branch
- **Strategy**: Blue-green deployment
- **Validation**: Full test suite
- **Rollback**: Manual trigger

### Production Environment
- **Trigger**: Manual approval after staging validation
- **Strategy**: Rolling deployment with canary analysis
- **Validation**: Health checks + performance monitoring
- **Rollback**: Automatic on health check failure

## Monitoring and Observability

### Workflow Metrics
- Build success/failure rates
- Deployment frequency
- Lead time for changes
- Mean time to recovery (MTTR)
- Test coverage trends

### Alerting
- Build failures → Slack notifications
- Security vulnerabilities → Email alerts
- Deployment failures → PagerDuty incidents
- Performance regressions → Team notifications

## Security Considerations

### Secrets Management
- Use GitHub Secrets for sensitive data
- Rotate secrets regularly
- Audit secret access
- Use environment-specific secrets

### Access Control
- Require signed commits
- Branch protection rules
- Required reviews from code owners
- Restrict workflow modifications

### Supply Chain Security
- Pin action versions to specific SHAs
- Verify action signatures
- Use official actions where possible
- Regular dependency updates

## Performance Optimization

### Parallel Execution
- Run independent jobs in parallel
- Use matrix builds for multi-platform testing
- Optimize Docker layer caching
- Parallelize test execution

### Resource Management
- Use appropriate runner sizes
- Optimize container images
- Clean up temporary resources
- Monitor resource usage

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check dependency conflicts
   - Verify environment variables
   - Review recent changes
   - Check resource limits

2. **Test Failures**
   - Review test logs
   - Check for flaky tests
   - Verify test data
   - Check service dependencies

3. **Deployment Issues**
   - Verify secrets configuration
   - Check network connectivity
   - Review deployment logs
   - Validate environment setup

### Debug Information

```yaml
# Enable debug logging
- name: Debug Information
  run: |
    echo "Runner OS: ${{ runner.os }}"
    echo "Runner Architecture: ${{ runner.arch }}"
    echo "GitHub Ref: ${{ github.ref }}"
    echo "GitHub SHA: ${{ github.sha }}"
    env
```

## Best Practices

### Workflow Design
- Keep workflows focused and modular
- Use reusable workflows for common patterns
- Implement proper error handling
- Add meaningful step names and descriptions

### Testing Strategy
- Fast feedback with unit tests
- Comprehensive integration testing
- Performance regression detection
- Security validation at every stage

### Documentation
- Document all workflow inputs/outputs
- Maintain up-to-date README files
- Include troubleshooting guides
- Document emergency procedures

## Maintenance

### Regular Tasks
- Review and update workflow dependencies
- Optimize workflow performance
- Update security scanning rules
- Review and update branch protection rules

### Quarterly Reviews
- Analyze workflow metrics
- Review security posture
- Update documentation
- Plan workflow improvements

## Support

For workflow-related issues:
1. Check the troubleshooting section above
2. Review workflow run logs in GitHub Actions
3. Consult the team in Slack #devops channel
4. Create an issue with the `workflow` label

---

**Last Updated**: 2024-07-28  
**Maintained By**: DevOps Team  
**Review Schedule**: Monthly