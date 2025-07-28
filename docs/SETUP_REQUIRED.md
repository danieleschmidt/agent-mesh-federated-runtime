# Manual Setup Requirements

Due to GitHub App permission limitations, the following setup steps require manual configuration by repository maintainers.

## GitHub Actions Workflows

**Action Required**: Copy workflow files from examples to active directory
```bash
cp docs/workflows/examples/*.yml .github/workflows/
```

**Files to copy**:
- `ci.yml` - Continuous Integration workflow
- `security-scan.yml` - Security scanning workflow

## Repository Settings

### Branch Protection Rules
Configure for `main` branch:
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

### Repository Secrets
Add the following secrets in Settings > Secrets and variables > Actions:
- `DOCKER_REGISTRY_TOKEN` - For container publishing
- `SECURITY_SCAN_TOKEN` - For security services
- `DEPLOYMENT_KEY` - For production deployments

### Environments
Create these environments in Settings > Environments:
1. **development** - Automatic deployment
2. **staging** - Manual approval required  
3. **production** - Manual approval + reviewers

## External Integrations

### Security Scanning
- Enable Dependabot alerts
- Configure CodeQL analysis
- Setup container vulnerability scanning

### Monitoring
- Connect monitoring services (Grafana, Prometheus)
- Configure alerting endpoints
- Setup log aggregation

## Dependencies Installation

Repository maintainers should run:
```bash
npm run setup  # Installs all dependencies and pre-commit hooks
```

## Validation

After manual setup, verify with:
```bash
npm run test     # Run all tests
npm run lint     # Check code quality
npm run security # Security validation
```

For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md) or create an issue.
