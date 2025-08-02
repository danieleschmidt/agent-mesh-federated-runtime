# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the Agent Mesh Federated Runtime project.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Configuration](#repository-configuration)
- [Workflow Installation](#workflow-installation)
- [Secrets Configuration](#secrets-configuration)
- [Environment Setup](#environment-setup)
- [Branch Protection](#branch-protection)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)

## 🔧 Prerequisites

Before setting up workflows, ensure you have:

- **Repository Admin Access**: Required to configure secrets and settings
- **GitHub Actions Enabled**: Check repository settings > Actions
- **Container Registry Access**: For Docker image publishing
- **Package Registry Access**: For Python package publishing (PyPI)
- **Kubernetes Access**: For deployment workflows (if applicable)

## ⚙️ Repository Configuration

### 1. Enable GitHub Actions

1. Navigate to your repository settings
2. Go to **Actions** > **General**
3. Select **Allow all actions and reusable workflows**
4. Enable **Allow GitHub Actions to create and approve pull requests**

### 2. Configure Dependabot (Optional)

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "python"
  
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "frontend-team"
    labels:
      - "dependencies"
      - "nodejs"
  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"
    labels:
      - "dependencies"
      - "docker"
  
  - package-ecosystem: "github-actions"
    directory: ".github/workflows"
    schedule:
      interval: "weekly"
    reviewers:
      - "devops-team"
    labels:
      - "dependencies"
      - "github-actions"
```

## 📁 Workflow Installation

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files

Copy the workflow files from the examples directory:

```bash
# Core workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/

# Optional workflows
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
```

### Step 3: Customize Workflow Configuration

Edit each workflow file to match your specific requirements:

#### CI Workflow Customization

1. **Update branch patterns**:
   ```yaml
   on:
     push:
       branches: [ main, develop, feature/* ]  # Add your branch patterns
   ```

2. **Adjust Python versions**:
   ```yaml
   strategy:
     matrix:
       python-version: ['3.9', '3.10', '3.11']  # Your supported versions
   ```

3. **Configure test commands**:
   ```yaml
   - name: Run tests
     run: |
       pytest tests/ --cov=src --cov-report=xml  # Your test command
   ```

#### CD Workflow Customization

1. **Update deployment targets**:
   ```yaml
   environment:
     name: production
     url: https://your-domain.com  # Your production URL
   ```

2. **Configure container registry**:
   ```yaml
   env:
     REGISTRY: ghcr.io  # or docker.io, your-registry.com
     IMAGE_NAME: your-org/agent-mesh
   ```

## 🔐 Secrets Configuration

### Required Secrets

Navigate to **Settings** > **Secrets and variables** > **Actions** and add:

#### Core Secrets

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `GITHUB_TOKEN` | Automatically provided | (auto-generated) |
| `GPG_PRIVATE_KEY` | GPG key for signing | (your GPG private key) |
| `GPG_PASSPHRASE` | GPG key passphrase | (your passphrase) |

#### Package Publishing

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `PYPI_API_TOKEN` | PyPI publishing token | Python package releases |
| `NPM_TOKEN` | npm publishing token | Node.js package releases |
| `DOCKER_PASSWORD` | Docker registry password | Container publishing |

#### Deployment Secrets

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `KUBE_CONFIG_STAGING` | Base64 kubeconfig for staging | Kubernetes deployments |
| `KUBE_CONFIG_PRODUCTION` | Base64 kubeconfig for production | Kubernetes deployments |
| `AWS_ACCESS_KEY_ID` | AWS access key | AWS deployments |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | AWS deployments |

#### Notification Secrets

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `SLACK_WEBHOOK_URL` | Slack webhook URL | Slack notifications |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL | Discord notifications |
| `PAGERDUTY_API_KEY` | PagerDuty API key | Critical alerts |

#### Security Scanning

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `SNYK_TOKEN` | Snyk authentication token | Vulnerability scanning |
| `SONAR_TOKEN` | SonarCloud token | Code quality analysis |
| `CODECOV_TOKEN` | Codecov upload token | Coverage reporting |

### Generating Secrets

#### GPG Key for Signing

```bash
# Generate GPG key
gpg --full-generate-key

# Export private key (base64 encoded)
gpg --armor --export-secret-keys YOUR_KEY_ID | base64 -w 0

# Export public key
gpg --armor --export YOUR_KEY_ID
```

#### Kubernetes Config

```bash
# Encode kubeconfig for GitHub secrets
base64 -w 0 ~/.kube/config-staging

# Or create specific service account
kubectl create serviceaccount github-actions
kubectl create clusterrolebinding github-actions --clusterrole=cluster-admin --serviceaccount=default:github-actions
```

#### PyPI Token

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create API token with scope for your project
3. Copy token starting with `pypi-`

## 🌍 Environment Setup

### Creating Environments

1. Go to **Settings** > **Environments**
2. Create environments: `development`, `staging`, `production`
3. Configure protection rules for each environment

#### Environment Protection Rules

**Staging Environment**:
- Required reviewers: 1 team member
- Wait timer: 0 minutes
- Deployment branches: `main`, `develop`

**Production Environment**:
- Required reviewers: 2 senior team members
- Wait timer: 5 minutes
- Deployment branches: `main` only

#### Environment Secrets

Add environment-specific secrets:

```yaml
# Staging
ENVIRONMENT: staging
DATABASE_URL: postgresql://staging-db
API_URL: https://api.staging.agent-mesh.com

# Production  
ENVIRONMENT: production
DATABASE_URL: postgresql://prod-db
API_URL: https://api.agent-mesh.com
```

## 🛡️ Branch Protection

### Main Branch Protection

1. Navigate to **Settings** > **Branches**
2. Add rule for `main` branch with:

#### Required Settings

- ✅ **Require a pull request before merging**
  - Required approving reviews: 2
  - Dismiss stale reviews when new commits are pushed
  - Require review from code owners

- ✅ **Require status checks to pass before merging**
  - Require branches to be up to date before merging
  - Required status checks:
    - `CI / Code Quality & Linting`
    - `CI / Security Scanning`
    - `CI / Unit Tests`
    - `CI / Integration Tests`
    - `CI / Build Validation`

- ✅ **Require signed commits**
- ✅ **Require linear history**
- ✅ **Include administrators**

#### Optional Settings

- ✅ **Allow force pushes** (disabled)
- ✅ **Allow deletions** (disabled)
- ✅ **Require deployments to succeed before merging**

### Development Branch Protection

For `develop` branch:
- Required approving reviews: 1
- Required status checks: CI workflows only
- Allow force pushes for maintainers

## 📊 Monitoring & Troubleshooting

### Workflow Monitoring

#### GitHub Actions Dashboard

Monitor workflows at: `https://github.com/your-org/agent-mesh/actions`

#### Key Metrics to Watch

1. **Success Rate**: Target >95% for CI workflows
2. **Duration**: Monitor for performance regression
3. **Queue Time**: Check for resource contention
4. **Failure Patterns**: Identify common failure points

#### Setting Up Alerts

Create alerts for:
- Workflow failures on main branch
- Security scan failures
- Deployment failures
- Long-running workflows (>30 minutes)

### Common Issues & Solutions

#### 1. Permission Errors

**Symptom**: `403 Forbidden` or `Permission denied`

**Solutions**:
- Check GITHUB_TOKEN permissions
- Verify repository settings allow Actions
- Ensure service accounts have correct roles

#### 2. Secret Access Issues

**Symptom**: `Secret not found` or empty values

**Solutions**:
- Verify secret names match exactly (case-sensitive)
- Check environment-specific secrets
- Ensure secrets are not empty or contain newlines

#### 3. Dependency Installation Failures

**Symptom**: Package installation timeouts or failures

**Solutions**:
- Add caching for dependencies
- Use specific package versions
- Configure private registry access

#### 4. Docker Build Failures

**Symptom**: Docker build timeouts or out of space

**Solutions**:
- Enable Docker layer caching
- Clean up intermediate containers
- Use multi-stage builds for smaller images

#### 5. Test Failures in CI Only

**Symptom**: Tests pass locally but fail in CI

**Solutions**:
- Check environment differences
- Review test isolation and dependencies
- Verify service dependencies are available

### Debugging Workflows

#### Enable Debug Logging

Add to workflow environment:

```yaml
env:
  ACTIONS_RUNNER_DEBUG: true
  ACTIONS_STEP_DEBUG: true
```

#### SSH Access for Debugging

Add debugging step:

```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 15
```

#### Log Analysis

Use workflow run logs to:
1. Identify failure points
2. Check resource usage
3. Verify environment setup
4. Review command outputs

### Performance Optimization

#### Workflow Optimization Tips

1. **Parallel Execution**: Use matrix strategies
2. **Caching**: Cache dependencies and build artifacts
3. **Conditional Steps**: Skip unnecessary steps
4. **Resource Management**: Right-size runners
5. **Artifact Management**: Clean up old artifacts

#### Example Optimizations

```yaml
# Conditional test execution
- name: Run integration tests
  if: contains(github.event.head_commit.message, '[integration]') || github.event_name == 'push'
  run: pytest tests/integration/

# Dependency caching
- name: Cache pip dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

# Parallel test execution
strategy:
  matrix:
    test-group: [unit, integration, e2e]
  fail-fast: false
```

## 🚀 Advanced Configuration

### Custom Actions

Create reusable actions in `.github/actions/`:

```yaml
# .github/actions/setup-agent-mesh/action.yml
name: 'Setup Agent Mesh Environment'
description: 'Setup Python, install dependencies, and configure environment'
inputs:
  python-version:
    description: 'Python version to use'
    required: false
    default: '3.11'
runs:
  using: 'composite'
  steps:
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ inputs.python-version }}
        cache: 'pip'
    - run: pip install -e ".[dev]"
      shell: bash
```

### Workflow Templates

Create organization-wide workflow templates in `.github/workflow-templates/`:

```yaml
# .github/workflow-templates/python-ci.yml
name: Python CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: ./.github/actions/setup-agent-mesh
    - run: pytest
```

### Integration with External Tools

#### SonarCloud Integration

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

#### Terraform Integration

```yaml
- name: Terraform Plan
  uses: hashicorp/terraform-github-actions@master
  with:
    tf_actions_version: 1.0.0
    tf_actions_subcommand: 'plan'
  env:
    TF_VAR_github_token: ${{ secrets.GITHUB_TOKEN }}
```

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening Guide](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
- [Monitoring and Troubleshooting](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review GitHub Actions logs
3. Consult the team documentation
4. Create an issue in the repository
5. Contact the DevOps team

---

**Note**: This guide assumes you have the necessary permissions and access to configure GitHub repository settings. Adjust the configuration based on your organization's specific requirements and security policies.