# Workflow Manual Setup Guide

This guide covers the manual setup steps required for full CI/CD implementation due to GitHub App permission limitations.

## Quick Setup Checklist

### 1. Copy Workflow Files (Required)
```bash
# Copy all workflow templates to active directory
cp docs/workflows/examples/*.yml .github/workflows/

# Verify files copied correctly
ls -la .github/workflows/
```

### 2. Configure Repository Settings
Go to repository Settings and configure:

**Branch Protection** (Settings > Branches):
- ✅ Require pull request reviews (1 reviewer minimum)
- ✅ Enable status checks
- ✅ Require conversation resolution
- ✅ Include administrators

**Actions** (Settings > Actions):
- ✅ Allow GitHub Actions
- ✅ Allow actions created by GitHub
- ✅ Enable dependency graph
- ✅ Enable Dependabot alerts

### 3. Add Repository Secrets
Go to Settings > Secrets and variables > Actions:

**Required Secrets**:
```
DOCKER_REGISTRY_TOKEN    # Container registry authentication
SECURITY_SCAN_TOKEN      # Security scanning service token
DEPLOYMENT_KEY           # Production deployment SSH key
```

**Optional Secrets**:
```
SLACK_WEBHOOK_URL        # For build notifications
TEAMS_WEBHOOK_URL        # Alternative notification channel
CODECOV_TOKEN            # Code coverage reporting
```

### 4. Create Environments
Go to Settings > Environments and create:

1. **development**
   - No approval required
   - Auto-deploy on feature branches

2. **staging**  
   - Require reviewers: 1 person
   - Deployment branch: develop

3. **production**
   - Require reviewers: 2+ people
   - Deployment branch: main only

### 5. Configure Notifications
**Slack Integration**:
- Install GitHub app in Slack workspace
- Configure channel notifications for:
  - Pull request reviews
  - Deployment status
  - Security alerts

## Validation Steps

After setup, verify everything works:

```bash
# Test CI workflow
git checkout -b test/workflow-validation
echo "# Test" > test.md
git add test.md && git commit -m "test: validate workflow"
git push origin test/workflow-validation

# Create test PR and verify:
# - CI workflow runs
# - Security scans execute
# - Branch protection rules apply
```

## Troubleshooting

**Common Issues**:

1. **Workflow not triggering**
   - Check `.github/workflows/` directory exists
   - Verify YAML syntax with `yamllint`
   - Ensure branch protection allows actions

2. **Secrets not accessible**
   - Verify secret names match workflow files exactly
   - Check repository permissions for secrets access

3. **Environment deployment failing**
   - Confirm environment exists and is configured
   - Verify deployment keys have correct permissions

**Debug Commands**:
```bash
# Check workflow files
find .github/workflows -name "*.yml" -exec yamllint {} \;

# Verify git configuration  
git config --list | grep -E "(user|remote)"

# Test secret access (safely)
echo "Secrets configured: $(gh secret list | wc -l)" 
```

## Support

- Review [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines
- Check existing workflow runs in Actions tab
- Create issue with `workflow` label for assistance
- Contact maintainers via Discord: [Join our community](https://discord.gg/agent-mesh)

---

**Quick Links**:
- [Workflow Examples](examples/)
- [Security Documentation](../security-workflows.md)
- [Deployment Guide](../deployment-guide.md)