# Contributing to Agent Mesh Federated Runtime

Thank you for your interest in contributing to the Agent Mesh Federated Runtime! This project aims to create a robust, decentralized system for federated learning and multi-agent coordination.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Architecture Guidelines](#architecture-guidelines)
- [Testing Requirements](#testing-requirements)
- [Security Considerations](#security-considerations)
- [Documentation Standards](#documentation-standards)
- [Submission Guidelines](#submission-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9+ with asyncio support
- Node.js 18+ and npm 8+ for dashboard components
- Docker and Docker Compose for containerized testing
- Git with LFS support for large test fixtures

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/agent-mesh-federated-runtime.git
   cd agent-mesh-federated-runtime
   ```

2. **Install Dependencies**
   ```bash
   # Install Python dependencies and development tools
   pip install -e ".[dev]"
   
   # Install Node.js dependencies for dashboard
   cd src/web/dashboard && npm install && cd ../../..
   
   # Setup pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run unit tests
   npm run test:unit
   
   # Run linting
   npm run lint
   
   # Start development environment
   npm run dev
   ```

4. **Environment Configuration**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration as needed
   vim .env
   ```

## Development Workflow

### Branching Strategy

We use a GitHub Flow-based approach:

- **main**: Production-ready code, protected branch
- **develop**: Integration branch for ongoing development
- **feature/xxx**: Feature development branches
- **bugfix/xxx**: Bug fix branches
- **hotfix/xxx**: Critical production fixes

### Branch Naming Convention

```
feature/consensus-optimization
bugfix/memory-leak-aggregation
hotfix/security-vulnerability-cvs-2024-001
docs/api-documentation-update
refactor/networking-layer-cleanup
```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `security`: Security-related changes

**Examples:**
```
feat(consensus): implement PBFT optimization for large networks

fix(networking): resolve memory leak in libp2p connection pooling

security(crypto): update to latest NaCl version for vulnerability patch

docs(api): add examples for federated learning configuration
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   - Write code following our coding standards
   - Add comprehensive tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass locally

3. **Pre-submission Checklist**
   - [ ] All tests pass (`npm run test`)
   - [ ] Code follows style guidelines (`npm run lint`)
   - [ ] Security checks pass (`npm run security`)
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated (for significant changes)
   - [ ] No merge conflicts with target branch

4. **Submit Pull Request**
   - Use descriptive title and detailed description
   - Reference related issues using `Fixes #123` or `Closes #456`
   - Add appropriate labels
   - Request review from relevant maintainers

## Architecture Guidelines

### Design Principles

1. **Decentralization First**: No single point of failure or central authority
2. **Security by Design**: All communications encrypted, zero-trust architecture
3. **Fault Tolerance**: Graceful degradation under adverse conditions
4. **Scalability**: Horizontal scaling without performance degradation
5. **Modularity**: Loosely coupled components with clear interfaces

### Code Organization

```
src/
├── core/              # Core system components
│   ├── mesh/         # P2P mesh networking
│   ├── consensus/    # Byzantine fault tolerance
│   └── security/     # Cryptographic components
├── federated/        # Federated learning algorithms
├── agents/           # Multi-agent coordination
├── protocols/        # Network protocol implementations
├── api/              # REST/gRPC API interfaces
└── web/              # Web dashboard and monitoring
```

### Coding Standards

**Python Code Style:**
- Follow PEP 8 with line length of 88 characters (Black default)
- Use type hints for all function signatures
- Comprehensive docstrings following Google style
- Async/await for all I/O operations

**JavaScript/TypeScript:**
- Follow Airbnb style guide
- Use TypeScript for type safety
- Functional components with hooks for React
- ESLint + Prettier for consistent formatting

**Documentation:**
- All public APIs must have comprehensive docstrings
- Include usage examples in docstrings
- Architecture Decision Records (ADRs) for significant design choices
- Mermaid diagrams for complex workflows

## Testing Requirements

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual components in isolation
   - Mock external dependencies
   - Fast execution, comprehensive coverage
   - Target: 95%+ code coverage

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - Use real network connections (controlled environment)
   - Database and external service integration
   - Target: Critical user workflows covered

3. **End-to-End Tests** (`tests/e2e/`)
   - Full system testing with multiple nodes
   - Real-world scenarios and failure conditions
   - Performance and load testing
   - Target: Major user journeys validated

4. **Security Tests** (`tests/security/`)
   - Cryptographic algorithm validation
   - Attack simulation and resistance testing
   - Vulnerability scanning integration
   - Target: Security requirements verified

### Test Guidelines

- **Naming**: Test file names should mirror source files with `test_` prefix
- **Structure**: Use AAA pattern (Arrange, Act, Assert)
- **Fixtures**: Reusable test data in `tests/fixtures/`
- **Mocking**: Use `pytest-mock` for Python, Jest for JavaScript
- **Performance**: Include performance benchmarks for critical paths

### Running Tests

```bash
# Unit tests with coverage
npm run test:unit

# Integration tests
npm run test:integration

# End-to-end tests (requires Docker)
npm run test:e2e

# Performance benchmarks
npm run test:performance

# Security tests
npm run security

# All tests
npm run test
```

## Security Considerations

### Security Review Process

All security-related changes require additional review:

1. **Cryptographic Changes**: Review by security team + external audit
2. **Network Protocol Changes**: Threat modeling and penetration testing
3. **Authentication/Authorization**: RBAC validation and privilege escalation testing
4. **Data Handling**: Privacy impact assessment and compliance review

### Security Guidelines

- **Never log sensitive data**: Keys, tokens, personal information
- **Use secure defaults**: Fail closed, encrypt by default
- **Validate all inputs**: Sanitize and validate external data
- **Principle of least privilege**: Minimal permissions for all components
- **Regular updates**: Keep dependencies current, monitor vulnerabilities

### Vulnerability Reporting

For security vulnerabilities, please follow our [Security Policy](SECURITY.md):
- **Private disclosure**: security@agent-mesh.org
- **GPG key**: Available on project website
- **Response time**: 48 hours acknowledgment, 90 days disclosure

## Documentation Standards

### Required Documentation

1. **API Documentation**: Auto-generated from docstrings with examples
2. **Architecture Decision Records**: Document significant design choices
3. **User Guides**: Step-by-step tutorials for common use cases
4. **Developer Guides**: Internal architecture and contribution guidelines
5. **Deployment Guides**: Production deployment and configuration

### Documentation Tools

- **API Docs**: Sphinx with autodoc for Python, JSDoc for JavaScript
- **Diagrams**: Mermaid for architecture, PlantUML for sequence diagrams
- **Tutorials**: Markdown with runnable code examples
- **Website**: GitHub Pages with automated deployment

### Writing Guidelines

- **Clear and concise**: Avoid jargon, explain technical terms
- **Examples included**: Show real-world usage patterns
- **Keep current**: Update docs with code changes
- **Multiple audiences**: Separate user and developer documentation

## Submission Guidelines

### Pull Request Template

When creating a pull request, use our template to ensure all information is provided:

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Security
- [ ] Security review completed (if applicable)
- [ ] No sensitive data exposed
- [ ] Cryptographic changes audited (if applicable)

## Documentation
- [ ] Documentation updated
- [ ] API documentation updated
- [ ] CHANGELOG.md updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No merge conflicts
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Peer Review**: At least one maintainer review required
3. **Security Review**: Additional review for security-sensitive changes
4. **Documentation Review**: Technical writing review for documentation changes
5. **Final Approval**: Maintainer approval and merge

### Merge Requirements

- All CI/CD checks pass
- At least one approved review from maintainer
- No merge conflicts
- Branch is up to date with target branch
- Security review completed (if applicable)

## Community

### Getting Help

- **Discord**: [Join our community](https://discord.gg/agent-mesh)
- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests
- **Stack Overflow**: Tag questions with `agent-mesh`

### Recognition

We value all contributions and recognize them through:

- **Contributors file**: All contributors listed in CONTRIBUTORS.md
- **Release notes**: Significant contributions highlighted
- **Community spotlight**: Monthly contributor recognition
- **Conference talks**: Opportunities to present your contributions

### Maintainer Responsibilities

Current maintainers commit to:

- **Timely responses**: 48-hour response time for issues and PRs
- **Code review**: Thorough review of all contributions
- **Community support**: Help new contributors get started
- **Documentation**: Keep project documentation current
- **Security**: Prompt response to security issues

---

## Questions?

If you have questions about contributing, please:

1. Check existing documentation and issues
2. Ask in our Discord community
3. Open a GitHub Discussion
4. Contact maintainers directly for sensitive issues

Thank you for contributing to the Agent Mesh Federated Runtime! 🚀