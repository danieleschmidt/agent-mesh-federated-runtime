# 🤝 Contributing to Agent Mesh Federated Runtime

Thank you for your interest in contributing to the Agent Mesh Federated Runtime! This project aims to build the most robust, scalable, and secure decentralized platform for federated learning and multi-agent systems.

## 🎯 Contributing Overview

We welcome contributions from everyone, whether you're:
- 🐛 Reporting bugs
- 💡 Suggesting features
- 📝 Improving documentation
- 💻 Contributing code
- 🧪 Writing tests
- 🔒 Enhancing security
- 🌍 Translating content

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to `conduct@terragon.ai`.

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Docker** and Docker Compose
- **Git** for version control
- **Basic knowledge** of:
  - Python async programming
  - P2P networking concepts
  - Federated learning fundamentals

### Quick Setup

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/agent-mesh-federated-runtime.git
   cd agent-mesh-federated-runtime
   ```
3. **Set up development environment**:
   ```bash
   # Using VS Code Dev Containers (recommended)
   code .  # Open in VS Code and use "Reopen in Container"
   
   # Or setup locally
   make setup-dev
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-awesome-feature
   ```

## 🛠️ Development Setup

### Using Dev Containers (Recommended)

The easiest way to get started is using VS Code Dev Containers:

1. Install VS Code and the Remote-Containers extension
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted
4. Everything will be set up automatically!

### Manual Setup

If you prefer to set up the environment manually:

```bash
# Install Python dependencies
pip install -e ".[dev]"

# Install Node.js dependencies
npm install

# Install pre-commit hooks
pre-commit install

# Setup environment variables
cp .env.example .env

# Run tests to verify setup
make test
```

### Available Commands

```bash
# Development
make dev          # Start development server
make test         # Run all tests
make lint         # Run linting
make format       # Format code
make docs         # Build documentation

# Docker
make docker-build # Build Docker image
make docker-run   # Run in Docker
make docker-test  # Run tests in Docker

# Cleanup
make clean        # Clean build artifacts
make clean-all    # Deep clean including caches
```

## 📋 Contributing Guidelines

### 🐛 Reporting Bugs

Before reporting a bug:
1. Check if it's already reported in [Issues](https://github.com/your-org/agent-mesh-federated-runtime/issues)
2. Test with the latest version
3. Provide minimal reproduction steps

**Bug Report Template:**
```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Bug appears

**Expected Behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.0]
- Agent Mesh version: [e.g., 1.0.0]

**Additional Context**
Screenshots, logs, etc.
```

### 💡 Feature Requests

We love new ideas! Before submitting:
1. Check existing [feature requests](https://github.com/your-org/agent-mesh-federated-runtime/issues?q=is%3Aissue+label%3Aenhancement)
2. Consider if it fits the project scope
3. Think about implementation complexity

**Feature Request Template:**
```markdown
**Feature Description**
Clear description of the proposed feature

**Problem/Use Case**
What problem does this solve?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other solutions you've considered

**Additional Context**
Mockups, examples, etc.
```

### 💻 Code Contributions

#### Code Style

We follow strict code quality standards:

- **Python**: PEP 8 with Black formatting (88 char line length)
- **JavaScript**: Prettier with ESLint
- **Documentation**: Markdown with consistent formatting
- **Commits**: Conventional Commits format

#### Coding Standards

```python
# Good: Clear, documented, type-hinted
async def aggregate_model_updates(
    updates: List[ModelUpdate],
    strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
    timeout: float = 30.0,
) -> AggregatedModel:
    """Aggregate model updates from multiple nodes.
    
    Args:
        updates: List of model updates from participating nodes
        strategy: Aggregation strategy to use
        timeout: Maximum time to wait for aggregation
        
    Returns:
        Aggregated model ready for distribution
        
    Raises:
        AggregationError: If aggregation fails
        TimeoutError: If aggregation times out
    """
    logger.info(f"Aggregating {len(updates)} model updates")
    # Implementation here...
```

#### Testing Requirements

All code contributions must include tests:

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows (for major features)
- **Performance Tests**: For performance-critical code

```python
# Example test structure
import pytest
from agent_mesh.consensus import PBFTConsensus

class TestPBFTConsensus:
    @pytest.fixture
    async def consensus_engine(self):
        return PBFTConsensus(fault_tolerance=0.33)
    
    async def test_consensus_proposal(self, consensus_engine):
        """Test basic consensus proposal workflow."""
        proposal = create_test_proposal()
        result = await consensus_engine.propose(proposal)
        assert result.status == ProposalStatus.ACCEPTED
    
    async def test_byzantine_fault_tolerance(self, consensus_engine):
        """Test behavior with Byzantine nodes."""
        # Test implementation...
```

#### Security Considerations

Security is paramount. Please:

- **Never** commit secrets, keys, or credentials
- **Always** validate user inputs
- **Use** established cryptographic libraries
- **Follow** secure coding practices
- **Consider** privacy implications of changes

## 🔄 Pull Request Process

### Before Submitting

1. **Sync** with the latest main branch
2. **Run** all tests and ensure they pass
3. **Format** code with our standards
4. **Update** documentation if needed
5. **Add** tests for new functionality

### Pull Request Template

```markdown
## 📋 Description
Brief description of changes

## 🎯 Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix/feature causing existing functionality to change)
- [ ] Documentation update

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No security issues introduced

## 🔗 Related Issues
Fixes #(issue_number)
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer reviews
3. **Testing**: All tests must pass
4. **Security Review**: For security-sensitive changes
5. **Documentation**: Ensure docs are updated
6. **Approval**: Required approvals from maintainers

### After Merge

- **Monitor**: Watch for any issues post-merge
- **Celebrate**: Your contribution is now part of the project! 🎉
- **Stay Engaged**: Consider becoming a regular contributor

## 📝 Issue Guidelines

### Issue Types

We use labels to categorize issues:

- 🐛 **bug**: Something isn't working
- ✨ **enhancement**: New feature or request
- 📝 **documentation**: Documentation improvements
- 🚀 **performance**: Performance improvements
- 🔒 **security**: Security-related issues
- 🧪 **testing**: Testing improvements
- 🎨 **refactoring**: Code structure improvements

### Issue Lifecycle

1. **Triage**: Team reviews and labels new issues
2. **Assignment**: Issues assigned to contributors
3. **Development**: Work begins on the issue
4. **Review**: Pull request reviewed and merged
5. **Testing**: Changes tested in development
6. **Release**: Changes included in next release

### Good First Issues

New to the project? Look for issues labeled:
- `good first issue`: Perfect for newcomers
- `help wanted`: Community help needed
- `documentation`: Usually easier to start with

## 🌟 Recognition

We value all contributions and recognize them through:

- **Contributors Page**: Listed on our website
- **Release Notes**: Mentioned in release announcements
- **Annual Report**: Highlighted in yearly summaries
- **Special Recognition**: For significant contributions

## 🎓 Learning Resources

### Project-Specific Resources
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [API Documentation](docs/api/)
- [Development Guide](docs/DEVELOPMENT.md)
- [Security Guidelines](docs/security/)

### General Learning
- [Federated Learning Overview](https://federated.withgoogle.com/)
- [P2P Networking Concepts](https://docs.libp2p.io/concepts/)
- [Byzantine Fault Tolerance](https://pmg.csail.mit.edu/papers/osdi99.pdf)
- [Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)

## 💬 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time chat with the community
- **Email**: `community@terragon.ai` for private matters

### Community Events

- **Monthly Meetups**: Virtual community gatherings
- **Hackathons**: Quarterly coding events
- **Conferences**: Speaking at relevant conferences
- **Workshops**: Educational sessions for contributors

### Mentorship Program

We offer mentorship for new contributors:
- **Pair Programming**: Work directly with maintainers
- **Code Reviews**: Learning-focused review sessions
- **Project Guidance**: Help choosing suitable issues
- **Career Development**: Advice on open source careers

## 🏆 Contributor Levels

As you contribute more, you can advance through levels:

### 🌱 Contributor
- Submitted at least one merged PR
- Familiar with project basics
- Follows contribution guidelines

### 🌟 Regular Contributor
- Multiple significant contributions
- Helps with code reviews
- Mentors new contributors

### 🚀 Core Contributor
- Deep project knowledge
- Leads feature development
- Helps with project direction

### 👑 Maintainer
- Commit access to repository
- Releases management
- Community leadership

## 📞 Getting Help

Stuck? Need help? Reach out:

1. **Documentation**: Check our comprehensive docs
2. **GitHub Discussions**: Ask the community
3. **Discord**: Real-time help from contributors
4. **Office Hours**: Weekly Q&A sessions with maintainers

## 🙏 Thank You

Your contributions make this project better for everyone. Whether you're fixing a typo, adding a feature, or helping others, you're part of building something amazing.

**Happy Contributing!** 🚀

---

*This contributing guide is a living document. We welcome suggestions for improvements!*
