# Contributing to Agent Mesh Federated Runtime

Thank you for your interest in contributing to the Agent Mesh Federated Runtime! This document outlines the development workflow and contribution guidelines.

## 🚀 Quick Start for Contributors

### 1. Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-org/agent-mesh-federated-runtime
cd agent-mesh-federated-runtime

# Use devcontainer for consistent environment
code .  # Open in VS Code and select "Reopen in Container"

# Or set up locally
pip install -e ".[dev]"
pre-commit install
```

### 2. Development Workflow

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes** following our coding standards
3. **Test your changes**: `make test`
4. **Lint and format**: `make lint`
5. **Commit with conventional format**: `git commit -m "feat: add new consensus algorithm"`
6. **Push and create PR**: `git push origin feature/your-feature-name`

## 📋 Contribution Areas

We welcome contributions in these priority areas:

### High Priority
- **Consensus Algorithms**: Implement new BFT consensus protocols
- **Network Protocols**: Add support for new P2P transport layers
- **Federated Learning**: Contribute aggregation strategies and privacy techniques
- **Edge Computing**: Optimize for resource-constrained environments

### Medium Priority
- **Security**: Enhance cryptographic protocols and access controls
- **Monitoring**: Improve observability and metrics collection
- **Documentation**: Expand guides, tutorials, and API documentation
- **Testing**: Add integration tests and performance benchmarks

## 🛠️ Development Standards

### Code Quality

- **Python Style**: Follow PEP 8, use Black formatter
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public APIs
- **Test Coverage**: Maintain >90% code coverage
- **Security**: No hardcoded secrets, follow security best practices

### Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
- `feat(consensus): add Tendermint consensus implementation`
- `fix(network): resolve libp2p connection timeout issue`
- `docs(api): update federated learning examples`

### Branch Naming

- Feature: `feature/description-of-feature`
- Bug fix: `fix/description-of-bug`
- Documentation: `docs/description-of-docs`
- Refactor: `refactor/description-of-refactor`

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Component integration tests
├── e2e/           # End-to-end system tests
├── performance/   # Performance and load tests
└── fixtures/      # Test data and mocks
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
pytest tests/unit/

# Integration tests (requires Docker)
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark-only

# With coverage
pytest --cov=agent_mesh tests/
```

### Writing Tests

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test full system workflows
- **Property Tests**: Use Hypothesis for property-based testing

```python
# Example unit test
def test_consensus_proposal_validation():
    consensus = ConsensusEngine(algorithm="pbft")
    proposal = Proposal(value="test", proposer="node1")
    
    assert consensus.validate_proposal(proposal) is True

# Example integration test
@pytest.mark.asyncio
async def test_mesh_node_communication():
    node1 = MeshNode(node_id="node1")
    node2 = MeshNode(node_id="node2")
    
    await node1.start()
    await node2.start()
    await node2.connect_to_peer(node1.peer_id)
    
    message = {"type": "ping", "data": "hello"}
    await node1.send_to_peer(node2.peer_id, message)
    
    received = await node2.receive_message(timeout=5)
    assert received["data"] == "hello"
```

## 📚 Documentation Guidelines

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials in `docs/guides/`
3. **Architecture Docs**: System design in `docs/architecture/`
4. **ADRs**: Architecture decisions in `docs/adr/`

### Writing Documentation

- **Be Clear**: Write for users with varying expertise levels
- **Include Examples**: Provide working code samples
- **Update Synchronously**: Keep docs in sync with code changes
- **Use Diagrams**: Include Mermaid diagrams for complex flows

```python
def create_mesh_node(node_id: str, role: str = "auto") -> MeshNode:
    """
    Create a new mesh network node.
    
    Args:
        node_id: Unique identifier for the node
        role: Node role - 'auto', 'trainer', 'aggregator', or 'validator'
        
    Returns:
        Configured MeshNode instance ready to join the network
        
    Example:
        >>> node = create_mesh_node("node-001", role="trainer")
        >>> await node.start()
        >>> await node.join_network(["peer1", "peer2"])
    """
```

## 🔄 Code Review Process

### Before Submitting PR

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No sensitive information in code

### PR Requirements

- **Descriptive Title**: Clear, concise description
- **Detailed Description**: Explain what, why, and how
- **Testing Notes**: How to test the changes
- **Breaking Changes**: List any breaking changes
- **Screenshots**: For UI changes

### Review Criteria

- **Functionality**: Does it work as intended?
- **Performance**: Does it impact system performance?
- **Security**: Are there security implications?
- **Maintainability**: Is the code readable and maintainable?
- **Testing**: Are changes adequately tested?

## 🐛 Bug Reports

Use the issue template and include:

- **Environment**: OS, Python version, package versions
- **Reproduction Steps**: Minimal steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Logs**: Relevant error messages or logs

## 💡 Feature Requests

For new features:

- **Use Case**: Describe the problem you're solving
- **Proposed Solution**: Your suggested approach
- **Alternatives**: Other solutions you considered
- **Implementation**: Willing to implement?

## 🏗️ Architecture Contributions

For significant architectural changes:

1. **Create an ADR**: Document the decision in `docs/adr/`
2. **Discuss in Issues**: Get community feedback first
3. **Prototype**: Create a proof-of-concept
4. **Iterate**: Refine based on feedback

## 🔐 Security Contributions

For security-related contributions:

- **Responsible Disclosure**: Report vulnerabilities privately first
- **Security Review**: All crypto/security code requires extra review
- **Documentation**: Update security documentation
- **Testing**: Include security-focused tests

## 📞 Getting Help

- **Discord**: [Join our community](https://discord.gg/your-org)
- **Discussions**: Use GitHub Discussions for questions
- **Email**: mesh@your-org.com for private matters
- **Office Hours**: Weekly maintainer office hours (see calendar)

## 🎯 Contribution Recognition

We recognize contributors through:

- **Contributors List**: Auto-updated in README
- **Release Notes**: Highlight major contributions
- **Swag**: Stickers and t-shirts for significant contributors
- **Conference Opportunities**: Speaking opportunities at events

## 📄 Legal

By contributing, you agree that your contributions will be licensed under the Apache License 2.0. You also certify that you have the right to make the contribution under this license.

See the [Developer Certificate of Origin](https://developercertificate.org/) for details.

---

Thank you for contributing to the future of decentralized AI! 🚀
