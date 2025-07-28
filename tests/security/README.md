# Security Tests

This directory contains security-focused tests to validate the security properties of the Agent Mesh system.

## ⚠️ Important Security Notice

**AUTHORIZED TESTING ONLY**: The penetration tests in this directory should only be run in authorized test environments. These tests may include simulated attacks and should never be run against production systems or systems you don't own.

## Test Categories

### Cryptography Tests (`crypto/`)
- Key generation and management
- Encryption/decryption operations
- Digital signatures and verification
- Hash function security
- Random number generation quality

### Network Security Tests (`network/`)
- TLS/SSL implementation security
- P2P protocol security
- Message authentication
- Network-level attack resistance
- Communication channel security

### Consensus Security Tests (`consensus/`)
- Byzantine fault tolerance
- Double-spending prevention
- Fork resistance
- Leader election security
- Vote verification

### Privacy Tests (`privacy/`)
- Differential privacy guarantees
- Secure aggregation protocols
- Data leakage prevention
- Anonymization effectiveness
- Privacy budget tracking

### Authentication Tests (`auth/`)
- Node identity verification
- Access control enforcement
- Certificate validation
- Multi-factor authentication
- Session management

### Penetration Tests (`pentest/`)
- Simulated attack scenarios
- Vulnerability assessment
- Security boundary testing
- Exploit attempt simulation
- Defense mechanism validation

## Running Security Tests

### Prerequisites
```bash
# Install security testing dependencies
pip install -e ".[dev]"

# Ensure you have authorization to run these tests
echo "I HAVE AUTHORIZATION TO RUN SECURITY TESTS" > .security_test_authorization
```

### Basic Security Tests
```bash
# Run all security tests
pytest tests/security/ -v

# Run specific test categories
pytest tests/security/crypto/ -v
pytest tests/security/network/ -v
pytest tests/security/privacy/ -v
```

### Penetration Tests (Requires Authorization)
```bash
# Run penetration tests (AUTHORIZED ENVIRONMENTS ONLY)
make pentest

# Or run specific pentest categories
pytest tests/security/pentest/ -v --tb=short
```

## Test Markers

- `@pytest.mark.security` - General security tests
- `@pytest.mark.crypto` - Cryptography tests
- `@pytest.mark.network_security` - Network security tests
- `@pytest.mark.privacy` - Privacy-related tests
- `@pytest.mark.pentest` - Penetration tests (requires authorization)
- `@pytest.mark.slow` - Long-running security tests

## Security Test Guidelines

### Writing Security Tests
1. **Test Real Vulnerabilities**: Focus on actual security concerns
2. **Use Realistic Scenarios**: Simulate real-world attack conditions
3. **Validate Defenses**: Ensure security measures work as intended
4. **Document Assumptions**: Clearly state security assumptions
5. **Fail Securely**: Tests should fail if security is compromised

### Best Practices
- Never hardcode secrets in test code
- Use deterministic randomness for reproducible tests
- Clean up test artifacts that could contain sensitive data
- Run tests in isolated environments
- Review test code for security implications

## Security Metrics

Security tests track various metrics:

- **Cryptographic Strength**: Key entropy, algorithm security
- **Attack Resistance**: Success rate of simulated attacks
- **Privacy Guarantees**: Differential privacy budget consumption
- **Performance Impact**: Security overhead measurements
- **Compliance**: Adherence to security standards

## Reporting Security Issues

If security tests reveal vulnerabilities:

1. **DO NOT** create public GitHub issues
2. **DO** report to security@your-org.com
3. **DO** include test reproduction steps
4. **DO** suggest mitigation strategies
5. **DO** follow responsible disclosure practices

## Compliance Testing

Security tests also validate compliance with:

- **GDPR**: Data protection and privacy
- **HIPAA**: Healthcare data security (if applicable)
- **SOC 2**: Security operational controls
- **ISO 27001**: Information security management
- **NIST Cybersecurity Framework**: Security best practices

## Continuous Security Testing

Security tests are integrated into CI/CD:

```yaml
# Example CI security checks
- name: Run Security Tests
  run: pytest tests/security/ --ignore=tests/security/pentest/

- name: Security Scan
  run: |
    bandit -r src/
    safety check
    
- name: Dependency Audit
  run: npm audit --audit-level moderate
```

## Resources

- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Cryptography Best Practices](https://crypto.stackexchange.com/)
- [Python Security Guidelines](https://python-security.readthedocs.io/)
