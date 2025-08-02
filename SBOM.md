# Software Bill of Materials (SBOM) Documentation

This document describes the Software Bill of Materials (SBOM) generation and management for the Agent Mesh Federated Runtime project.

## Overview

An SBOM provides a complete inventory of all software components, dependencies, and their relationships within the project. This is essential for:

- **Security**: Identifying vulnerable components
- **Compliance**: Meeting regulatory requirements (NIST, EU Cyber Resilience Act)
- **License Management**: Tracking license obligations
- **Supply Chain Security**: Understanding dependency risks

## SBOM Generation

### Automated Generation

SBOMs are automatically generated during:
- CI/CD builds
- Security scans
- Release processes
- Container image builds

### Tools Used

- **Syft**: Primary SBOM generation tool
- **CycloneDX**: Industry-standard SBOM format
- **SPDX**: Alternative SBOM format support
- **Grype**: Vulnerability scanning with SBOM integration

### Generation Commands

```bash
# Generate SBOM for the entire project
syft . -o spdx-json > sbom.spdx.json
syft . -o cyclonedx-json > sbom.cyclonedx.json

# Generate SBOM for container image
syft agent-mesh:latest -o spdx-json > container-sbom.spdx.json

# Generate SBOM for Python dependencies
syft python:requirements.txt -o spdx-json > python-deps-sbom.spdx.json
```

## SBOM Contents

### Components Tracked

1. **Source Code Components**
   - Main application code
   - Internal libraries and modules
   - Configuration files

2. **Dependencies**
   - Python packages (pip/pypi)
   - Node.js packages (npm)
   - System packages (apt/yum)
   - Container base images

3. **Build Tools**
   - Compilers and interpreters
   - Build systems and tools
   - Test frameworks

4. **Runtime Components**
   - Container runtime components
   - System libraries
   - Network protocols

### Metadata Included

- Component name and version
- Supplier/vendor information
- License information (SPDX IDs)
- Copyright notices
- Package URLs (PURL)
- File hashes and checksums
- Vulnerability identifiers (CVE)
- Dependency relationships

## SBOM Formats

### SPDX (Software Package Data Exchange)

Primary format for compliance and legal teams:

```json
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "agent-mesh-federated-runtime",
  "documentNamespace": "https://terragon.ai/sbom/agent-mesh-v1.0.0",
  "creationInfo": {
    "created": "2024-01-15T10:30:00Z",
    "creators": ["Tool: syft"]
  },
  "packages": [...]
}
```

### CycloneDX

Primary format for security and DevOps teams:

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.4",
  "serialNumber": "urn:uuid:12345678-1234-5678-9abc-123456789012",
  "version": 1,
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "tools": [{"name": "syft", "version": "0.98.0"}],
    "component": {
      "type": "application",
      "name": "agent-mesh-federated-runtime",
      "version": "1.0.0"
    }
  },
  "components": [...]
}
```

## Vulnerability Management

### SBOM-Based Scanning

```bash
# Scan SBOM for vulnerabilities
grype sbom:sbom.spdx.json -o json > vulnerability-report.json

# Continuous monitoring
grype sbom:sbom.spdx.json --fail-on high
```

### Integration with Security Tools

- **GitHub Security Advisories**: Automatic CVE matching
- **NIST NVD**: Vulnerability database integration
- **OSV**: Open Source Vulnerabilities database
- **Snyk**: Commercial vulnerability intelligence

## Compliance Integration

### Regulatory Requirements

1. **NIST SSDF (Secure Software Development Framework)**
   - Practice PO.3.2: Archive and protect each software release
   - Practice PO.5.1: Gather and safeguard evidence of compliance

2. **EU Cyber Resilience Act**
   - Article 13: CE marking requirements
   - Annex II: Essential cybersecurity requirements

3. **Executive Order 14028**
   - Section 4(e): Software bill of materials requirements

### Compliance Validation

```bash
# Validate SBOM completeness
scripts/validate-sbom.py --sbom sbom.spdx.json --standard nist-ssdf

# Check license compliance
scripts/check-licenses.py --sbom sbom.spdx.json --policy license-policy.yml
```

## License Management

### License Detection

- Automatic license detection from package metadata
- File-level license scanning
- License compatibility analysis
- Policy violation detection

### Supported License Formats

- SPDX License Identifiers
- Custom license texts
- Dual licensing support
- License exceptions

### License Policy

```yaml
# license-policy.yml
allowed_licenses:
  - MIT
  - Apache-2.0
  - BSD-3-Clause
  - BSD-2-Clause
  - ISC

restricted_licenses:
  - GPL-3.0
  - AGPL-3.0
  - LGPL-3.0

forbidden_licenses:
  - proprietary
  - unlicense
```

## SBOM Storage and Distribution

### Storage Locations

- **Source Repository**: Version-controlled SBOMs
- **Container Registry**: OCI-compliant SBOM attachments
- **Release Artifacts**: Signed SBOM files
- **Security Database**: Centralized SBOM storage

### Distribution Methods

1. **OCI Artifacts**
   ```bash
   # Attach SBOM to container image
   cosign attach sbom --sbom sbom.spdx.json agent-mesh:latest
   ```

2. **Release Assets**
   ```bash
   # Include SBOM in GitHub releases
   gh release upload v1.0.0 sbom.spdx.json sbom.cyclonedx.json
   ```

3. **API Endpoints**
   ```bash
   # Serve SBOM via API
   curl -H "Accept: application/spdx+json" \
        https://api.terragon.ai/sbom/agent-mesh/v1.0.0
   ```

## SBOM Verification

### Digital Signatures

```bash
# Sign SBOM with Sigstore
cosign sign-blob --bundle sbom.bundle sbom.spdx.json

# Verify SBOM signature  
cosign verify-blob --bundle sbom.bundle sbom.spdx.json
```

### Integrity Checking

```bash
# Generate SBOM checksums
sha256sum sbom.spdx.json > sbom.spdx.json.sha256
sha512sum sbom.spdx.json > sbom.spdx.json.sha512

# Verify integrity
sha256sum -c sbom.spdx.json.sha256
```

## Automation and CI/CD

### GitHub Actions Integration

```yaml
# .github/workflows/sbom.yml
- name: Generate SBOM
  run: |
    syft . -o spdx-json > sbom.spdx.json
    syft . -o cyclonedx-json > sbom.cyclonedx.json
    
- name: Scan for vulnerabilities
  run: grype sbom:sbom.spdx.json

- name: Sign SBOM
  run: cosign sign-blob --bundle sbom.bundle sbom.spdx.json
```

### Container Image Integration

```dockerfile
# Multi-stage build with SBOM generation
FROM scratch AS sbom
COPY sbom.spdx.json /sbom.spdx.json

FROM python:3.11-slim AS final
COPY --from=sbom /sbom.spdx.json /usr/share/sbom/
```

## Monitoring and Updates

### Continuous Monitoring

- Daily vulnerability scans
- Weekly dependency updates
- Monthly compliance audits
- Quarterly SBOM reviews

### Update Triggers

- New dependency versions
- Security advisories
- License changes
- Build environment updates

## Tools and Scripts

### SBOM Generation Scripts

```bash
# scripts/generate-sbom.sh
#!/bin/bash
set -euo pipefail

echo "Generating SBOM for Agent Mesh Federated Runtime..."

# Generate comprehensive SBOM
syft . -o spdx-json > sbom.spdx.json
syft . -o cyclonedx-json > sbom.cyclonedx.json

# Generate language-specific SBOMs
syft python:requirements.txt -o spdx-json > python-sbom.spdx.json
syft javascript:package.json -o spdx-json > js-sbom.spdx.json

# Validate SBOM format
python scripts/validate-sbom.py --sbom sbom.spdx.json

echo "SBOM generation completed successfully"
```

### Validation Scripts

```python
# scripts/validate-sbom.py
import json
import sys
from pathlib import Path

def validate_spdx_sbom(sbom_path):
    """Validate SPDX SBOM format and completeness."""
    with open(sbom_path) as f:
        sbom = json.load(f)
    
    required_fields = [
        'spdxVersion', 'dataLicense', 'SPDXID', 
        'name', 'documentNamespace', 'creationInfo'
    ]
    
    for field in required_fields:
        if field not in sbom:
            print(f"ERROR: Missing required field: {field}")
            return False
    
    if not sbom.get('packages'):
        print("ERROR: No packages found in SBOM")
        return False
    
    print(f"SBOM validation successful: {len(sbom['packages'])} packages")
    return True

if __name__ == "__main__":
    if validate_spdx_sbom(sys.argv[1]):
        sys.exit(0)
    else:
        sys.exit(1)
```

## Best Practices

1. **Generate Early and Often**
   - Include SBOM generation in CI/CD pipeline
   - Update SBOMs with every build
   - Version control SBOM files

2. **Comprehensive Coverage**
   - Include all dependencies (direct and transitive)
   - Cover all deployment targets
   - Document build-time and runtime components

3. **Security Integration**
   - Scan SBOMs for vulnerabilities
   - Monitor for new security advisories
   - Automate patching workflows

4. **Compliance Alignment**
   - Follow industry standards (SPDX, CycloneDX)
   - Document compliance mapping
   - Regular audit and review

## References

- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [NIST SSDF Guidelines](https://csrc.nist.gov/Projects/ssdf)
- [CISA SBOM Resources](https://www.cisa.gov/sbom)
- [Syft Documentation](https://github.com/anchore/syft)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)