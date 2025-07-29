# GitHub Actions Workflow Setup

Due to GitHub permissions requirements, workflow files must be added manually. Here are the complete CI/CD pipeline configurations:

## Required Workflows

### 1. CI Pipeline (.github/workflows/ci.yml)

```yaml
name: Continuous Integration

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        
    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
      
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src/agent_mesh --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
        
  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test
    
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
        
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
        
    - name: Run Bandit security scan
      run: |
        bandit -r src/ -f json -o bandit-report.json
        
    - name: Run Safety dependency scan
      run: |
        safety check --json --output safety-report.json
        
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
          
  build:
    name: Build Package
    runs-on: ubuntu-latest
    needs: [test, integration-test, security-scan]
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: |
        python -m build
        
    - name: Check package
      run: |
        twine check dist/*
        
    - name: Upload build artifacts
      uses: actions/upload-artifact@v4
      with:
        name: dist
        path: dist/
```

### 2. Security Scanning (.github/workflows/security.yml)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM UTC

jobs:
  dependency-scan:
    name: Dependency Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        
    - name: Install Safety
      run: pip install safety
      
    - name: Run Safety scan
      run: |
        safety check --json --output safety-report.json || true
        safety check --short-report
        
    - name: Upload Safety report
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: safety-report
        path: safety-report.json
        
  code-security-scan:
    name: Code Security Analysis
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        
    - name: Install Bandit
      run: pip install bandit[toml]
      
    - name: Run Bandit scan
      run: |
        bandit -r src/ -f json -o bandit-report.json || true
        bandit -r src/ -f txt
        
    - name: Upload Bandit report
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: bandit-report
        path: bandit-report.json
        
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Build Docker image
      run: |
        docker build -t agent-mesh:security-scan .
        
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'agent-mesh:security-scan'
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
        
  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Run TruffleHog
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
        extra_args: --debug --only-verified
```

### 3. Release Automation (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    branches:
      - main
  workflow_dispatch:

permissions:
  contents: write
  packages: write
  pull-requests: write

jobs:
  release:
    name: Release
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install python-semantic-release build twine
        
    - name: Run tests
      run: |
        pip install -r requirements-dev.txt
        pytest tests/unit/ -v
        
    - name: Python Semantic Release
      id: release
      uses: python-semantic-release/python-semantic-release@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Build package
      if: steps.release.outputs.released == 'true'
      run: |
        python -m build
        
    - name: Upload to PyPI
      if: steps.release.outputs.released == 'true'
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/*
        
    - name: Build and push Docker image
      if: steps.release.outputs.released == 'true'
      run: |
        echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
        docker build -t ghcr.io/${{ github.repository }}:latest .
        docker build -t ghcr.io/${{ github.repository }}:${{ steps.release.outputs.tag }} .
        docker push ghcr.io/${{ github.repository }}:latest
        docker push ghcr.io/${{ github.repository }}:${{ steps.release.outputs.tag }}
```

### 4. Performance Testing (.github/workflows/performance.yml)

```yaml
name: Performance Testing

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * 0'  # Weekly Sunday 4 AM UTC

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: 'pip'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark-results.json
        
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        
  load-test:
    name: Load Testing
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
        pip install locust
        
    - name: Start application
      run: |
        python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
        sleep 10
        
    - name: Run load tests
      run: |
        locust -f tests/load/locustfile.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000
        
    - name: Upload load test results
      uses: actions/upload-artifact@v4
      if: always()
      with:
        name: load-test-results
        path: |
          *.html
          *.csv
```

## Setup Instructions

1. **Create workflow files manually**: Copy each workflow configuration above into the corresponding file path
2. **Configure secrets**: Add required secrets in GitHub repository settings:
   - `PYPI_TOKEN`: For automated PyPI releases
   - `CODECOV_TOKEN`: For code coverage reporting (optional)
3. **Enable workflows**: Ensure Actions are enabled in repository settings
4. **Test workflows**: Create a test PR to verify all workflows execute correctly

## Workflow Features

- **Multi-Python version testing**: Ensures compatibility across Python 3.9-3.12
- **Security-first approach**: Comprehensive security scanning on every PR
- **Performance regression detection**: Automated benchmarking with alerts
- **Automated releases**: Semantic versioning with PyPI and Docker registry publishing
- **Parallel execution**: Optimized CI/CD pipeline for fast feedback

These workflows provide enterprise-grade CI/CD capabilities for the Agent Mesh Federated Runtime project.