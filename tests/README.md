# Testing Documentation

This directory contains the comprehensive test suite for the Agent Mesh Federated Runtime project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── coverage.ini             # Coverage configuration
├── e2e/                     # End-to-end tests
├── fixtures/                # Test data and fixtures
├── integration/             # Integration tests
├── performance/             # Performance and load tests
├── unit/                    # Unit tests
└── utils/                   # Test utilities and helpers
```

## Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation.

**Markers:** `@pytest.mark.unit`

**Coverage:** Individual functions, classes, and modules
- Component functionality
- Error handling
- Edge cases
- Input validation

### Integration Tests (`tests/integration/`)
Test component interactions and system integration.

**Markers:** `@pytest.mark.integration`

**Coverage:** Component interactions
- P2P networking
- Consensus protocols
- Message passing
- Database operations

### End-to-End Tests (`tests/e2e/`)
Test complete user workflows and system behavior.

**Markers:** `@pytest.mark.e2e`

**Coverage:** Full system workflows
- Network formation
- Federated learning rounds
- Node joining/leaving
- Fault recovery

### Performance Tests (`tests/performance/`)
Test system performance, scalability, and resource usage.

**Markers:** `@pytest.mark.performance`, `@pytest.mark.benchmark`

**Coverage:** Performance characteristics
- Throughput measurement
- Latency analysis
- Resource utilization
- Scalability limits

## Test Markers

The test suite uses pytest markers to categorize and filter tests:

### Functional Markers
- `unit` - Unit tests
- `integration` - Integration tests  
- `e2e` - End-to-end tests
- `performance` - Performance tests

### Speed Markers
- `fast` - Fast tests (< 1 second)
- `slow` - Slow tests (> 5 seconds)

### Resource Markers
- `network` - Requires network access
- `gpu` - Requires GPU
- `docker` - Requires Docker
- `kubernetes` - Requires Kubernetes

### Domain Markers
- `consensus` - Consensus algorithm tests
- `federated` - Federated learning tests
- `security` - Security tests
- `crypto` - Cryptographic tests
- `byzantine` - Byzantine fault tolerance tests

### Environment Markers
- `mock` - Uses extensive mocking
- `real` - Uses real network connections
- `distributed` - Distributed system tests

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m "unit or integration"

# Run tests by directory
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src/agent_mesh --cov-report=html

# Run specific test file
pytest tests/unit/test_example.py

# Run specific test function
pytest tests/unit/test_example.py::test_specific_function
```

### Performance Testing

```bash
# Run performance tests
pytest -m performance

# Run benchmarks
pytest -m benchmark --benchmark-only

# Run load tests
pytest tests/performance/test_load_scenarios.py -v

# Run scalability tests
pytest -m scalability -s
```

### Advanced Test Options

```bash
# Run tests in parallel
pytest -n auto

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Run failed tests from last run
pytest --lf

# Run only changed tests
pytest --testmon

# Generate JUnit XML report
pytest --junitxml=test-results.xml

# Run with specific log level
pytest --log-cli-level=DEBUG
```

### Docker-based Testing

```bash
# Run tests in Docker container
docker-compose -f docker-compose.test.yml up --build

# Run specific test category in Docker
docker-compose run test pytest -m integration
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Coverage settings
- Marker definitions
- Logging configuration
- Timeout settings

### Coverage Configuration (`tests/coverage.ini`)
- Source paths
- Exclusion patterns
- Report formats
- Coverage thresholds

### Environment Variables
Set these environment variables for testing:

```bash
export ENVIRONMENT=testing
export LOG_LEVEL=DEBUG
export TEST_TIMEOUT=300
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

## Writing Tests

### Test Naming Conventions

- Test files: `test_*.py` or `*_test.py`
- Test classes: `Test*` or `*Tests`
- Test functions: `test_*`

### Test Structure

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.unit
async def test_feature_functionality(fixture_name):
    """Test specific functionality with clear description."""
    # Arrange
    setup_test_data()
    
    # Act
    result = await function_under_test()
    
    # Assert
    assert result.is_expected()
    assert result.meets_requirements()
```

### Using Fixtures

```python
# Use existing fixtures
def test_with_config(test_config):
    assert test_config["environment"] == "testing"

# Create test-specific fixtures
@pytest.fixture
def custom_fixture():
    return {"custom": "data"}

# Parameterized tests
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    assert double(input) == expected
```

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functions."""
    result = await async_function()
    assert result is not None

# Use async fixtures  
@pytest.fixture
async def async_fixture():
    resource = await create_async_resource()
    yield resource
    await cleanup_resource(resource)
```

### Mocking Guidelines

```python
# Mock external dependencies
@patch('module.external_dependency')
def test_with_mock(mock_external):
    mock_external.return_value = "mocked_response"
    result = function_that_uses_external()
    assert result == "expected_result"

# Use AsyncMock for async functions
async def test_async_mock():
    mock_service = AsyncMock()
    mock_service.async_method.return_value = "result"
    
    result = await function_using_service(mock_service)
    assert result == "expected"
    mock_service.async_method.assert_called_once()
```

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)
- Event loop configuration
- Database connections
- Network configurations
- Mock objects

### Data Fixtures (`fixtures/`)
- Sample datasets for federated learning
- Network topologies
- Configuration templates
- Cryptographic test data

### Mock Helpers (`utils/mock_helpers.py`)
- Network simulation
- Byzantine node behavior
- Resource constraints
- Fault injection

## Continuous Integration

### GitHub Actions Integration
Tests run automatically on:
- Pull requests
- Main branch pushes
- Release tags

### Test Stages
1. **Fast Tests**: Unit tests and quick integration tests
2. **Integration Tests**: Component interaction tests
3. **E2E Tests**: Full system workflow tests
4. **Performance Tests**: Load and scalability tests

### Quality Gates
- Minimum 85% code coverage
- All tests must pass
- Performance regression detection
- Security vulnerability scanning

## Debugging Tests

### Local Debugging

```bash
# Run with debugger
pytest --pdb tests/unit/test_example.py::test_function

# Debug on failure
pytest --pdb-trace

# Capture output
pytest -s

# Verbose output with timings
pytest -v --durations=10
```

### Test Isolation

```bash
# Run single test in isolation
pytest tests/unit/test_example.py::test_function -v

# Run with fresh environment
pytest --forked tests/unit/test_example.py
```

### Log Analysis

```bash
# Show all logs
pytest --log-cli-level=DEBUG --log-cli-format='%(levelname)s:%(name)s:%(message)s'

# Capture logs to file
pytest --log-file=test.log --log-file-level=DEBUG
```

## Performance Testing

### Benchmark Tests
Use `pytest-benchmark` for performance measurements:

```python
def test_performance(benchmark):
    result = benchmark(function_to_measure)
    assert result is not None
```

### Load Testing
Simulate high-load scenarios:

```python
@pytest.mark.performance
async def test_high_load():
    # Create load scenario
    scenario = LoadTestScenario(requests_per_second=1000)
    results = await scenario.run()
    
    # Assert performance requirements
    assert results["success_rate"] > 0.95
    assert results["average_latency"] < 0.1
```

### Resource Testing
Test under resource constraints:

```python
def test_memory_limit():
    with memory_limit(100_000_000):  # 100MB limit
        result = memory_intensive_function()
        assert result is not None
```

## Best Practices

### Test Design
1. **Arrange, Act, Assert**: Structure tests clearly
2. **Single Responsibility**: One test per behavior
3. **Descriptive Names**: Clear test function names
4. **Independent Tests**: No test dependencies
5. **Fast Feedback**: Optimize test execution time

### Mock Usage
1. **Mock External Dependencies**: Don't test external systems
2. **Verify Interactions**: Assert mock calls when relevant
3. **Reset Mocks**: Clean state between tests
4. **Realistic Mocks**: Behavior should match real objects

### Data Management
1. **Deterministic Data**: Use fixed seeds for random data
2. **Minimal Data**: Use smallest data sets that test the behavior
3. **Clean Fixtures**: Isolated and reusable test data
4. **Parameterized Tests**: Test multiple scenarios efficiently

### Error Testing
1. **Happy Path**: Test normal operation
2. **Edge Cases**: Test boundary conditions
3. **Error Conditions**: Test exception handling
4. **Resource Exhaustion**: Test failure scenarios

## Troubleshooting

### Common Issues

#### Test Discovery Problems
```bash
# Check test discovery
pytest --collect-only

# Verify test file patterns
ls tests/**/*test*.py
```

#### Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Install package in development mode
pip install -e .
```

#### Async Test Issues
```bash
# Use asyncio mode
pytest --asyncio-mode=auto

# Check event loop policy
pytest -s tests/unit/test_async.py
```

#### Coverage Issues
```bash
# Debug coverage measurement
pytest --cov=src/agent_mesh --cov-report=term-missing

# Check coverage configuration
coverage config --show-config
```

### Getting Help

1. Check the [pytest documentation](https://docs.pytest.org/)
2. Review test examples in `tests/unit/test_example.py`
3. Examine fixture definitions in `conftest.py`
4. Ask in project discussions or issues

## Contributing Tests

### Adding New Tests
1. Follow naming conventions
2. Use appropriate markers
3. Add fixtures to `conftest.py` if reusable
4. Update this documentation if adding new test categories

### Test Review Checklist
- [ ] Tests have descriptive names
- [ ] Appropriate markers are used
- [ ] Mocks are used for external dependencies
- [ ] Tests are independent and idempotent
- [ ] Coverage requirements are met
- [ ] Performance tests have reasonable thresholds