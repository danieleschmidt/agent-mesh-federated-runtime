# Test Fixtures and Data

This directory contains test data, fixtures, and mock objects used across the test suite.

## Directory Structure

```
fixtures/
├── data/                    # Test data files
│   ├── models/             # Sample model files
│   ├── datasets/           # Test datasets
│   └── configs/            # Test configuration files
├── mocks/                  # Mock objects and responses
│   ├── network/            # Network-related mocks
│   ├── consensus/          # Consensus protocol mocks
│   └── federated/          # Federated learning mocks
└── factories/              # Test object factories
    ├── node_factory.py     # Create test nodes
    ├── task_factory.py     # Create test tasks
    └── model_factory.py    # Create test models
```

## Usage Guidelines

### Test Data Files
- Store static test data in `data/` subdirectories
- Use JSON format for configuration data
- Use binary formats for model weights and datasets
- Keep file sizes small (< 1MB) for fast test execution

### Mock Objects
- Place reusable mocks in `mocks/` subdirectories
- Use descriptive names for mock files
- Include docstrings explaining mock behavior
- Avoid overly complex mocks that hide bugs

### Test Factories
- Create factory functions for complex test objects
- Use randomization for non-critical properties
- Allow customization of important properties
- Follow the Factory pattern for consistency

## Examples

### Creating Test Nodes
```python
from tests.fixtures.factories.node_factory import create_test_node

# Create a basic test node
node = create_test_node()

# Create a node with specific configuration
node = create_test_node(
    node_id="test-node-001",
    role="trainer",
    port=14001
)
```

### Using Test Data
```python
import json
from pathlib import Path

# Load test configuration
config_path = Path(__file__).parent / "fixtures/data/configs/test_config.json"
with open(config_path) as f:
    config = json.load(f)
```

### Using Mocks
```python
from tests.fixtures.mocks.network.p2p_mock import MockP2PNetwork

# Use a mock P2P network
mock_network = MockP2PNetwork()
await mock_network.connect_peer("peer-001")
```
