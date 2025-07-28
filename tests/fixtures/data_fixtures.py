"""Data fixtures for federated learning tests."""

import numpy as np
import pytest
from typing import Dict, List, Tuple, Any


@pytest.fixture
def sample_dataset():
    """Generate a simple synthetic dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)
    return X, y


@pytest.fixture
def mnist_data_partitions(request):
    """Create MNIST data partitions for federated learning."""
    num_clients = getattr(request, 'param', 5)
    
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        
        # Use a small subset for testing
        X = np.random.randn(1000, 784)  # Mock MNIST-like data
        y = np.random.randint(0, 10, 1000)
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Partition data among clients
        client_data = {}
        samples_per_client = len(X_train) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train)
            
            client_data[f"client_{i}"] = {
                "X_train": X_train[start_idx:end_idx],
                "y_train": y_train[start_idx:end_idx],
                "X_test": X_test,
                "y_test": y_test,
            }
        
        return client_data
    
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture
def cifar10_data_partitions(request):
    """Create CIFAR-10 data partitions for federated learning."""
    num_clients = getattr(request, 'param', 5)
    
    # Mock CIFAR-10 data
    X = np.random.randn(1000, 32, 32, 3)
    y = np.random.randint(0, 10, 1000)
    
    client_data = {}
    samples_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        
        client_data[f"client_{i}"] = {
            "X": X[start_idx:end_idx],
            "y": y[start_idx:end_idx],
        }
    
    return client_data


@pytest.fixture
def synthetic_federated_data():
    """Generate synthetic federated learning data with varying distributions."""
    np.random.seed(42)
    
    num_clients = 5
    num_features = 20
    num_classes = 3
    
    client_data = {}
    
    for i in range(num_clients):
        # Create different data distributions for each client
        samples_per_client = np.random.randint(100, 500)
        
        # Introduce distribution skew
        class_probs = np.random.dirichlet(np.ones(num_classes) * (i + 1))
        
        X = np.random.randn(samples_per_client, num_features)
        y = np.random.choice(num_classes, samples_per_client, p=class_probs)
        
        # Add client-specific bias to features
        X += np.random.randn(num_features) * (i + 1) * 0.1
        
        client_data[f"client_{i}"] = {
            "X": X,
            "y": y,
            "class_distribution": class_probs,
            "num_samples": samples_per_client,
        }
    
    return client_data


@pytest.fixture
def non_iid_data_splits():
    """Create non-IID data splits for testing federated learning."""
    np.random.seed(42)
    
    # Generate base dataset
    total_samples = 10000
    num_features = 50
    num_classes = 10
    num_clients = 8
    
    X = np.random.randn(total_samples, num_features)
    y = np.random.randint(0, num_classes, total_samples)
    
    # Create different non-IID scenarios
    scenarios = {
        "label_skew": _create_label_skew_split(X, y, num_clients),
        "feature_skew": _create_feature_skew_split(X, y, num_clients),
        "quantity_skew": _create_quantity_skew_split(X, y, num_clients),
        "temporal_skew": _create_temporal_skew_split(X, y, num_clients),
    }
    
    return scenarios


def _create_label_skew_split(X: np.ndarray, y: np.ndarray, num_clients: int) -> Dict[str, Any]:
    """Create label distribution skew among clients."""
    client_data = {}
    num_classes = len(np.unique(y))
    
    # Each client gets data from only 2-3 classes
    for i in range(num_clients):
        # Select 2-3 classes for this client
        num_client_classes = np.random.randint(2, 4)
        client_classes = np.random.choice(num_classes, num_client_classes, replace=False)
        
        # Get samples from selected classes
        mask = np.isin(y, client_classes)
        client_indices = np.where(mask)[0]
        
        # Randomly sample from available indices
        samples_per_client = min(len(client_indices), np.random.randint(100, 500))
        selected_indices = np.random.choice(client_indices, samples_per_client, replace=False)
        
        client_data[f"client_{i}"] = {
            "X": X[selected_indices],
            "y": y[selected_indices],
            "classes": client_classes.tolist(),
            "num_samples": samples_per_client,
        }
    
    return client_data


def _create_feature_skew_split(X: np.ndarray, y: np.ndarray, num_clients: int) -> Dict[str, Any]:
    """Create feature distribution skew among clients."""
    client_data = {}
    samples_per_client = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        
        # Add feature-specific bias
        X_client = X[start_idx:end_idx].copy()
        feature_bias = np.random.randn(X.shape[1]) * (i + 1) * 0.2
        X_client += feature_bias
        
        client_data[f"client_{i}"] = {
            "X": X_client,
            "y": y[start_idx:end_idx],
            "feature_bias": feature_bias,
            "num_samples": end_idx - start_idx,
        }
    
    return client_data


def _create_quantity_skew_split(X: np.ndarray, y: np.ndarray, num_clients: int) -> Dict[str, Any]:
    """Create quantity skew among clients."""
    client_data = {}
    
    # Generate different sample sizes following power law distribution
    total_samples = len(X)
    sample_sizes = np.random.power(0.5, num_clients)  # Power law distribution
    sample_sizes = (sample_sizes / sample_sizes.sum() * total_samples).astype(int)
    sample_sizes[-1] = total_samples - sample_sizes[:-1].sum()  # Ensure sum is correct
    
    start_idx = 0
    for i, size in enumerate(sample_sizes):
        end_idx = start_idx + size
        
        client_data[f"client_{i}"] = {
            "X": X[start_idx:end_idx],
            "y": y[start_idx:end_idx],
            "num_samples": size,
            "sample_ratio": size / total_samples,
        }
        
        start_idx = end_idx
    
    return client_data


def _create_temporal_skew_split(X: np.ndarray, y: np.ndarray, num_clients: int) -> Dict[str, Any]:
    """Create temporal skew among clients."""
    client_data = {}
    
    # Simulate temporal concept drift
    for i in range(num_clients):
        # Each client represents a different time period
        time_period = i
        drift_factor = time_period * 0.1
        
        # Sample data for this time period
        samples_per_client = len(X) // num_clients
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X)
        
        X_client = X[start_idx:end_idx].copy()
        y_client = y[start_idx:end_idx].copy()
        
        # Apply concept drift
        X_client += np.random.randn(*X_client.shape) * drift_factor
        
        # Label drift (some classes become more/less frequent over time)
        if time_period > 0:
            label_drift_prob = 0.1 * time_period
            drift_mask = np.random.random(len(y_client)) < label_drift_prob
            y_client[drift_mask] = np.random.randint(0, len(np.unique(y)), drift_mask.sum())
        
        client_data[f"client_{i}"] = {
            "X": X_client,
            "y": y_client,
            "time_period": time_period,
            "drift_factor": drift_factor,
            "num_samples": end_idx - start_idx,
        }
    
    return client_data


@pytest.fixture
def federated_learning_metrics():
    """Sample federated learning metrics for testing."""
    return {
        "round_metrics": [
            {
                "round": 1,
                "global_loss": 2.3,
                "global_accuracy": 0.12,
                "client_metrics": {
                    "client_0": {"loss": 2.4, "accuracy": 0.10, "samples": 150},
                    "client_1": {"loss": 2.2, "accuracy": 0.14, "samples": 200},
                    "client_2": {"loss": 2.5, "accuracy": 0.11, "samples": 120},
                },
                "convergence_metrics": {
                    "loss_improvement": 0.0,
                    "accuracy_improvement": 0.0,
                    "variance": 0.15,
                }
            },
            {
                "round": 2,
                "global_loss": 1.8,
                "global_accuracy": 0.35,
                "client_metrics": {
                    "client_0": {"loss": 1.9, "accuracy": 0.32, "samples": 150},
                    "client_1": {"loss": 1.7, "accuracy": 0.38, "samples": 200},
                    "client_2": {"loss": 1.8, "accuracy": 0.35, "samples": 120},
                },
                "convergence_metrics": {
                    "loss_improvement": 0.5,
                    "accuracy_improvement": 0.23,
                    "variance": 0.08,
                }
            },
        ],
        "aggregation_stats": {
            "participation_rate": 0.85,
            "communication_rounds": 2,
            "total_samples": 470,
            "average_client_contribution": 156.67,
        }
    }


@pytest.fixture
def privacy_noise_profiles():
    """Different privacy noise profiles for testing differential privacy."""
    return {
        "low_privacy": {
            "epsilon": 10.0,
            "delta": 1e-3,
            "mechanism": "gaussian",
            "sensitivity": 1.0,
        },
        "medium_privacy": {
            "epsilon": 1.0,
            "delta": 1e-5,
            "mechanism": "gaussian",
            "sensitivity": 1.0,
        },
        "high_privacy": {
            "epsilon": 0.1,
            "delta": 1e-7,
            "mechanism": "laplace",
            "sensitivity": 1.0,
        },
        "compositions": [
            {"epsilon": 1.0, "delta": 1e-5, "rounds": 10},
            {"epsilon": 0.5, "delta": 1e-6, "rounds": 20},
            {"epsilon": 0.1, "delta": 1e-7, "rounds": 50},
        ]
    }