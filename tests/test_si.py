"""Tests for SI (Synaptic Intelligence) implementation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptive_ml.training.si import SI, SIStats


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )
    for param in model.parameters():
        param.requires_grad = True
    return model


@pytest.fixture
def si(simple_model):
    """Create a SI instance for testing."""
    return SI(simple_model, lambda_si=100.0)


@pytest.fixture
def dataloader():
    """Create a simple dataloader for testing."""
    X = torch.randn(50, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


class TestSI:
    """Tests for SI class."""

    def test_init(self, si, simple_model):
        """Test initialization."""
        assert si.model == simple_model
        assert si.lambda_si == 100.0
        assert len(si.importance) > 0
        assert len(si.saved_params) == 0
        assert len(si.prev_grads) == 0

    def test_init_importance(self, si):
        """Test importance initialization."""
        for name, importance in si.importance.items():
            assert importance.shape == si.model.state_dict()[name].shape
            assert torch.all(importance == 0)

    def test_update_importance(self, si, dataloader):
        """Test updating importance."""
        si.update_importance(dataloader, num_batches=2)
        
        # Check that importance was updated
        for name, importance in si.importance.items():
            # SI importance can be negative
            assert not torch.all(importance == 0)

    def test_get_si_loss(self, si, dataloader):
        """Test SI loss computation."""
        si.update_importance(dataloader, num_batches=2)
        loss = si.get_si_loss()
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_get_penalty(self, si, dataloader):
        """Test penalty computation for specific parameter."""
        si.update_importance(dataloader, num_batches=2)
        
        # Get first parameter name
        param_name = next(iter(si.importance.keys()))
        penalty = si.get_penalty(param_name)
        
        assert isinstance(penalty, torch.Tensor)

    def test_get_stats(self, si, dataloader):
        """Test statistics computation."""
        si.update_importance(dataloader, num_batches=2)
        stats = si.get_stats()
        
        assert isinstance(stats, SIStats)
        assert stats.num_parameters > 0

    def test_get_importance_dict(self, si):
        """Test getting importance dictionary."""
        importance_dict = si.get_importance_dict()
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) > 0

    def test_get_param_values(self, si):
        """Test getting parameter values."""
        # Initially empty
        param_values = si.get_param_values()
        assert len(param_values) == 0

    def test_reset(self, si, dataloader):
        """Test reset."""
        si.update_importance(dataloader, num_batches=2)
        si.reset()
        
        # Check that importance was reset
        for name, importance in si.importance.items():
            assert torch.all(importance == 0)
        assert len(si.saved_params) == 0
        assert len(si.prev_grads) == 0

    def test_update_lambda(self, si):
        """Test updating lambda."""
        si.update_lambda(200.0)
        assert si.lambda_si == 200.0

    def test_repr(self, si):
        """Test string representation."""
        repr_str = repr(si)
        assert "SI" in repr_str
        assert "lambda" in repr_str


class TestSIStats:
    """Tests for SIStats class."""

    def test_init(self):
        """Test initialization."""
        stats = SIStats()
        assert stats.num_parameters == 0
        assert stats.mean_importance == 0.0
        assert stats.max_importance == 0.0
