"""Tests for MAS (Memory Aware Synapses) implementation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptive_ml.training.mas import MAS, MASStats


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
def mas(simple_model):
    """Create a MAS instance for testing."""
    return MAS(simple_model, lambda_mas=100.0)


@pytest.fixture
def dataloader():
    """Create a simple dataloader for testing."""
    X = torch.randn(50, 10)
    y = (X[:, 0] + X[:, 1] > 0).long()
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


class TestMAS:
    """Tests for MAS class."""

    def test_init(self, mas, simple_model):
        """Test initialization."""
        assert mas.model == simple_model
        assert mas.lambda_mas == 100.0
        assert len(mas.importance) > 0
        assert len(mas.saved_params) == 0

    def test_init_importance(self, mas):
        """Test importance initialization."""
        for name, importance in mas.importance.items():
            assert importance.shape == mas.model.state_dict()[name].shape
            assert torch.all(importance == 0)

    def test_update_importance(self, mas, dataloader):
        """Test updating importance."""
        mas.update_importance(dataloader, num_batches=2)
        
        # Check that importance was updated
        for name, importance in mas.importance.items():
            assert not torch.all(importance == 0)

    def test_get_mas_loss(self, mas, dataloader):
        """Test MAS loss computation."""
        mas.update_importance(dataloader, num_batches=2)
        loss = mas.get_mas_loss()
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_get_penalty(self, mas, dataloader):
        """Test penalty computation for specific parameter."""
        mas.update_importance(dataloader, num_batches=2)
        
        # Get first parameter name
        param_name = next(iter(mas.importance.keys()))
        penalty = mas.get_penalty(param_name)
        
        assert isinstance(penalty, torch.Tensor)

    def test_get_stats(self, mas, dataloader):
        """Test statistics computation."""
        mas.update_importance(dataloader, num_batches=2)
        stats = mas.get_stats()
        
        assert isinstance(stats, MASStats)
        assert stats.num_parameters > 0
        assert stats.mean_importance >= 0
        assert stats.max_importance >= 0

    def test_get_importance_dict(self, mas):
        """Test getting importance dictionary."""
        importance_dict = mas.get_importance_dict()
        
        assert isinstance(importance_dict, dict)
        assert len(importance_dict) > 0

    def test_get_param_values(self, mas):
        """Test getting parameter values."""
        # Initially empty
        param_values = mas.get_param_values()
        assert len(param_values) == 0

    def test_reset(self, mas, dataloader):
        """Test reset."""
        mas.update_importance(dataloader, num_batches=2)
        mas.reset()
        
        # Check that importance was reset
        for name, importance in mas.importance.items():
            assert torch.all(importance == 0)
        assert len(mas.saved_params) == 0

    def test_update_lambda(self, mas):
        """Test updating lambda."""
        mas.update_lambda(200.0)
        assert mas.lambda_mas == 200.0

    def test_repr(self, mas):
        """Test string representation."""
        repr_str = repr(mas)
        assert "MAS" in repr_str
        assert "lambda" in repr_str


class TestMASStats:
    """Tests for MASStats class."""

    def test_init(self):
        """Test initialization."""
        stats = MASStats()
        assert stats.num_parameters == 0
        assert stats.mean_importance == 0.0
        assert stats.max_importance == 0.0
