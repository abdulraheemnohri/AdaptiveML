"""Tests for Elastic Weight Consolidation (EWC)."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adaptive_ml.training.ewc import EWC, EWCStats


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )


@pytest.fixture
def ewc(simple_model):
    """Create an EWC instance."""
    return EWC(simple_model, lambda_ewc=1000.0)


@pytest.fixture
def dataloader():
    """Create a simple dataloader."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


class TestEWC:
    """Tests for EWC class."""

    def test_init(self, simple_model, ewc):
        """Test initialization."""
        assert ewc.model == simple_model
        assert ewc.lambda_ewc == 1000.0
        assert ewc.fisher_diagonal is True
        assert len(ewc.fisher) > 0
        assert len(ewc.param_values) > 0

    def test_update_fisher(self, simple_model, dataloader):
        """Test updating Fisher Information."""
        ewc = EWC(simple_model, lambda_ewc=1000.0)
        
        # Update Fisher
        ewc.update_fisher(dataloader)
        
        # Check that Fisher has been updated
        for name, fisher in ewc.fisher.items():
            # Fisher should have non-zero values after update
            assert fisher.sum() > 0

    def test_penalty(self, simple_model, dataloader):
        """Test penalty computation."""
        ewc = EWC(simple_model, lambda_ewc=1000.0)
        
        # Update Fisher
        ewc.update_fisher(dataloader)
        
        # Compute penalty
        penalty = ewc.penalty()
        
        # Penalty should be a scalar tensor
        assert isinstance(penalty, torch.Tensor)
        assert penalty.dim() == 0
        
        # Penalty should be non-negative
        assert penalty.item() >= 0

    def test_get_ewc_loss(self, simple_model, dataloader):
        """Test EWC loss computation."""
        ewc = EWC(simple_model, lambda_ewc=1000.0)
        ewc.update_fisher(dataloader)
        
        # Create a task loss
        task_loss = torch.tensor(1.0)
        
        # Compute EWC loss
        total_loss = ewc.get_ewc_loss(task_loss)
        
        # Total loss should be greater than task loss (due to penalty)
        assert total_loss.item() >= task_loss.item()

    def test_update_lambda(self, ewc):
        """Test updating lambda."""
        assert ewc.lambda_ewc == 1000.0
        
        ewc.update_lambda(500.0)
        assert ewc.lambda_ewc == 500.0

    def test_reset(self, simple_model, dataloader):
        """Test reset method."""
        ewc = EWC(simple_model, lambda_ewc=1000.0)
        ewc.update_fisher(dataloader)
        
        # Fisher should have non-zero values
        for fisher in ewc.fisher.values():
            assert fisher.sum() > 0
        
        # Reset
        ewc.reset()
        
        # Fisher should be zero again
        for fisher in ewc.fisher.values():
            assert fisher.sum() == 0

    def test_get_stats(self, simple_model, dataloader):
        """Test get_stats method."""
        ewc = EWC(simple_model, lambda_ewc=1000.0)
        ewc.update_fisher(dataloader)
        
        stats = ewc.get_stats()
        
        assert isinstance(stats, EWCStats)
        assert stats.num_parameters > 0

    def test_get_fisher_dict(self, ewc):
        """Test get_fisher_dict method."""
        fisher_dict = ewc.get_fisher_dict()
        
        assert isinstance(fisher_dict, dict)
        assert len(fisher_dict) > 0

    def test_get_param_values(self, ewc):
        """Test get_param_values method."""
        param_values = ewc.get_param_values()
        
        assert isinstance(param_values, dict)
        assert len(param_values) > 0

    def test_repr(self, ewc):
        """Test __repr__ method."""
        repr_str = repr(ewc)
        
        assert "EWC" in repr_str
        assert "lambda=" in repr_str


class TestEWCStats:
    """Tests for EWCStats dataclass."""

    def test_init(self):
        """Test initialization."""
        stats = EWCStats()
        
        assert stats.num_parameters == 0
        assert stats.num_important_parameters == 0
        assert stats.mean_importance == 0.0
        assert stats.max_importance == 0.0
        assert stats.min_importance == 0.0
