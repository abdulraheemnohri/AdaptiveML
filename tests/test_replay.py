"""Tests for Replay Buffer."""

import pytest
import torch

from adaptive_ml.core.config import AdaptiveMLConfig
from adaptive_ml.core.types import SamplingStrategy
from adaptive_ml.memory.replay import ReplayBuffer, ReplayStats


@pytest.fixture
def config():
    """Create a test configuration."""
    config = AdaptiveMLConfig()
    config.memory.buffer_size = 100
    return config


@pytest.fixture
def replay_buffer(config):
    """Create a test replay buffer."""
    return ReplayBuffer(config)


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_init(self, replay_buffer):
        """Test initialization."""
        assert replay_buffer.capacity == 100
        assert len(replay_buffer) == 0
        assert replay_buffer.default_strategy == SamplingStrategy.BALANCED

    def test_add(self, replay_buffer):
        """Test adding entries."""
        replay_buffer.add(
            task_id="task_a",
            data="data1",
            label=0,
        )
        assert len(replay_buffer) == 1
        
        replay_buffer.add(
            task_id="task_a",
            data="data2",
            label=1,
        )
        assert len(replay_buffer) == 2

    def test_add_with_metadata(self, replay_buffer):
        """Test adding entries with metadata."""
        replay_buffer.add(
            task_id="task_a",
            data="data1",
            label=0,
            importance=2.0,
            uncertainty=0.5,
            metadata={"key": "value"},
        )
        
        entry = replay_buffer[0]
        assert entry.importance == 2.0
        assert entry.uncertainty == 0.5
        assert entry.metadata["key"] == "value"

    def test_reservoir_sampling(self, config):
        """Test reservoir sampling when buffer is full."""
        config.memory.buffer_size = 5
        buffer = ReplayBuffer(config)
        
        # Fill buffer
        for i in range(5):
            buffer.add(task_id="task_a", data=f"data_{i}", label=i)
        
        assert len(buffer) == 5
        
        # Add more entries (should use reservoir sampling)
        for i in range(5, 15):
            buffer.add(task_id="task_a", data=f"data_{i}", label=i)
        
        # Buffer should still be size 5
        assert len(buffer) == 5

    def test_sample_uniform(self, replay_buffer):
        """Test uniform sampling."""
        # Add some data
        for i in range(20):
            replay_buffer.add(task_id="task_a", data=f"data_{i}", label=i)
        
        # Sample
        data, labels, metadata = replay_buffer.sample(
            batch_size=5,
            strategy=SamplingStrategy.UNIFORM,
        )
        
        assert len(data) == 5
        assert len(labels) == 5
        assert len(metadata["task_ids"]) == 5

    def test_sample_balanced(self, config):
        """Test balanced sampling."""
        config.memory.buffer_size = 100
        buffer = ReplayBuffer(config)
        
        # Add data from two tasks
        for i in range(10):
            buffer.add(task_id="task_a", data=f"a_{i}", label=0)
        for i in range(10):
            buffer.add(task_id="task_b", data=f"b_{i}", label=1)
        
        # Sample with balanced strategy
        data, labels, metadata = buffer.sample(
            batch_size=10,
            strategy=SamplingStrategy.BALANCED,
        )
        
        assert len(data) == 10
        # Check that both tasks are represented
        task_ids = metadata["task_ids"]
        assert "task_a" in task_ids
        assert "task_b" in task_ids

    def test_sample_importance(self, config):
        """Test importance-weighted sampling."""
        config.memory.buffer_size = 100
        buffer = ReplayBuffer(config)
        
        # Add data with different importance
        for i in range(10):
            buffer.add(
                task_id="task_a",
                data=f"data_{i}",
                label=i,
                importance=1.0 if i < 5 else 10.0,  # Last 5 are more important
            )
        
        # Sample with importance strategy
        data, labels, metadata = buffer.sample(
            batch_size=10,
            strategy=SamplingStrategy.IMPORTANCE,
            return_entries=False,
        )
        
        assert len(data) == 10

    def test_get_stats(self, replay_buffer):
        """Test get_stats method."""
        # Add data
        for i in range(10):
            replay_buffer.add(task_id="task_a", data=f"a_{i}", label=0)
        for i in range(5):
            replay_buffer.add(task_id="task_b", data=f"b_{i}", label=1)
        
        stats = replay_buffer.get_stats()
        
        assert stats.size == 15
        assert stats.capacity == 100
        assert stats.num_tasks == 2
        assert stats.task_distribution["task_a"] == 10
        assert stats.task_distribution["task_b"] == 5

    def test_get_task_data(self, replay_buffer):
        """Test get_task_data method."""
        replay_buffer.add(task_id="task_a", data="a1", label=0)
        replay_buffer.add(task_id="task_a", data="a2", label=1)
        replay_buffer.add(task_id="task_b", data="b1", label=0)
        
        task_a_data = replay_buffer.get_task_data("task_a")
        assert len(task_a_data) == 2
        
        task_b_data = replay_buffer.get_task_data("task_b")
        assert len(task_b_data) == 1

    def test_clear(self, replay_buffer):
        """Test clear method."""
        replay_buffer.add(task_id="task_a", data="data1", label=0)
        replay_buffer.add(task_id="task_a", data="data2", label=1)
        
        assert len(replay_buffer) == 2
        
        replay_buffer.clear()
        assert len(replay_buffer) == 0

    def test_remove_task(self, replay_buffer):
        """Test remove_task method."""
        replay_buffer.add(task_id="task_a", data="a1", label=0)
        replay_buffer.add(task_id="task_a", data="a2", label=1)
        replay_buffer.add(task_id="task_b", data="b1", label=0)
        
        assert len(replay_buffer) == 3
        
        num_removed = replay_buffer.remove_task("task_a")
        assert num_removed == 2
        assert len(replay_buffer) == 1

    def test_to_list(self, replay_buffer):
        """Test to_list method."""
        replay_buffer.add(task_id="task_a", data="data1", label=0)
        replay_buffer.add(task_id="task_a", data="data2", label=1)
        
        entries = replay_buffer.to_list()
        assert len(entries) == 2

    def test_repr(self, replay_buffer):
        """Test __repr__ method."""
        replay_buffer.add(task_id="task_a", data="data1", label=0)
        
        repr_str = repr(replay_buffer)
        assert "ReplayBuffer" in repr_str
        assert "size=1" in repr_str


class TestReplayStats:
    """Tests for ReplayStats dataclass."""

    def test_init(self):
        """Test initialization."""
        stats = ReplayStats()
        assert stats.size == 0
        assert stats.capacity == 0
        assert stats.num_tasks == 0
        assert stats.utilization == 0.0

    def test_utilization(self):
        """Test utilization calculation."""
        stats = ReplayStats(size=50, capacity=100)
        assert stats.utilization == 0.5
