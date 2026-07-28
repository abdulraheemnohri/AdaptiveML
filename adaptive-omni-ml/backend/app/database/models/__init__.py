"""
Database Models

All SQLAlchemy models for the Adaptive Omni ML platform.
"""

# Import all models to register them with Base.metadata
from backend.app.database.models.users import User, Role, Permission
from backend.app.database.models.settings import SystemSetting
from backend.app.database.models.data_sources import DataSource, DataCollection
from backend.app.database.models.documents import Document
from backend.app.database.models.datasets import Dataset, DatasetVersion, DatasetSample
from backend.app.database.models.training import TrainingJob, TrainingCheckpoint, Experiment
from backend.app.database.models.evaluation import EvaluationRun, TestSuite, BenchmarkResult
from backend.app.database.models.models import Model, ModelVersion, ModelDeployment, Adapter
from backend.app.database.models.providers import Provider, ProviderModel, RoutingRule
from backend.app.database.models.conversations import Conversation, Message
from backend.app.database.models.memory import Memory, KnowledgeItem, KnowledgeGap
from backend.app.database.models.agents import Agent, AgentTask
from backend.app.database.models.audit import AuditLog, SystemEvent, Alert

__all__ = [
    # Users & Auth
    "User",
    "Role",
    "Permission",
    # Settings
    "SystemSetting",
    # Data
    "DataSource",
    "DataCollection",
    "Document",
    # Datasets
    "Dataset",
    "DatasetVersion",
    "DatasetSample",
    # Training
    "TrainingJob",
    "TrainingCheckpoint",
    "Experiment",
    # Evaluation
    "EvaluationRun",
    "TestSuite",
    "BenchmarkResult",
    # Models
    "Model",
    "ModelVersion",
    "ModelDeployment",
    "Adapter",
    # Providers
    "Provider",
    "ProviderModel",
    "RoutingRule",
    # Conversations
    "Conversation",
    "Message",
    # Memory & Knowledge
    "Memory",
    "KnowledgeItem",
    "KnowledgeGap",
    # Agents
    "Agent",
    "AgentTask",
    # Audit
    "AuditLog",
    "SystemEvent",
    "Alert",
]
