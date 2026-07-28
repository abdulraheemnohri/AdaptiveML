"""
Database models for Adaptive Omni ML platform.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from backend.app.database.session import Base


# ============== ENUMS ==============

class UserRole(enum.Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    TRAINER = "trainer"
    VIEWER = "viewer"


class ModelStatus(enum.Enum):
    DRAFT = "draft"
    TRAINING = "training"
    CANDIDATE = "candidate"
    TESTING = "testing"
    APPROVED = "approved"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class TrainingStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataSourceType(enum.Enum):
    WEBSITE = "website"
    RSS = "rss"
    SITEMAP = "sitemap"
    YOUTUBE = "youtube"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    GITHUB = "github"
    GIT = "git"
    LOCAL_FOLDER = "local_folder"
    DATABASE = "database"
    CUSTOM = "custom"


class ServingMode(enum.Enum):
    LOCAL_ONLY = "local_only"
    API_ONLY = "api_only"
    LOCAL_FIRST = "local_first"
    API_FIRST = "api_first"
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class ProviderType(enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    XAI = "xai"
    COHERE = "cohere"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


# ============== USER & AUTH ==============

class User(Base):
    """User model with role-based access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    permissions = relationship("Permission", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class Permission(Base):
    """Permission model for fine-grained access control."""
    __tablename__ = "permissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    resource = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    granted_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="permissions")


# ============== SETTINGS ==============

class SystemSetting(Base):
    """System-wide settings storage."""
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    category = Column(String(50))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# ============== DATA SOURCES ==============

class DataSource(Base):
    """External data sources for collection."""
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    source_type = Column(SQLEnum(DataSourceType), nullable=False)
    url = Column(String(2048))
    config = Column(JSON, default={})
    is_enabled = Column(Boolean, default=True)
    last_sync = Column(DateTime(timezone=True))
    next_sync = Column(DateTime(timezone=True))
    sync_schedule = Column(String(100))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    collections = relationship("DataCollection", back_populates="source")


class DataCollection(Base):
    """Collections from data sources."""
    __tablename__ = "data_collections"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"))
    status = Column(String(50), default="pending")
    items_collected = Column(Integer, default=0)
    items_processed = Column(Integer, default=0)
    items_failed = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    source = relationship("DataSource", back_populates="collections")
    documents = relationship("Document", back_populates="collection")


class Document(Base):
    """Individual documents/items from collections."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("data_collections.id"))
    title = Column(String(500))
    content = Column(Text)
    content_hash = Column(String(64), index=True)
    source_url = Column(String(2048))
    file_path = Column(String(500))
    file_type = Column(String(50))
    language = Column(String(10))
    word_count = Column(Integer)
    token_count = Column(Integer)
    quality_score = Column(Float)
    trust_score = Column(Float)
    is_deduplicated = Column(Boolean, default=False)
    is_safe = Column(Boolean, default=True)
    safety_flags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    collection = relationship("DataCollection", back_populates="documents")


# ============== DATASETS ==============

class Dataset(Base):
    """Training datasets."""
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    parent_dataset_id = Column(Integer, ForeignKey("datasets.id"))
    total_samples = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    languages = Column(JSON, default=[])
    avg_quality_score = Column(Float)
    avg_trust_score = Column(Float)
    duplicate_count = Column(Integer, default=0)
    source_ids = Column(JSON, default=[])
    is_locked = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    versions = relationship("DatasetVersion", back_populates="dataset")
    samples = relationship("DatasetSample", back_populates="dataset")


class DatasetVersion(Base):
    """Dataset versions for tracking changes."""
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    version = Column(String(50), nullable=False)
    change_description = Column(Text)
    samples_added = Column(Integer, default=0)
    samples_removed = Column(Integer, default=0)
    checksum = Column(String(64))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="versions")


class DatasetSample(Base):
    """Individual samples in a dataset."""
    __tablename__ = "dataset_samples"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))
    input_text = Column(Text)
    output_text = Column(Text)
    modality = Column(String(50))
    language = Column(String(10))
    quality_score = Column(Float)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="samples")


# ============== TRAINING ==============

class TrainingJob(Base):
    """Training job records."""
    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    base_model_id = Column(Integer, ForeignKey("models.id"))
    status = Column(SQLEnum(TrainingStatus), default=TrainingStatus.PENDING)
    training_type = Column(String(50))
    config = Column(JSON, default={})
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer)
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer)
    loss = Column(Float)
    validation_loss = Column(Float)
    learning_rate = Column(Float)
    gpu_usage = Column(Float)
    vram_usage = Column(Float)
    eta_seconds = Column(Integer)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    checkpoint_path = Column(String(500))
    adapter_path = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    checkpoints = relationship("TrainingCheckpoint", back_populates="job")


class TrainingCheckpoint(Base):
    """Training checkpoints."""
    __tablename__ = "training_checkpoints"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"))
    epoch = Column(Integer)
    step = Column(Integer)
    loss = Column(Float)
    path = Column(String(500), nullable=False)
    optimizer_state_path = Column(String(500))
    scheduler_state_path = Column(String(500))
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("TrainingJob", back_populates="checkpoints")


class Experiment(Base):
    """Training experiments for comparison."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    config = Column(JSON, default={})
    status = Column(String(50), default="running")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True))

    jobs = relationship("TrainingJob", back_populates="experiment")


# ============== MODELS ==============

class Model(Base):
    """Model registry entries."""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    parent_model_id = Column(Integer, ForeignKey("models.id"))
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.DRAFT)
    model_type = Column(String(50))
    path = Column(String(500))
    adapter_path = Column(String(500))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    dataset_version = Column(String(50))
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    config = Column(JSON, default={})
    hardware_config = Column(JSON, default={})
    benchmark_results = Column(JSON, default={})
    forgetting_score = Column(Float, default=0.0)
    safety_score = Column(Float, default=1.0)
    quality_score = Column(Float)
    vram_required = Column(Integer)
    ram_required = Column(Integer)
    avg_latency_ms = Column(Float)
    throughput_tokens_s = Column(Float)
    tags = Column(JSON, default=[])
    is_quantized = Column(Boolean, default=False)
    quantization_type = Column(String(20))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True))
    deployed_at = Column(DateTime(timezone=True))
    archived_at = Column(DateTime(timezone=True))

    deployments = relationship("ModelDeployment", back_populates="model")
    adapters = relationship("ModelAdapter", back_populates="model")


class ModelDeployment(Base):
    """Model deployment records."""
    __tablename__ = "model_deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    environment = Column(String(50), default="production")
    is_active = Column(Boolean, default=True)
    deployed_by = Column(Integer, ForeignKey("users.id"))
    deployment_config = Column(JSON, default={})
    deployed_at = Column(DateTime(timezone=True), server_default=func.now())
    rolled_back_at = Column(DateTime(timezone=True))
    rollback_reason = Column(Text)

    model = relationship("Model", back_populates="deployments")


class ModelAdapter(Base):
    """LoRA/QLoRA adapter records."""
    __tablename__ = "model_adapters"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    name = Column(String(255), nullable=False)
    adapter_type = Column(String(50))
    path = Column(String(500), nullable=False)
    rank = Column(Integer, default=8)
    alpha = Column(Integer, default=16)
    target_modules = Column(JSON, default=[])
    config = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("Model", back_populates="adapters")


# ============== EVALUATION ==============

class EvaluationRun(Base):
    """Evaluation run records."""
    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    test_suite_id = Column(Integer, ForeignKey("test_suites.id"))
    status = Column(String(50), default="pending")
    overall_score = Column(Float)
    results = Column(JSON, default={})
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    report_path = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TestSuite(Base):
    """Test suite definitions."""
    __tablename__ = "test_suites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    tests = Column(JSON, default=[])
    is_builtin = Column(Boolean, default=False)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class BenchmarkResult(Base):
    """Benchmark test results."""
    __tablename__ = "benchmark_results"

    id = Column(Integer, primary_key=True, index=True)
    evaluation_run_id = Column(Integer, ForeignKey("evaluation_runs.id"))
    benchmark_name = Column(String(100), nullable=False)
    category = Column(String(50))
    score = Column(Float)
    metrics = Column(JSON, default={})
    samples_evaluated = Column(Integer)
    samples_passed = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============== AI PROVIDERS ==============

class AIProvider(Base):
    """External AI provider configurations."""
    __tablename__ = "ai_providers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    provider_type = Column(SQLEnum(ProviderType), nullable=False)
    api_key_encrypted = Column(String(500))
    endpoint_url = Column(String(500))
    default_model = Column(String(100))
    available_models = Column(JSON, default=[])
    is_enabled = Column(Boolean, default=True)
    priority = Column(Integer, default=0)
    rate_limit_per_minute = Column(Integer)
    budget_limit_monthly = Column(Float)
    current_usage = Column(Float, default=0.0)
    config = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class RoutingRule(Base):
    """AI routing rules configuration."""
    __tablename__ = "routing_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    priority = Column(Integer, default=0)
    conditions = Column(JSON, default={})
    target = Column(String(100), nullable=False)
    fallback_target = Column(String(100))
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============== CONVERSATIONS & MEMORY ==============

class Conversation(Base):
    """Chat conversations."""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(500))
    model_used = Column(String(100))
    provider_used = Column(String(100))
    serving_mode = Column(String(50))
    is_archived = Column(Boolean, default=False)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation")


class Message(Base):
    """Chat messages."""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text")
    attachments = Column(JSON, default=[])
    model_response_metadata = Column(JSON, default={})
    latency_ms = Column(Integer)
    tokens_used = Column(Integer)
    cost_usd = Column(Float)
    feedback_score = Column(Integer)
    feedback_comment = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")


class Memory(Base):
    """Long-term memory storage."""
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    content = Column(Text, nullable=False)
    memory_type = Column(String(50))
    importance = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============== KNOWLEDGE ==============

class KnowledgeItem(Base):
    """Knowledge base items."""
    __tablename__ = "knowledge_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    category = Column(String(100))
    tags = Column(JSON, default=[])
    source_document_id = Column(Integer, ForeignKey("documents.id"))
    embedding_model = Column(String(100))
    embedding = Column(JSON)
    is_verified = Column(Boolean, default=False)
    verification_source = Column(String(255))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class KnowledgeGap(Base):
    """Identified knowledge gaps."""
    __tablename__ = "knowledge_gaps"

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String(500), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    priority = Column(Integer, default=0)
    source = Column(String(100))
    related_benchmark = Column(String(100))
    status = Column(String(50), default="identified")
    research_task_id = Column(Integer, ForeignKey("agent_tasks.id"))
    resolution_notes = Column(Text)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))


# ============== AGENTS ==============

class Agent(Base):
    """Autonomous agent definitions."""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    agent_type = Column(String(100))
    description = Column(Text)
    config = Column(JSON, default={})
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AgentTask(Base):
    """Agent task records."""
    __tablename__ = "agent_tasks"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), default="pending")
    input_data = Column(JSON, default={})
    output_data = Column(JSON, default={})
    error_message = Column(Text)
    progress = Column(Float, default=0.0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============== MONITORING ==============

class AuditLog(Base):
    """Audit log for security and compliance."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(Integer)
    details = Column(JSON, default={})
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="audit_logs")


class SystemEvent(Base):
    """System events for monitoring."""
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="info")
    message = Column(Text, nullable=False)
    source = Column(String(100))
    details = Column(JSON, default={})
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    acknowledged_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Alert(Base):
    """System alerts."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), default="warning")
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============== BACKUPS ==============

class Backup(Base):
    """Backup records."""
    __tablename__ = "backups"

    id = Column(Integer, primary_key=True, index=True)
    backup_type = Column(String(50), nullable=False)
    path = Column(String(500), nullable=False)
    size_bytes = Column(Integer)
    checksum = Column(String(64))
    status = Column(String(50), default="pending")
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
