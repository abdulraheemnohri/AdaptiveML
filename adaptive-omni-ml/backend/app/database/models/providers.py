"""
Provider, ProviderModel, and RoutingRule models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class Provider(Base):
    """External AI provider configuration."""
    
    __tablename__ = "providers"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_type: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, anthropic, google, etc.
    
    # Configuration
    base_url: Mapped[str] = mapped_column(String(500), nullable=False)
    api_key_encrypted: Mapped[str | None] = mapped_column(String(500), nullable=True)
    
    # Status
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    last_health_check: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    # Rate limiting
    requests_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_per_minute: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Cost tracking
    cost_per_1k_input: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_1k_output: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Capabilities
    capabilities: Mapped[list] = mapped_column(JSON, default=list)  # ['text', 'image', 'audio']
    max_context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    models: Mapped[list["ProviderModel"]] = relationship(
        back_populates="provider",
        cascade="all, delete-orphan",
    )
    routing_rules: Mapped[list["RoutingRule"]] = relationship(
        back_populates="provider",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<Provider(id={self.id}, name={self.name})>"


class ProviderModel(Base):
    """A model available from an external provider."""
    
    __tablename__ = "provider_models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider_id: Mapped[int] = mapped_column(Integer, ForeignKey("providers.id"), nullable=False)
    
    # Model info
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    provider_model_id: Mapped[str] = mapped_column(String(200), nullable=False)  # Actual model ID at provider
    
    # Capabilities
    capabilities: Mapped[list] = mapped_column(JSON, default=list)
    max_context_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Pricing
    cost_per_1k_input: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_1k_output: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Status
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    provider: Mapped["Provider"] = relationship(back_populates="models")
    
    def __repr__(self) -> str:
        return f"<ProviderModel(id={self.id}, name={self.name})>"


class RoutingRule(Base):
    """Routing rule for AI request routing."""
    
    __tablename__ = "routing_rules"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Provider
    provider_id: Mapped[int] = mapped_column(Integer, ForeignKey("providers.id"), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)  # Higher = more priority
    
    # Conditions
    task_types: Mapped[list] = mapped_column(JSON, default=list)  # e.g., ['coding', 'reasoning']
    capabilities_required: Mapped[list] = mapped_column(JSON, default=list)
    privacy_sensitive: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # True = route to local
    max_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_cost_per_request: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Routing mode
    routing_mode: Mapped[str] = mapped_column(String(20), default="manual")  # manual, automatic, fallback
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    provider: Mapped["Provider"] = relationship(back_populates="routing_rules")
    
    def __repr__(self) -> str:
        return f"<RoutingRule(id={self.id}, name={self.name})>"
