"""
Evaluation, test suite, and benchmark result models.
"""

from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Text, Integer, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from backend.app.database.session import Base


class EvaluationRun(Base):
    """A model evaluation run."""
    
    __tablename__ = "evaluation_runs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Target
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("models.id"), nullable=False)
    model_version_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("model_versions.id"), nullable=True)
    
    # Configuration
    test_suite_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("test_suites.id"), nullable=True)
    evaluation_type: Mapped[str] = mapped_column(String(50), default="general")  # general, benchmark, regression, forgetting
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed
    
    # Results summary
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tests: Mapped[int | None] = mapped_column(Integer, nullable=True)
    passed_tests: Mapped[int | None] = mapped_column(Integer, nullable=True)
    failed_tests: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Detailed results
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model: Mapped["Model"] = relationship(back_populates="evaluations")
    test_suite: Mapped["TestSuite | None"] = relationship(back_populates="evaluation_runs")
    benchmark_results: Mapped[list["BenchmarkResult"]] = relationship(
        back_populates="evaluation_run",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<EvaluationRun(id={self.id}, model_id={self.model_id})>"


class TestSuite(Base):
    """A collection of tests for model evaluation."""
    
    __tablename__ = "test_suites"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    # Test configuration
    tests: Mapped[list] = mapped_column(JSON, default=list)  # List of test configurations
    datasets: Mapped[list] = mapped_column(JSON, default=list)  # Dataset IDs for testing
    
    # Categories covered
    categories: Mapped[list] = mapped_column(JSON, default=list)  # e.g., ['reasoning', 'math', 'coding', 'vision']
    
    # Thresholds
    passing_threshold: Mapped[float | None] = mapped_column(Float, default=0.7)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)  # Built-in vs custom
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    evaluation_runs: Mapped[list["EvaluationRun"]] = relationship(back_populates="test_suite")
    
    def __repr__(self) -> str:
        return f"<TestSuite(id={self.id}, name={self.name})>"


class BenchmarkResult(Base):
    """Result of a specific benchmark test."""
    
    __tablename__ = "benchmark_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("evaluation_runs.id"), nullable=False)
    
    # Benchmark info
    benchmark_name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'reasoning', 'math', 'coding'
    
    # Metrics
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rouge_l: Mapped[float | None] = mapped_column(Float, nullable=True)
    bleu: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Performance
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_per_second: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Details
    num_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    correct_predictions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Comparison with baseline
    baseline_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    improvement: Mapped[float | None] = mapped_column(Float, nullable=True)  # Percentage improvement
    regression_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    evaluation_run: Mapped["EvaluationRun"] = relationship(back_populates="benchmark_results")
    
    def __repr__(self) -> str:
        return f"<BenchmarkResult(id={self.id}, benchmark={self.benchmark_name})>"


# Import delayed to avoid circular imports
from backend.app.database.models.models import Model
