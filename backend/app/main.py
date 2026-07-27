"""
Adaptive Omni ML - Backend API Server
FastAPI-based backend for the Adaptive Multimodal Continual Learning AI Platform
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

app = FastAPI(
    title="Adaptive Omni ML API",
    description="API for the Adaptive Multimodal Continual Learning AI Platform built on Qwen2.5-Omni-3B",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class SystemHealth(BaseModel):
    cpu: float
    gpu: float
    vram: int
    ram: int
    disk: float
    training_speed: float
    model_latency: float


class ModelInfo(BaseModel):
    name: str
    version: str
    status: str
    adapters: List[str]
    last_updated: str


class LearningProgress(BaseModel):
    knowledge_growth: float
    skills_growth: float
    languages_count: int
    new_data_count: int
    verified_facts: int
    learning_tasks: int
    forgetting_score: float


class DashboardResponse(BaseModel):
    system_health: SystemHealth
    current_model: ModelInfo
    learning_progress: LearningProgress
    active_experiments: int
    data_pipeline_status: str


class DataSource(BaseModel):
    id: str
    name: str
    type: str
    status: str
    last_collection: Optional[str] = None
    schedule: Optional[str] = None


class Dataset(BaseModel):
    id: str
    name: str
    source_id: str
    size: int
    quality_score: float
    trust_score: float
    status: str
    created_at: str


class TrainingConfig(BaseModel):
    task_id: str
    learning_rate: float = 1e-4
    epochs: int = 3
    batch_size: int = 32
    replay_ratio: float = 0.3
    use_ewc: bool = True
    use_distillation: bool = True
    ewc_lambda: float = 1000.0
    distill_alpha: float = 0.5


class Experiment(BaseModel):
    id: str
    name: str
    base_model: str
    dataset: str
    config: TrainingConfig
    status: str
    results: Optional[Dict[str, Any]] = None


class ModelVersion(BaseModel):
    version: str
    parent_model: Optional[str] = None
    base_model: str
    adapters: List[str]
    datasets: List[str]
    metrics: Dict[str, float]
    created_at: str
    status: str


# Mock data store
mock_health = SystemHealth(
    cpu=15.2,
    gpu=0.0,
    vram=2048,
    ram=4096,
    disk=45.3,
    training_speed=0.0,
    model_latency=0.0
)

mock_model = ModelInfo(
    name="Qwen2.5-Omni-3B",
    version="v1.0.0",
    status="production",
    adapters=[],
    last_updated="2025-01-15T10:30:00Z"
)

mock_progress = LearningProgress(
    knowledge_growth=12.5,
    skills_growth=8.3,
    languages_count=3,
    new_data_count=24830,
    verified_facts=12400,
    learning_tasks=34,
    forgetting_score=0.004
)

mock_data_sources: List[DataSource] = [
    DataSource(id="1", name="Wikipedia", type="web", status="active", last_collection="2025-01-15T09:00:00Z", schedule="0 */6 * * *"),
    DataSource(id="2", name="ArXiv Papers", type="web", status="active", last_collection="2025-01-15T08:00:00Z", schedule="0 */12 * * *"),
    DataSource(id="3", name="YouTube Transcripts", type="video", status="active", last_collection="2025-01-15T07:00:00Z", schedule="0 0 * * *"),
    DataSource(id="4", name="GitHub Repos", type="git", status="paused", last_collection="2025-01-14T12:00:00Z"),
]

mock_datasets: List[Dataset] = [
    Dataset(id="ds1", name="Science QA Dataset", source_id="1", size=15000, quality_score=0.95, trust_score=0.92, status="validated", created_at="2025-01-15T09:00:00Z"),
    Dataset(id="ds2", name="ML Papers 2024", source_id="2", size=5000, quality_score=0.88, trust_score=0.90, status="processing", created_at="2025-01-15T08:00:00Z"),
]

mock_experiments: List[Experiment] = [
    Experiment(id="exp1", name="Urdu Adapter Training", base_model="Qwen2.5-Omni-3B", dataset="ds1", config=TrainingConfig(task_id="urdu_v2"), status="completed", results={"accuracy": 0.89}),
    Experiment(id="exp2", name="Coding Skills Enhancement", base_model="Qwen2.5-Omni-3B", dataset="ds2", config=TrainingConfig(task_id="coding_v1"), status="running"),
]

mock_models: List[ModelVersion] = [
    ModelVersion(version="v1.0.0", base_model="Qwen2.5-Omni-3B", adapters=[], datasets=[], metrics={"accuracy": 0.85}, created_at="2025-01-01T00:00:00Z", status="production"),
    ModelVersion(version="v1.1.0", parent_model="v1.0.0", base_model="Qwen2.5-Omni-3B", adapters=["urdu-v1"], datasets=["ds1"], metrics={"accuracy": 0.87, "urdu_score": 0.89}, created_at="2025-01-10T00:00:00Z", status="archived"),
]


# API Routes

@app.get("/")
async def root():
    return {"message": "Adaptive Omni ML API", "version": "1.0.0"}


@app.get("/api/health")
async def get_health():
    """Get system health metrics"""
    return mock_health


@app.get("/api/dashboard")
async def get_dashboard():
    """Get complete dashboard data"""
    return DashboardResponse(
        system_health=mock_health,
        current_model=mock_model,
        learning_progress=mock_progress,
        active_experiments=len([e for e in mock_experiments if e.status == "running"]),
        data_pipeline_status="active"
    )


@app.get("/api/data-sources")
async def list_data_sources():
    """List all configured data sources"""
    return mock_data_sources


@app.post("/api/data-sources")
async def create_data_source(source: DataSource):
    """Create a new data source"""
    mock_data_sources.append(source)
    return {"message": "Data source created", "id": source.id}


@app.get("/api/datasets")
async def list_datasets(status: Optional[str] = None):
    """List datasets with optional status filter"""
    if status:
        return [d for d in mock_datasets if d.status == status]
    return mock_datasets


@app.get("/api/experiments")
async def list_experiments():
    """List all experiments"""
    return mock_experiments


@app.post("/api/experiments")
async def create_experiment(experiment: Experiment):
    """Create a new experiment"""
    mock_experiments.append(experiment)
    return {"message": "Experiment created", "id": experiment.id}


@app.get("/api/models")
async def list_models():
    """List all model versions"""
    return mock_models


@app.get("/api/models/current")
async def get_current_model():
    """Get current production model"""
    return mock_model


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start a new training cycle"""
    return {
        "message": "Training started",
        "task_id": config.task_id,
        "status": "running"
    }


@app.post("/api/training/stop")
async def stop_training(task_id: str):
    """Stop a running training cycle"""
    return {"message": f"Training stopped for task {task_id}"}


@app.get("/api/learning/gaps")
async def detect_knowledge_gaps():
    """Detect knowledge gaps in the current model"""
    return {
        "gaps": [
            {"domain": "Quantum Physics", "priority": 0.85, "confidence": 0.45},
            {"domain": "Advanced Mathematics", "priority": 0.72, "confidence": 0.52},
            {"domain": "Medical Diagnostics", "priority": 0.68, "confidence": 0.38},
        ]
    }


@app.post("/api/models/promote")
async def promote_model(version: str):
    """Promote a model version to production"""
    return {"message": f"Model {version} promoted to production"}


@app.post("/api/models/rollback")
async def rollback_model():
    """Rollback to previous model version"""
    return {"message": "Model rolled back successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
