"""
Adaptive Omni ML - Main Application Entry Point

One Adaptive Brain. Continuous Learning. Zero Catastrophic Forgetting.
Local Intelligence + Global AI.
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.app.config.settings import settings
from backend.app.api.routes import (
    auth,
    dashboard,
    modes,
    training,
    data_sources,
    data_pipeline,
    datasets,
    research,
    knowledge,
    knowledge_gaps,
    continual_learning,
    anti_forgetting,
    evaluation,
    benchmarks,
    experiments,
    models,
    model_registry,
    model_promotion,
    model_rollback,
    adapters,
    serving,
    local_model,
    providers,
    ai_router,
    conversations,
    memory,
    agents,
    monitoring,
    logs,
    system_settings,
    backups,
)
from backend.app.database.session import init_db
from backend.app.services.websocket_manager import websocket_manager


# Configure logging
logger.remove()
logger.add(
    settings.LOG_FILE,
    rotation="500 MB",
    retention="10 days",
    level=settings.LOG_LEVEL,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)
logger.add(sys.stdout, level=settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Adaptive Omni ML...")
    logger.info(f"📊 App Name: {settings.APP_NAME}")
    logger.info(f"🔧 Environment: {settings.APP_ENV}")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Create necessary directories
    for dir_path in [
        settings.MODEL_DIR,
        settings.DATASET_DIR,
        settings.CHECKPOINT_DIR,
        settings.ADAPTER_DIR,
        settings.EXPERIMENT_DIR,
        settings.EVALUATION_REPORT_DIR,
        settings.BACKUP_DIR,
        Path(settings.LOG_FILE).parent,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    logger.info("✅ Directories created")
    
    # Start background tasks
    background_tasks = []
    
    logger.info("✅ Adaptive Omni ML is ready!")
    logger.info("🧠 Training Mode: Available")
    logger.info("🚀 Serving Mode: Available")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Adaptive Omni ML...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    # Close WebSocket connections
    await websocket_manager.disconnect_all()
    
    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Adaptive Omni ML Platform

**One Adaptive Brain. Continuous Learning. Zero Catastrophic Forgetting. Local Intelligence + Global AI.**

### Features

#### 🧠 Training Mode
- Data acquisition and pipeline
- Continual learning with anti-forgetting
- Model training with LoRA/QLoRA
- Evaluation and testing
- Model registry and versioning
- Automatic promotion gates

#### 🚀 Serving Mode
- Local model inference (Qwen2.5-Omni-3B)
- Multi-provider AI API support
- Intelligent routing
- RAG and memory
- Unified chat workspace

### Architecture

The platform operates around a continuous learning loop:
1. Data Collection → Validation → Knowledge Gaps
2. Dataset Creation → Adaptive Training
3. Continual Learning → MemoryGuard Protection
4. Testing → Approval/Rejection
5. Production Model → Serving
6. User Feedback → New Knowledge (loop)
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(modes.router, prefix="/api/modes", tags=["Modes"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(data_sources.router, prefix="/api/data-sources", tags=["Data Sources"])
app.include_router(data_pipeline.router, prefix="/api/data-pipeline", tags=["Data Pipeline"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(research.router, prefix="/api/research", tags=["Research"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["Knowledge"])
app.include_router(knowledge_gaps.router, prefix="/api/knowledge-gaps", tags=["Knowledge Gaps"])
app.include_router(continual_learning.router, prefix="/api/continual-learning", tags=["Continual Learning"])
app.include_router(anti_forgetting.router, prefix="/api/anti-forgetting", tags=["Anti-Forgetting"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(benchmarks.router, prefix="/api/benchmarks", tags=["Benchmarks"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["Experiments"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(model_registry.router, prefix="/api/model-registry", tags=["Model Registry"])
app.include_router(model_promotion.router, prefix="/api/model-promotion", tags=["Model Promotion"])
app.include_router(model_rollback.router, prefix="/api/model-rollback", tags=["Model Rollback"])
app.include_router(adapters.router, prefix="/api/adapters", tags=["Adapters"])
app.include_router(serving.router, prefix="/api/serving", tags=["Serving"])
app.include_router(local_model.router, prefix="/api/local-model", tags=["Local Model"])
app.include_router(providers.router, prefix="/api/providers", tags=["AI Providers"])
app.include_router(ai_router.router, prefix="/api/ai-router", tags=["AI Router"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["Conversations"])
app.include_router(memory.router, prefix="/api/memory", tags=["Memory"])
app.include_router(agents.router, prefix="/api/agents", tags=["Agents"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(system_settings.router, prefix="/api/settings", tags=["Settings"])
app.include_router(backups.router, prefix="/api/backups", tags=["Backups"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": "0.1.0",
        "tagline": "One Adaptive Brain. Continuous Learning. Zero Catastrophic Forgetting.",
        "modes": ["training", "serving"],
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages
            await websocket_manager.broadcast({
                "type": "message",
                "data": data,
            })
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


def cli():
    """CLI entry point."""
    import click
    
    @click.group()
    def main():
        """Adaptive Omni ML CLI"""
        pass
    
    @main.command()
    @click.option("--host", default=settings.BACKEND_HOST, help="Host to bind")
    @click.option("--port", default=settings.BACKEND_PORT, help="Port to bind")
    @click.option("--reload", is_flag=True, help="Enable auto-reload")
    def serve(host, port, reload):
        """Start the API server."""
        uvicorn.run(
            "backend.app.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    
    @main.command()
    def setup():
        """Run initial setup."""
        from scripts.setup import run_setup
        run_setup()
    
    @main.command()
    def migrate():
        """Run database migrations."""
        from backend.app.database.migrate import run_migrations
        run_migrations()
    
    main()


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=(settings.APP_ENV == "development"),
        log_level="info",
    )
