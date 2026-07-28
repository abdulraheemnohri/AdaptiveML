"""
Dashboard routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def get_dashboard_data():
    """Get main dashboard data."""
    return {
        "current_mode": "serving",  # or "training"
        "production_model": {
            "id": 1,
            "name": "Qwen2.5-Omni-3B-Adaptive-v1",
            "version": "1.0.0",
            "status": "active",
        },
        "training_status": {
            "is_training": False,
            "current_job": None,
        },
        "serving_status": {
            "is_serving": True,
            "engine": "local",
            "model_loaded": True,
        },
        "knowledge_growth": {
            "total_datasets": 5,
            "total_samples": 10000,
            "growth_rate": 0.15,
        },
        "forgetting_score": 0.01,
        "model_quality": 0.92,
        "dataset_count": 5,
        "active_jobs": [],
        "gpu_usage": {
            "utilization": 0.0,
            "vram_used_gb": 6.2,
            "vram_total_gb": 24,
        },
        "ram_usage": {
            "used_gb": 8.5,
            "total_gb": 32,
        },
        "storage": {
            "models_gb": 12.5,
            "datasets_gb": 2.3,
            "checkpoints_gb": 5.1,
        },
        "recent_activity": [
            {"type": "evaluation", "message": "Model evaluation completed", "timestamp": "2024-01-15T10:30:00Z"},
            {"type": "data_collection", "message": "Collected 500 new samples", "timestamp": "2024-01-15T09:15:00Z"},
        ],
    }


@router.get("/quick-actions")
async def get_quick_actions():
    """Get available quick actions."""
    return {
        "actions": [
            {"id": "start_training", "label": "Start Training", "icon": "🧠"},
            {"id": "open_serving", "label": "Open Serving", "icon": "🚀"},
            {"id": "test_model", "label": "Test Model", "icon": "🧪"},
            {"id": "collect_data", "label": "Collect Data", "icon": "📊"},
            {"id": "run_evaluation", "label": "Run Evaluation", "icon": "📈"},
            {"id": "compare_models", "label": "Compare Models", "icon": "⚖️"},
            {"id": "rollback", "label": "Roll Back", "icon": "↩️"},
        ],
    }


@router.get("/system-health")
async def get_system_health():
    """Get system health status."""
    return {
        "overall_status": "healthy",
        "components": {
            "backend": {"status": "healthy", "latency_ms": 5},
            "database": {"status": "healthy", "connections": 5},
            "model_engine": {"status": "healthy", "model_loaded": True},
            "gpu": {"status": "healthy", "temperature_c": 45},
            "redis": {"status": "healthy", "memory_mb": 128},
        },
    }
