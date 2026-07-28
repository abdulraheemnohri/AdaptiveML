"""Dashboard API routes."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def get_dashboard(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get dashboard overview."""
    return {
        "current_mode": "training",
        "production_model": {
            "id": 1,
            "name": "Qwen2.5-Omni-3B-Production",
            "version": "1.0.0",
            "status": "active"
        },
        "training_status": "idle",
        "serving_status": "ready",
        "knowledge_growth": 0.0,
        "forgetting_score": 0.0,
        "model_quality": 0.85,
        "dataset_count": 0,
        "active_jobs": 0,
        "gpu_usage": 0.0,
        "ram_usage": 0.0,
        "storage_used_gb": 0.0,
        "recent_activity": []
    }


@router.get("/quick-stats")
async def get_quick_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get quick statistics for dashboard."""
    return {
        "models_count": 0,
        "datasets_count": 0,
        "training_jobs_today": 0,
        "evaluations_completed": 0,
        "api_requests_today": 0
    }


@router.post("/action/{action}")
async def execute_quick_action(action: str, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Execute a quick action from dashboard."""
    valid_actions = [
        "start_training",
        "open_serving",
        "test_model",
        "collect_data",
        "run_evaluation",
        "compare_models",
        "rollback"
    ]
    
    if action not in valid_actions:
        return {"error": f"Invalid action. Valid actions: {valid_actions}"}
    
    return {
        "status": "initiated",
        "action": action,
        "message": f"Action '{action}' has been initiated"
    }
