"""Models API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def list_models(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all models in registry."""
    return []


@router.get("/production")
async def get_production_model(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get current production model."""
    return {
        "id": 1,
        "name": "Qwen2.5-Omni-3B-Production",
        "version": "1.0.0",
        "status": "active",
        "path": "./models/production/v1"
    }


@router.post("/candidate")
async def create_candidate_model(model_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Create a new candidate model from training."""
    return {
        "status": "created",
        "model_id": 1,
        "message": "Candidate model created"
    }


@router.post("/{model_id}/promote")
async def promote_to_production(model_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Promote a candidate model to production."""
    return {"status": "promoted", "model_id": model_id}


@router.post("/{model_id}/archive")
async def archive_model(model_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Archive a model."""
    return {"status": "archived", "model_id": model_id}


@router.post("/{model_id}/rollback")
async def rollback_model(model_id: int, reason: str = "", db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Rollback to a previous model version."""
    return {"status": "rolled_back", "model_id": model_id}


@router.get("/{model_id}")
async def get_model(model_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get model details."""
    return {
        "id": model_id,
        "name": f"Model {model_id}",
        "version": "1.0.0",
        "status": "draft",
        "benchmark_results": {},
        "forgetting_score": 0.0,
        "safety_score": 1.0
    }


@router.get("/compare")
async def compare_models(model_ids: List[int], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Compare multiple models."""
    return {
        "models": model_ids,
        "comparison": {}
    }


@router.get("/{model_id}/adapters")
async def list_model_adapters(model_id: int, db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List adapters for a model."""
    return []


@router.post("/{model_id}/adapters")
async def create_adapter(model_id: int, adapter_config: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Create a new adapter for a model."""
    return {"status": "created", "adapter_id": 1}
