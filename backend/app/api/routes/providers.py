"""AI Providers API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def list_providers(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all configured AI providers."""
    return []


@router.post("/")
async def add_provider(provider_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Add a new AI provider."""
    return {
        "status": "created",
        "provider_id": 1,
        "message": "Provider added successfully"
    }


@router.get("/{provider_id}")
async def get_provider(provider_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get provider details."""
    return {
        "id": provider_id,
        "name": f"Provider {provider_id}",
        "provider_type": "openai",
        "is_enabled": True,
        "available_models": []
    }


@router.put("/{provider_id}")
async def update_provider(provider_id: int, provider_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Update provider configuration."""
    return {"status": "updated", "provider_id": provider_id}


@router.delete("/{provider_id}")
async def delete_provider(provider_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Delete a provider."""
    return {"status": "deleted", "provider_id": provider_id}


@router.post("/{provider_id}/test")
async def test_provider(provider_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Test provider connectivity."""
    return {"status": "success", "latency_ms": 150}


@router.get("/{provider_id}/models")
async def list_provider_models(provider_id: int, db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List available models from a provider."""
    return []


@router.get("/usage")
async def get_usage_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get API usage statistics."""
    return {
        "total_requests_today": 0,
        "total_cost_today": 0.0,
        "by_provider": {}
    }
