"""AI Router API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def get_router_status(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get AI router status."""
    return {
        "routing_mode": "automatic",
        "local_available": False,
        "providers_registered": 0,
        "active_rules": 0
    }


@router.get("/config")
async def get_router_config(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get router configuration."""
    return {
        "default_mode": "automatic",
        "rules": [],
        "provider_priorities": {}
    }


@router.post("/mode")
async def set_routing_mode(mode_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Set routing mode."""
    return {"status": "updated", "mode": mode_data.get("mode")}


@router.get("/rules")
async def list_routing_rules(db: AsyncSession = Depends(get_db)) -> List[Dict[str, Any]]:
    """List all routing rules."""
    return []


@router.post("/rules")
async def create_routing_rule(rule_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Create a new routing rule."""
    return {"status": "created", "rule_id": 1}


@router.put("/rules/{rule_id}")
async def update_routing_rule(rule_id: int, rule_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Update a routing rule."""
    return {"status": "updated", "rule_id": rule_id}


@router.delete("/rules/{rule_id}")
async def delete_routing_rule(rule_id: int, db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Delete a routing rule."""
    return {"status": "deleted", "rule_id": rule_id}


@router.post("/test")
async def test_routing(test_request: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Test routing decision for a sample request."""
    return {
        "selected_target": "local",
        "target_type": "local",
        "reasoning": "Automatic routing: local processing; low latency",
        "estimated_latency_ms": 100,
        "estimated_cost_usd": 0.0
    }


@router.get("/stats")
async def get_routing_stats(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get routing statistics."""
    return {
        "total_requests": 0,
        "local_requests": 0,
        "api_requests": 0,
        "fallbacks": 0,
        "avg_latency_ms": 0.0
    }
