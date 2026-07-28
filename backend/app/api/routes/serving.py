"""Serving API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def get_serving_status(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get current serving status."""
    return {
        "status": "ready",
        "mode": "local_first",
        "active_engine": "local",
        "model_loaded": False,
        "providers_available": 0
    }


@router.get("/config")
async def get_serving_config(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get serving configuration."""
    return {
        "default_mode": "local_first",
        "local_model_path": "./models/production",
        "max_context_length": 8192,
        "quantization": "int4"
    }


@router.post("/mode")
async def set_serving_mode(mode_config: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Set serving mode."""
    return {"status": "updated", "mode": mode_config.get("mode")}


@router.post("/local/load")
async def load_local_model(model_config: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Load local model for inference."""
    return {"status": "loading", "model_id": model_config.get("model_id")}


@router.post("/local/unload")
async def unload_local_model(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Unload local model."""
    return {"status": "unloaded"}


@router.get("/local/status")
async def get_local_model_status(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get local model status."""
    return {
        "loaded": False,
        "model_id": None,
        "vram_usage_mb": 0,
        "context_length": 0
    }


@router.post("/infer")
async def run_inference(request_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Run inference using configured serving engine."""
    return {
        "response": "Inference result",
        "target": "local",
        "latency_ms": 100,
        "tokens_used": 50
    }


@router.post("/chat")
async def chat_completion(chat_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Chat completion endpoint."""
    return {
        "id": "chat-1",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }
