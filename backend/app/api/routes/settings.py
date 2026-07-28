"""Settings API routes."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from backend.app.database.session import get_db

router = APIRouter()


@router.get("/")
async def get_all_settings(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get all system settings."""
    return {
        "training": {
            "batch_size": 4,
            "learning_rate": 2e-4,
            "epochs": 3,
            "gradient_accumulation_steps": 4
        },
        "serving": {
            "default_mode": "local_first",
            "max_context_length": 8192,
            "quantization": "int4"
        },
        "continual_learning": {
            "replay_buffer_size": 10000,
            "replay_ratio": 0.2,
            "distillation_weight": 0.5,
            "ewc_strength": 1000.0
        },
        "anti_forgetting": {
            "forgetting_threshold": 0.02,
            "regression_threshold": 0.01,
            "quality_gate_threshold": 0.90,
            "safety_gate_threshold": 0.95
        },
        "privacy": {
            "local_only_mode": False,
            "allow_api_file_upload": True,
            "allow_api_image_upload": True,
            "allow_api_history_send": False
        }
    }


@router.put("/{category}")
async def update_settings(category: str, settings_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Update settings for a specific category."""
    valid_categories = ["training", "serving", "continual_learning", "anti_forgetting", "privacy"]
    
    if category not in valid_categories:
        raise HTTPException(status_code=400, detail=f"Invalid category. Valid categories: {valid_categories}")
    
    return {"status": "updated", "category": category}


@router.get("/continual-learning")
async def get_continual_learning_settings(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get continual learning specific settings."""
    return {
        "replay_buffer_size": 10000,
        "replay_ratio": 0.2,
        "prioritized_replay": True,
        "distillation_enabled": True,
        "distillation_temperature": 2.0,
        "distillation_weight": 0.5,
        "ewc_enabled": True,
        "ewc_strength": 1000.0,
        "lwf_enabled": False
    }


@router.get("/anti-forgetting")
async def get_anti_forgetting_settings(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Get anti-forgetting gate settings."""
    return {
        "forgetting_threshold": 0.02,
        "regression_threshold": 0.01,
        "quality_gate_threshold": 0.90,
        "safety_gate_threshold": 0.95,
        "protected_capabilities": [
            "reasoning",
            "mathematics",
            "coding",
            "general_knowledge",
            "language_en",
            "language_urdu",
            "vision",
            "audio",
            "speech",
            "safety"
        ]
    }


@router.post("/reset")
async def reset_settings(reset_data: Dict[str, Any], db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Reset settings to defaults."""
    return {"status": "reset", "message": "Settings reset to defaults"}
