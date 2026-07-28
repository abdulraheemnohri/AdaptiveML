"""Modes routes - Training/Serving mode switching."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_modes():
    return {"modes": ["training", "serving"], "current_mode": "serving"}

@router.post("/switch")
async def switch_mode(mode: str):
    return {"message": f"Switched to {mode} mode", "mode": mode}
