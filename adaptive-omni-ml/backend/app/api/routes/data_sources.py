"""Data sources routes."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_data_sources():
    return {"sources": [], "total": 0}

@router.post("/")
async def add_data_source(source: dict):
    return {"message": "Data source added", "id": 1}

@router.put("/{source_id}")
async def update_data_source(source_id: int, source: dict):
    return {"message": f"Data source {source_id} updated"}

@router.delete("/{source_id}")
async def delete_data_source(source_id: int):
    return {"message": f"Data source {source_id} deleted"}

@router.post("/{source_id}/test")
async def test_data_source(source_id: int):
    return {"message": f"Data source {source_id} tested", "status": "ok"}

@router.post("/{source_id}/run")
async def run_data_source(source_id: int):
    return {"message": f"Data source {source_id} collection started"}
