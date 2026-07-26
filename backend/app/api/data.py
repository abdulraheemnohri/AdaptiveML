"""
Data Collection and Management API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional

router = APIRouter()


@router.get("/sources")
async def list_data_sources():
    """List all data sources"""
    return [{"id": "1", "name": "Test Source", "type": "web"}]


@router.get("/sources/{source_id}")
async def get_data_source(source_id: str):
    """Get a specific data source"""
    return {"id": source_id, "name": "Test Source", "type": "web"}