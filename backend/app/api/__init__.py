"""
API Routers for Adaptive Omni ML Platform
"""

from fastapi import APIRouter

# Create main router
api_router = APIRouter()

# Include sub-routers
# api_router.include_router(data.router, prefix="/data", tags=["Data"])