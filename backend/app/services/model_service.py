"""
Model Service - Manages AI models and model registry
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime
import uuid
import logging

from app.models.model import Model, ModelStatus, ModelType
from app.models.model_version import ModelVersion
from app.models.dataset import Dataset
from app.models.training_session import TrainingSession
from app.models.evaluation import Evaluation, EvaluationType
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        query = select(Model)
        if model_type:
            query = query.where(Model.model_type == model_type)
        if status:
            query = query.where(Model.status == status)
        query = query.order_by(desc(Model.created_at))
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_model(self, model_id: str) -> Optional[Model]:
        query = select(Model).where(Model.id == model_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_model_by_name(self, name: str) -> Optional[Model]:
        query = select(Model).where(Model.name == name)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_model(self, model_data: Dict[str, Any], base_model_id: Optional[str] = None) -> Model:
        base_model = None
        if base_model_id:
            base_model = await self.get_model(base_model_id)
            if not base_model:
                raise ValueError(f"Base model not found: {base_model_id}")
        model_type = model_data.get("model_type", ModelType.BASE)
        if base_model and model_type == ModelType.BASE:
            model_type = ModelType.FINE_TUNED
        model = Model(
            id=str(uuid.uuid4()),
            name=model_data.get("name", f"Model-{uuid.uuid4().hex[:8]}"),
            description=model_data.get("description", ""),
            model_type=model_type,
            base_model_id=base_model_id,
            status=model_data.get("status", ModelStatus.DRAFT),
            config=model_data.get("config", {}),
            hyperparameters=model_data.get("hyperparameters", {}),
            parameter_count=model_data.get("parameter_count"),
            quantization=model_data.get("quantization", "none"),
            device=model_data.get("device", "cpu"),
            file_path=model_data.get("file_path"),
            file_size=model_data.get("file_size"),
            version=model_data.get("version", "1.0.0"),
        )
        self.db.add(model)
        await self.db.flush()
        await self.db.refresh(model)
        version = ModelVersion(
            id=str(uuid.uuid4()),
            model_id=model.id,
            version=model.version,
            changelog="Initial version",
            performance_scores=model_data.get("performance_scores", {}),
        )
        self.db.add(version)
        await self.db.flush()
        logger.info(f"Created model: {model.id} - {model.name}")
        return model
    
    async def update_model(self, model_id: str, update_data: Dict[str, Any]) -> Optional[Model]:
        model = await self.get_model(model_id)
        if not model:
            return None
        for key, value in update_data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        model.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(model)
        logger.info(f"Updated model: {model_id}")
        return model
    
    async def delete_model(self, model_id: str) -> Optional[Model]:
        model = await self.get_model(model_id)
        if not model:
            return None
        await self.db.delete(model)
        await self.db.flush()
        logger.info(f"Deleted model: {model_id}")
        return model
    
    async def get_production_model(self) -> Optional[Model]:
        query = select(Model).where(Model.status == ModelStatus.PRODUCTION)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def set_production_model(self, model_id: str) -> Optional[Model]:
        existing = await self.get_production_model()
        if existing and existing.id != model_id:
            existing.status = ModelStatus.APPROVED
            existing.updated_at = func.now()
            await self.db.flush()
        model = await self.get_model(model_id)
        if not model:
            return None
        model.status = ModelStatus.PRODUCTION
        model.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(model)
        logger.info(f"Set production model: {model_id}")
        return model
    
    async def promote_model(self, model_id: str, version: str = None) -> Optional[Model]:
        model = await self.get_model(model_id)
        if not model:
            return None
        if version:
            versions = await self.get_model_versions(model_id)
            parent_id = versions[-1].id if versions else None
            new_version = ModelVersion(
                id=str(uuid.uuid4()),
                model_id=model.id,
                version=version,
                changelog=f"Promoted to candidate",
                parent_version_id=parent_id,
            )
            self.db.add(new_version)
            model.version = version
            await self.db.flush()
        model.status = ModelStatus.CANDIDATE
        model.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(model)
        logger.info(f"Promoted model to candidate: {model_id}")
        return model
    
    async def approve_model(self, model_id: str) -> Optional[Model]:
        model = await self.get_model(model_id)
        if not model:
            return None
        if model.status != ModelStatus.CANDIDATE:
            raise ValueError(f"Model {model_id} is not a candidate")
        model.status = ModelStatus.APPROVED
        model.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(model)
        logger.info(f"Approved model: {model_id}")
        return model
    
    async def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        query = select(ModelVersion).where(ModelVersion.model_id == model_id)
        query = query.order_by(ModelVersion.created_at)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        model = await self.get_model(model_id)
        if not model:
            return {}
        versions = await self.get_model_versions(model_id)
        return {
            "model": model.to_dict(),
            "versions": [v.to_dict() for v in versions],
            "is_production": model.status == ModelStatus.PRODUCTION,
        }