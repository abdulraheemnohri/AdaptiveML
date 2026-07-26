"""
Dataset Service - Manages datasets and data processing
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime
import uuid
import logging

from app.models.dataset import Dataset, DatasetStatus
from app.models.data_sample import DataSample, DataQualityScore
from app.core.config import settings

logger = logging.getLogger(__name__)


class DatasetService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_datasets(self, status: Optional[DatasetStatus] = None, limit: int = 100, offset: int = 0) -> List[Dataset]:
        query = select(Dataset)
        if status:
            query = query.where(Dataset.status == status)
        query = query.order_by(desc(Dataset.created_at))
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        query = select(Dataset).where(Dataset.id == dataset_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_dataset(self, dataset_data: Dict[str, Any], source_id: Optional[str] = None) -> Dataset:
        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=dataset_data.get("name", f"Dataset-{uuid.uuid4().hex[:8]}"),
            description=dataset_data.get("description", ""),
            source_id=source_id,
            status=dataset_data.get("status", DatasetStatus.PENDING),
            file_path=dataset_data.get("file_path"),
            file_size=dataset_data.get("file_size"),
            num_samples=dataset_data.get("num_samples", 0),
            modality=dataset_data.get("modality", []),
            language=dataset_data.get("language"),
            metadata=dataset_data.get("metadata", {}),
            tags=dataset_data.get("tags", []),
            version=dataset_data.get("version", "1.0.0"),
        )
        self.db.add(dataset)
        await self.db.flush()
        await self.db.refresh(dataset)
        logger.info(f"Created dataset: {dataset.id} - {dataset.name}")
        return dataset
    
    async def update_dataset(self, dataset_id: str, update_data: Dict[str, Any]) -> Optional[Dataset]:
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return None
        for key, value in update_data.items():
            if hasattr(dataset, key):
                setattr(dataset, key, value)
        dataset.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(dataset)
        logger.info(f"Updated dataset: {dataset_id}")
        return dataset
    
    async def delete_dataset(self, dataset_id: str) -> Optional[Dataset]:
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            return None
        await self.db.delete(dataset)
        await self.db.flush()
        logger.info(f"Deleted dataset: {dataset_id}")
        return dataset
    
    async def list_samples(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[DataSample]:
        query = select(DataSample).where(DataSample.dataset_id == dataset_id)
        query = query.order_by(DataSample.created_at)
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_sample(self, sample_id: str) -> Optional[DataSample]:
        query = select(DataSample).where(DataSample.id == sample_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def add_sample(self, dataset_id: str, sample_data: Dict[str, Any]) -> DataSample:
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset not found: {dataset_id}")
        sample = DataSample(
            id=str(uuid.uuid4()),
            dataset_id=dataset_id,
            content=sample_data.get("content", ""),
            raw_content=sample_data.get("raw_content", ""),
            metadata=sample_data.get("metadata", {}),
            modality=sample_data.get("modality", []),
            language=sample_data.get("language"),
            quality_score=sample_data.get("quality_score", 0.0),
            relevance_score=sample_data.get("relevance_score", 0.0),
            confidence_score=sample_data.get("confidence_score", 0.0),
            safety_score=sample_data.get("safety_score", 0.0),
            trust_score=sample_data.get("trust_score", 0.0),
        )
        if sample.quality_score >= 90:
            sample.quality_category = DataQualityScore.EXCELLENT
        elif sample.quality_score >= 70:
            sample.quality_category = DataQualityScore.GOOD
        elif sample.quality_score >= 50:
            sample.quality_category = DataQualityScore.FAIR
        elif sample.quality_score >= 30:
            sample.quality_category = DataQualityScore.POOR
        else:
            sample.quality_category = DataQualityScore.BAD
        self.db.add(sample)
        await self.db.flush()
        await self.db.refresh(sample)
        dataset.num_samples += 1
        dataset.updated_at = func.now()
        await self.db.flush()
        logger.info(f"Added sample: {sample.id} to dataset {dataset_id}")
        return sample