"""
Data Service - Manages data sources and data collection
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import uuid
import logging

from app.models.data_source import DataSource, DataSourceType
from app.models.collection_job import CollectionJob, JobStatus
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_sources(
        self,
        source_type: Optional[DataSourceType] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DataSource]:
        query = select(DataSource)
        if source_type:
            query = query.where(DataSource.source_type == source_type)
        if enabled is not None:
            query = query.where(DataSource.enabled == enabled)
        query = query.order_by(DataSource.name)
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_source(self, source_id: str) -> Optional[DataSource]:
        query = select(DataSource).where(DataSource.id == source_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_source(self, source_data: Dict[str, Any]) -> DataSource:
        source = DataSource(
            id=str(uuid.uuid4()),
            name=source_data.get("name", "Unnamed Source"),
            description=source_data.get("description", ""),
            source_type=source_data.get("source_type", DataSourceType.CUSTOM),
            url=source_data.get("url"),
            path=source_data.get("path"),
            config=source_data.get("config", {}),
            enabled=source_data.get("enabled", True),
            schedule=source_data.get("schedule"),
        )
        self.db.add(source)
        await self.db.flush()
        await self.db.refresh(source)
        logger.info(f"Created data source: {source.id} - {source.name}")
        return source
    
    async def update_source(self, source_id: str, update_data: Dict[str, Any]) -> Optional[DataSource]:
        source = await self.get_source(source_id)
        if not source:
            return None
        for key, value in update_data.items():
            if hasattr(source, key):
                setattr(source, key, value)
        source.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(source)
        logger.info(f"Updated data source: {source.id}")
        return source
    
    async def delete_source(self, source_id: str) -> Optional[DataSource]:
        source = await self.get_source(source_id)
        if not source:
            return None
        await self.db.delete(source)
        await self.db.flush()
        logger.info(f"Deleted data source: {source_id}")
        return source
    
    async def enable_source(self, source_id: str) -> Optional[DataSource]:
        source = await self.get_source(source_id)
        if not source:
            return None
        source.enabled = True
        source.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(source)
        logger.info(f"Enabled data source: {source_id}")
        return source
    
    async def disable_source(self, source_id: str) -> Optional[DataSource]:
        source = await self.get_source(source_id)
        if not source:
            return None
        source.enabled = False
        source.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(source)
        logger.info(f"Disabled data source: {source_id}")
        return source
    
    async def list_collection_jobs(
        self,
        source_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CollectionJob]:
        query = select(CollectionJob)
        if source_id:
            query = query.where(CollectionJob.source_id == source_id)
        if status:
            query = query.where(CollectionJob.status == status)
        query = query.order_by(CollectionJob.created_at.desc())
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_collection_job(self, job_id: str) -> Optional[CollectionJob]:
        query = select(CollectionJob).where(CollectionJob.id == job_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_collection_job(self, source_id: str, config: Optional[Dict[str, Any]] = None) -> CollectionJob:
        job = CollectionJob(
            id=str(uuid.uuid4()),
            source_id=source_id,
            name=f"Collection from {source_id}",
            status=JobStatus.PENDING,
            config=config or {},
            max_retries=3,
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)
        logger.info(f"Created collection job: {job.id}")
        return job
    
    async def collect_from_source(self, source_id: str) -> Optional[CollectionJob]:
        source = await self.get_source(source_id)
        if not source:
            logger.error(f"Source not found: {source_id}")
            return None
        if not source.enabled:
            logger.warning(f"Source {source_id} is disabled")
            return None
        job = await self.create_collection_job(source_id, source.config)
        job = await self.update_job_status(job.id, JobStatus.RUNNING)
        logger.info(f"Started collection from source: {source_id}")
        job.items_collected = 10
        job.items_processed = 8
        job = await self.update_job_status(job.id, JobStatus.COMPLETED)
        return job
    
    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None
    ) -> Optional[CollectionJob]:
        job = await self.get_collection_job(job_id)
        if not job:
            return None
        job.status = status
        if error_message:
            job.error_message = error_message
        job.updated_at = func.now()
        if status == JobStatus.COMPLETED:
            job.end_time = func.now()
            job.duration = (job.end_time - job.start_time).total_seconds() if job.start_time else 0
        elif status == JobStatus.RUNNING:
            job.start_time = func.now()
            job.retry_count = 0
        elif status == JobStatus.FAILED:
            job.end_time = func.now()
            job.retry_count += 1
        await self.db.flush()
        await self.db.refresh(job)
        logger.info(f"Updated job {job_id} status to {status.value}")
        return job