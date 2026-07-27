"""
Training Service - Manages model training and continual learning
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime
import uuid
import logging

from app.models.model import Model, ModelStatus, ModelType
from app.models.training_session import TrainingSession, TrainingStatus
from app.models.dataset import Dataset, DatasetStatus
from app.models.evaluation import Evaluation, EvaluationType
from app.core.config import settings

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_sessions(
        self,
        model_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingSession]:
        query = select(TrainingSession)
        if model_id:
            query = query.where(TrainingSession.model_id == model_id)
        if status:
            query = query.where(TrainingSession.status == status)
        query = query.order_by(desc(TrainingSession.created_at))
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_session(self, session_id: str) -> Optional[TrainingSession]:
        query = select(TrainingSession).where(TrainingSession.id == session_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_session(self, model_id: str, session_data: Dict[str, Any]) -> TrainingSession:
        model = await self._get_model(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")
        training_dataset_id = session_data.get("training_dataset_id")
        validation_dataset_id = session_data.get("validation_dataset_id")
        session = TrainingSession(
            id=str(uuid.uuid4()),
            model_id=model_id,
            name=session_data.get("name", f"Training-{uuid.uuid4().hex[:8]}"),
            description=session_data.get("description", ""),
            training_dataset_id=training_dataset_id,
            validation_dataset_id=validation_dataset_id,
            status=TrainingStatus.PENDING,
            config=session_data.get("config", {}),
            hyperparameters=session_data.get("hyperparameters", {}),
            epochs=session_data.get("epochs", settings.TRAINING_EPOCHS),
            batch_size=session_data.get("batch_size", settings.TRAINING_BATCH_SIZE),
            learning_rate=session_data.get("learning_rate", settings.LEARNING_RATE),
            replay_ratio=session_data.get("replay_ratio", settings.REPLAY_RATIO),
            ewc_strength=session_data.get("ewc_strength", settings.EWC_STRENGTH),
            distillation_strength=session_data.get("distillation_strength", settings.DISTILLATION_STRENGTH),
        )
        self.db.add(session)
        await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Created training session: {session.id} - {session.name}")
        return session
    
    async def start_session(self, session_id: str) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        if session.status != TrainingStatus.PENDING:
            raise ValueError(f"Session {session_id} is not pending")
        session.status = TrainingStatus.PREPARING
        session.start_time = func.now()
        session.updated_at = func.now()
        await self.db.flush()
        model = await self._get_model(session.model_id)
        if model:
            model.status = ModelStatus.TRAINING
            model.updated_at = func.now()
            await self.db.flush()
        logger.info(f"Started training session: {session_id}")
        session.status = TrainingStatus.TRAINING
        await self.db.flush()
        return session
    
    async def pause_session(self, session_id: str) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        if session.status not in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            raise ValueError(f"Cannot pause session {session_id} in status {session.status.value}")
        session.status = TrainingStatus.PAUSED
        session.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Paused training session: {session_id}")
        return session
    
    async def resume_session(self, session_id: str) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        if session.status != TrainingStatus.PAUSED:
            raise ValueError(f"Cannot resume session {session_id} in status {session.status.value}")
        session.status = TrainingStatus.TRAINING
        session.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Resumed training session: {session_id}")
        return session
    
    async def stop_session(self, session_id: str) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        if session.status == TrainingStatus.COMPLETED:
            return session
        session.status = TrainingStatus.STOPPED
        session.end_time = func.now()
        session.duration = (session.end_time - session.start_time).total_seconds() if session.start_time else 0
        session.updated_at = func.now()
        await self.db.flush()
        model = await self._get_model(session.model_id)
        if model and model.status == ModelStatus.TRAINING:
            model.status = ModelStatus.DRAFT
            model.updated_at = func.now()
            await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Stopped training session: {session_id}")
        return session
    
    async def update_session_progress(
        self,
        session_id: str,
        current_epoch: int,
        current_step: int,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None
    ) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        session.current_epoch = current_epoch
        session.current_step = current_step
        if loss is not None:
            session.loss = loss
        if val_loss is not None:
            session.val_loss = val_loss
            if session.best_score is None or val_loss < session.best_score:
                session.best_score = val_loss
                session.best_epoch = current_epoch
        session.updated_at = func.now()
        await self.db.flush()
        await self.db.refresh(session)
        return session
    
    async def complete_session(self, session_id: str, final_scores: Optional[Dict[str, float]] = None) -> Optional[TrainingSession]:
        session = await self.get_session(session_id)
        if not session:
            return None
        session.status = TrainingStatus.COMPLETED
        session.end_time = func.now()
        session.duration = (session.end_time - session.start_time).total_seconds() if session.start_time else 0
        session.updated_at = func.now()
        model = await self._get_model(session.model_id)
        if model:
            model.training_dataset_id = session.training_dataset_id
            model.training_completed_at = session.end_time
            model.training_duration = session.duration
            model.status = ModelStatus.TESTING
            model.updated_at = func.now()
            if final_scores:
                for key, value in final_scores.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
            await self.db.flush()
        await self.db.flush()
        await self.db.refresh(session)
        logger.info(f"Completed training session: {session_id}")
        return session
    
    async def _get_model(self, model_id: str) -> Optional[Model]:
        query = select(Model).where(Model.id == model_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        query = select(Dataset).where(Dataset.id == dataset_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()