"""
Database Configuration and Session Management
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.declarative import DeclarativeMeta
from app.core.config import settings
import os

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base: DeclarativeMeta = declarative_base()


async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Ensure directories exist
    os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.DATA_STORAGE_PATH, exist_ok=True)
    os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
    os.makedirs(settings.PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(settings.VALIDATED_DATA_PATH, exist_ok=True)
    os.makedirs(settings.QUARANTINE_DATA_PATH, exist_ok=True)
    os.makedirs(settings.REPLAY_BUFFER_PATH, exist_ok=True)
    
    print("Database tables created successfully")


async def close_db():
    """Close database connection"""
    await engine.dispose()
    print("Database connection closed")