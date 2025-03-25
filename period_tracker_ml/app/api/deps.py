from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import SessionLocal
from app.core.config import settings
from app.services.prediction_service import PredictionService
from app.services.model_service import ModelService
from app.services.training_service import TrainingService

async def get_db() -> Generator[AsyncSession, None, None]:
    """
    Get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=settings.API_KEY_NAME)
) -> bool:
    """
    Verify API key in request header.
    """
    if settings.API_KEY_REQUIRED:
        if not x_api_key or x_api_key != settings.API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": settings.API_KEY_NAME},
            )
    return True

async def get_current_user(
    api_key_valid: bool = Depends(verify_api_key),
    user_id: Optional[int] = Header(None, alias="X-User-ID")
) -> int:
    """
    Get and validate current user from request header.
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID is required",
        )
    # In a real-world scenario, we might validate if the user exists in the database
    return user_id

async def get_prediction_service(
    db: AsyncSession = Depends(get_db)
) -> PredictionService:
    """
    Get prediction service instance.
    """
    return PredictionService(db)

async def get_model_service(
    db: AsyncSession = Depends(get_db)
) -> ModelService:
    """
    Get model service instance.
    """
    return ModelService(db)

async def get_training_service(
    db: AsyncSession = Depends(get_db)
) -> TrainingService:
    """
    Get training service instance.
    """
    return TrainingService(db)
