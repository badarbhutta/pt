from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any

from app.schemas.training import (
    TrainingRequest, TrainingResponse, 
    RetrainingRequest, RetrainingResponse
)
from app.services.training_service import TrainingService
from app.api.deps import get_current_user, get_training_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/train-user-model", response_model=TrainingResponse)
async def train_user_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """
    Train a user-specific model based on their cycle history and features.
    """
    try:
        logger.info(f"Starting user model training for user {user_id}")
        # Validate if user has enough data for training
        if not await training_service.has_sufficient_data(user_id):
            return TrainingResponse(
                success=False,
                message="Insufficient data for training user model. Need at least 6 complete cycles.",
                training_id=None,
                status="rejected"
            )
        
        # Schedule training in background
        training_id = await training_service.schedule_user_model_training(
            user_id=user_id,
            features=request.features,
            background_tasks=background_tasks
        )
        
        return TrainingResponse(
            success=True,
            message="User model training scheduled successfully",
            training_id=training_id,
            status="scheduled"
        )
    except Exception as e:
        logger.error(f"Error scheduling user model training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule user model training: {str(e)}"
        )

@router.post("/retrain-global-model", response_model=RetrainingResponse)
async def retrain_global_model(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks,
    user_id: int = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """
    Retrain the global model with new data (admin only).
    """
    try:
        # Check if user has admin privileges
        if not await training_service.is_admin(user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only administrators can retrain global models"
            )
        
        logger.info(f"Starting global model retraining by admin user {user_id}")
        training_id = await training_service.schedule_global_model_retraining(
            features=request.features,
            model_type=request.model_type,
            background_tasks=background_tasks
        )
        
        return RetrainingResponse(
            success=True,
            message="Global model retraining scheduled successfully",
            training_id=training_id,
            status="scheduled"
        )
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error scheduling global model retraining: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule global model retraining: {str(e)}"
        )

@router.get("/status/{training_id}", response_model=Dict[str, Any])
async def get_training_status(
    training_id: str,
    user_id: int = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service)
):
    """
    Get the status of a training job.
    """
    try:
        logger.info(f"Getting training status for job {training_id} requested by user {user_id}")
        status_info = await training_service.get_training_status(training_id, user_id)
        return status_info
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )
