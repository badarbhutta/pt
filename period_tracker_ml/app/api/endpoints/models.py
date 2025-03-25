from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Dict, Any

from app.schemas.model import (
    ModelInfoResponse, ModelVersionResponse, 
    ModelEvaluationRequest, ModelEvaluationResponse
)
from app.services.model_service import ModelService
from app.api.deps import get_current_user, get_model_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/info", response_model=List[ModelInfoResponse])
async def get_models_info(
    user_id: int = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get information about available models including global and user-specific models.
    """
    try:
        logger.info(f"Getting model info for user {user_id}")
        models_info = await model_service.get_models_info(user_id)
        return models_info
    except Exception as e:
        logger.error(f"Error getting models info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models information: {str(e)}"
        )

@router.get("/versions/{model_id}", response_model=List[ModelVersionResponse])
async def get_model_versions(
    model_id: str,
    user_id: int = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Get available versions for a specific model.
    """
    try:
        logger.info(f"Getting versions for model {model_id}")
        versions = await model_service.get_model_versions(model_id, user_id)
        return versions
    except Exception as e:
        logger.error(f"Error getting model versions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model versions: {str(e)}"
        )

@router.post("/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model(
    request: ModelEvaluationRequest,
    user_id: int = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Evaluate a model's performance on user data.
    """
    try:
        logger.info(f"Evaluating model {request.model_id} for user {user_id}")
        evaluation = await model_service.evaluate_model(
            model_id=request.model_id,
            user_id=user_id,
            cycle_data=request.cycle_data,
            version=request.version
        )
        return evaluation
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate model: {str(e)}"
        )

@router.post("/promote/{model_id}/{version}", response_model=Dict[str, Any])
async def promote_model_version(
    model_id: str,
    version: str,
    user_id: int = Depends(get_current_user),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Promote a specific model version to production.
    """
    try:
        logger.info(f"Promoting model {model_id} version {version} to production")
        result = await model_service.promote_model_version(model_id, version, user_id)
        return {"success": True, "message": f"Model {model_id} version {version} promoted to production", "details": result}
    except Exception as e:
        logger.error(f"Error promoting model version: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to promote model version: {str(e)}"
        )
