# app/api/api_v1/endpoints/training.py
"""API endpoints for model training."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional

from app.schemas.ml import (
    TrainingRequest,
    TrainingResponse,
    ModelInfo,
    ModelsList,
    ModelMetrics
)
from app.ml.training import TrainingService

router = APIRouter()
training_service = TrainingService()

@router.post("/start", response_model=TrainingResponse, summary="Start Model Training")
async def start_training(
    training_request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start training a new model.
    
    - **model_type**: Type of model to train (arima, random_forest, gradient_boosting, ensemble)
    - **training_parameters**: Optional parameters for training
    - **user_id**: Optional user ID for user-specific models
    
    Returns a training job ID that can be used to check training status.
    """
    try:
        training_id = await training_service.start_training(
            model_type=training_request.model_type,
            training_params=training_request.training_parameters,
            user_id=training_request.user_id,
            background_tasks=background_tasks
        )
        
        # Get initial status
        status = await training_service.get_training_status(training_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/status/{training_id}", response_model=TrainingResponse, summary="Check Training Status")
async def get_training_status(training_id: str):
    """
    Check the status of a training job.
    
    - **training_id**: Training job ID
    
    Returns the current status of the training job including progress and any results.
    """
    try:
        status = await training_service.get_training_status(training_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking training status: {str(e)}")

@router.get("/models", response_model=ModelsList, summary="List Available Models")
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    active_only: bool = Query(False, description="Only show active models")
):
    """
    List available trained models.
    
    - **model_type**: Optional filter by model type
    - **user_id**: Optional filter by user ID
    - **active_only**: Only show currently active models
    
    Returns a list of available models with their basic information.
    """
    try:
        models = await training_service.list_models(
            model_type=model_type,
            user_id=user_id,
            active_only=active_only
        )
        
        # Count models by type
        global_models = len([m for m in models if m.get("user_id") is None])
        user_models = len(models) - global_models
        
        return {
            "models": models,
            "count": len(models),
            "global_models": global_models,
            "user_models": user_models
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.get("/models/{model_id}", response_model=ModelInfo, summary="Get Model Information")
async def get_model_info(model_id: str):
    """
    Get detailed information about a model.
    
    - **model_id**: Model ID
    
    Returns detailed information about the specified model.
    """
    try:
        model_info = await training_service.get_model_info(model_id)
        return model_info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@router.get("/models/{model_id}/metrics", response_model=ModelMetrics, summary="Get Model Metrics")
async def get_model_metrics(model_id: str):
    """
    Get performance metrics for a model.
    
    - **model_id**: Model ID
    
    Returns performance metrics for the specified model.
    """
    try:
        metrics = await training_service.get_model_metrics(model_id)
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model metrics: {str(e)}")

@router.post("/models/{model_id}/activate", summary="Activate Model")
async def activate_model(model_id: str):
    """
    Activate a model for predictions.
    
    - **model_id**: Model ID
    
    Activates the specified model for use in predictions.
    """
    try:
        success = await training_service.activate_model(model_id)
        if success:
            return {"status": "success", "message": f"Model {model_id} activated successfully"}
        else:
            return {"status": "error", "message": "Failed to activate model"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error activating model: {str(e)}")

@router.get("/models/{model_id}/feature-importance", summary="Get Feature Importance")
async def get_feature_importance(model_id: str):
    """
    Get feature importance for a model.
    
    - **model_id**: Model ID
    
    Returns feature importance values for the specified model.
    """
    try:
        importance = await training_service.get_feature_importance(model_id)
        return importance
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feature importance: {str(e)}")