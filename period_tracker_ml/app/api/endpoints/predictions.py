from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService
from app.api.deps import get_current_user, get_prediction_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.post("/predict-next-period", response_model=PredictionResponse)
async def predict_next_period(
    request: PredictionRequest,
    user_id: int = Depends(get_current_user),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict next period date based on user's cycle history and relevant features.
    """
    try:
        logger.info(f"Predicting next period for user {user_id}")
        result = await prediction_service.predict_next_period(
            user_id=user_id,
            cycle_data=request.cycle_data,
            symptoms=request.symptoms,
            lifestyle_factors=request.lifestyle_factors
        )
        return PredictionResponse(
            predicted_date=result.predicted_date,
            confidence=result.confidence,
            prediction_window=result.prediction_window,
            features_importance=result.features_importance,
            model_version=result.model_version
        )
    except Exception as e:
        logger.error(f"Error predicting next period: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict next period: {str(e)}"
        )

@router.post("/predict-fertility-window", response_model=PredictionResponse)
async def predict_fertility_window(
    request: PredictionRequest,
    user_id: int = Depends(get_current_user),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict fertility window based on user's cycle history and relevant features.
    """
    try:
        logger.info(f"Predicting fertility window for user {user_id}")
        result = await prediction_service.predict_fertility_window(
            user_id=user_id,
            cycle_data=request.cycle_data,
            symptoms=request.symptoms,
            lifestyle_factors=request.lifestyle_factors
        )
        return PredictionResponse(
            predicted_date=result.predicted_date,
            fertility_window_start=result.fertility_window_start,
            fertility_window_end=result.fertility_window_end,
            confidence=result.confidence,
            prediction_window=result.prediction_window,
            features_importance=result.features_importance,
            model_version=result.model_version
        )
    except Exception as e:
        logger.error(f"Error predicting fertility window: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict fertility window: {str(e)}"
        )
