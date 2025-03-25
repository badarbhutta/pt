# app/api/api_v1/endpoints/predictions.py
"""API endpoints for period, fertility and ovulation predictions."""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from app.schemas.prediction import (
    PeriodPrediction,
    FertilityPrediction,
    OvulationPrediction,
    PredictionFeedback
)
from app.ml.model_implementations import PredictionService

router = APIRouter()
prediction_service = PredictionService()

# Sample cycle history for demo purposes
demo_cycle_history = [
    {
        "user_id": 1,
        "start_date": (datetime.now() - timedelta(days=60)).isoformat(),
        "end_date": (datetime.now() - timedelta(days=55)).isoformat(),
        "cycle_length": 28,
        "period_length": 5
    },
    {
        "user_id": 1,
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": (datetime.now() - timedelta(days=25)).isoformat(),
        "cycle_length": 30,
        "period_length": 5
    }
]

@router.get("/period/{user_id}", response_model=PeriodPrediction, summary="Predict Next Period")
async def predict_period(user_id: int):
    """
    Predict the next period start date, cycle length, and fertility window.
    
    - **user_id**: User ID to make predictions for
    
    Returns period prediction details including:
    - Next period start date
    - Predicted cycle length
    - Days until next period
    - Fertility window
    - Ovulation date
    - Confidence score
    """
    try:
        # In a real app, we would fetch cycle history from database
        # For demo, using sample data
        prediction = await prediction_service.predict_next_period(user_id, demo_cycle_history)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/fertility/{user_id}", response_model=FertilityPrediction, summary="Get Fertility Status")
async def predict_fertility(user_id: int):
    """
    Predict fertility status and fertile window.
    
    - **user_id**: User ID to make predictions for
    
    Returns fertility prediction details including:
    - Current cycle day
    - Fertility window start/end dates
    - Whether user is currently in fertile window
    - Ovulation date
    - Confidence score
    """
    try:
        # In a real app, we would fetch cycle history from database
        prediction = await prediction_service.predict_fertility(user_id, demo_cycle_history)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/ovulation/{user_id}", response_model=OvulationPrediction, summary="Predict Ovulation")
async def predict_ovulation(user_id: int):
    """
    Predict ovulation date and days until ovulation.
    
    - **user_id**: User ID to make predictions for
    
    Returns ovulation prediction details including:
    - Ovulation date
    - Days until ovulation
    - Confidence score
    """
    try:
        # In a real app, we would fetch cycle history from database
        prediction = await prediction_service.predict_ovulation(user_id, demo_cycle_history)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/feedback", summary="Submit Prediction Feedback")
async def submit_feedback(feedback: PredictionFeedback):
    """
    Submit feedback on prediction accuracy.
    
    - **feedback**: Feedback data including actual dates and accuracy assessment
    
    This feedback will be used to improve future predictions.
    """
    try:
        result = await prediction_service.add_user_feedback(feedback.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")