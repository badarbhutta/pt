# app/schemas/prediction.py
"""Schema definitions for predictions and user feedback."""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class FertilityWindow(BaseModel):
    """Fertility window prediction."""
    start: str = Field(..., description="Start date of fertility window (ISO format)")
    end: str = Field(..., description="End date of fertility window (ISO format)")

class PeriodPrediction(BaseModel):
    """Period prediction results."""
    user_id: int = Field(..., description="User ID")
    next_period_start: str = Field(..., description="Predicted start date of next period (ISO format)")
    predicted_cycle_length: int = Field(..., description="Predicted cycle length in days")
    days_until_next_period: int = Field(..., description="Days until next period")
    fertility_window: FertilityWindow = Field(..., description="Predicted fertility window")
    ovulation_date: str = Field(..., description="Predicted ovulation date (ISO format)")
    confidence: float = Field(..., description="Confidence score for prediction (0-1)")
    model_contributions: Optional[Dict[str, float]] = Field(None, description="Contribution of each model to prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "next_period_start": "2023-11-05T00:00:00.000Z",
                "predicted_cycle_length": 28,
                "days_until_next_period": 12,
                "fertility_window": {
                    "start": "2023-10-18T00:00:00.000Z",
                    "end": "2023-10-23T00:00:00.000Z"
                },
                "ovulation_date": "2023-10-22T00:00:00.000Z",
                "confidence": 0.85,
                "model_contributions": {
                    "arima": 0.6,
                    "random_forest": 0.4
                }
            }
        }

class FertilityPrediction(BaseModel):
    """Fertility status prediction."""
    user_id: int = Field(..., description="User ID")
    current_cycle_day: int = Field(..., description="Current day of the cycle")
    fertility_window_start: str = Field(..., description="Start date of fertility window (ISO format)")
    fertility_window_end: str = Field(..., description="End date of fertility window (ISO format)")
    is_fertile_now: bool = Field(..., description="Whether user is currently in fertile window")
    ovulation_date: str = Field(..., description="Predicted ovulation date (ISO format)")
    confidence: float = Field(..., description="Confidence score for prediction (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "current_cycle_day": 12,
                "fertility_window_start": "2023-10-18T00:00:00.000Z",
                "fertility_window_end": "2023-10-23T00:00:00.000Z",
                "is_fertile_now": True,
                "ovulation_date": "2023-10-22T00:00:00.000Z",
                "confidence": 0.82
            }
        }

class OvulationPrediction(BaseModel):
    """Ovulation prediction."""
    user_id: int = Field(..., description="User ID")
    ovulation_date: str = Field(..., description="Predicted ovulation date (ISO format)")
    days_to_ovulation: int = Field(..., description="Days until ovulation")
    confidence: float = Field(..., description="Confidence score for prediction (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "ovulation_date": "2023-10-22T00:00:00.000Z",
                "days_to_ovulation": 7,
                "confidence": 0.78
            }
        }

class PredictionFeedback(BaseModel):
    """User feedback on prediction accuracy."""
    user_id: int = Field(..., description="User ID")
    prediction_id: Optional[str] = Field(None, description="ID of the prediction receiving feedback")
    prediction_type: str = Field(..., description="Type of prediction (period, fertility, ovulation)")
    predicted_date: str = Field(..., description="Predicted date (ISO format)")
    actual_date: str = Field(..., description="Actual date reported by user (ISO format)")
    accuracy_rating: int = Field(..., ge=1, le=5, description="User rating of accuracy (1-5)")
    comments: Optional[str] = Field(None, description="Additional user comments")
    submitted_at: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Feedback submission timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 1,
                "prediction_id": "pred_12345678",
                "prediction_type": "period",
                "predicted_date": "2023-10-15T00:00:00.000Z",
                "actual_date": "2023-10-16T00:00:00.000Z",
                "accuracy_rating": 4,
                "comments": "Almost perfect, just one day off",
                "submitted_at": "2023-10-16T10:30:00.000Z"
            }
        }