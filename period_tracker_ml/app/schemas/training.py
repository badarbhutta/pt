from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    PERIOD_PREDICTION = "period_prediction"
    FERTILITY_PREDICTION = "fertility_prediction"
    SYMPTOM_PREDICTION = "symptom_prediction"

class TrainingStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"

class TrainingRequest(BaseModel):
    """
    Request model for training a user-specific model.
    """
    features: List[str] = Field(..., description="Features to include in the model")
    model_type: ModelType = Field(ModelType.PERIOD_PREDICTION, description="Type of model to train")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters for the model")

class TrainingResponse(BaseModel):
    """
    Response model for training requests.
    """
    success: bool = Field(..., description="Whether the training request was accepted")
    message: str = Field(..., description="Informational message about the training request")
    training_id: Optional[str] = Field(None, description="Unique identifier for the training job")
    status: TrainingStatus = Field(..., description="Current status of the training job")

class RetrainingRequest(BaseModel):
    """
    Request model for retraining global models (admin only).
    """
    model_type: ModelType = Field(..., description="Type of model to retrain")
    features: List[str] = Field(..., description="Features to include in the model")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Custom hyperparameters for the model")
    training_data_percentage: Optional[float] = Field(80.0, description="Percentage of data to use for training vs validation")

class RetrainingResponse(BaseModel):
    """
    Response model for global model retraining requests.
    """
    success: bool = Field(..., description="Whether the retraining request was accepted")
    message: str = Field(..., description="Informational message about the retraining request")
    training_id: Optional[str] = Field(None, description="Unique identifier for the training job")
    status: TrainingStatus = Field(..., description="Current status of the training job")

class TrainingStatusResponse(BaseModel):
    """
    Response model for training status checks.
    """
    training_id: str = Field(..., description="Unique identifier for the training job")
    status: TrainingStatus = Field(..., description="Current status of the training job")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    start_time: Optional[datetime] = Field(None, description="When training started")
    end_time: Optional[datetime] = Field(None, description="When training completed or failed")
    model_id: Optional[str] = Field(None, description="ID of the resulting model (if completed)")
    model_version: Optional[str] = Field(None, description="Version of the resulting model (if completed)")
    metrics: Optional[Dict[str, float]] = Field(None, description="Training metrics (if completed)")
    error_message: Optional[str] = Field(None, description="Error message (if failed)")
