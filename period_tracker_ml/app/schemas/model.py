from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import date, datetime

from app.schemas.prediction import CycleData

class ModelInfoResponse(BaseModel):
    """
    Response model for model information.
    """
    model_id: str = Field(..., description="Unique identifier for the model")
    model_type: str = Field(..., description="Type of model (global, user-specific)")
    features: List[str] = Field(..., description="Features used by the model")
    target_metric: str = Field(..., description="Primary metric the model is optimized for")
    current_version: str = Field(..., description="Current active version of the model")
    created_at: datetime = Field(..., description="When the model was initially created")
    last_updated: datetime = Field(..., description="When the model was last updated")
    accuracy: Optional[float] = Field(None, description="Overall accuracy of the model")
    average_error_days: Optional[float] = Field(None, description="Average error in days")
    is_user_specific: bool = Field(..., description="Whether this is a user-specific model")
    available_versions: Optional[int] = Field(None, description="Number of available versions")

class ModelVersionResponse(BaseModel):
    """
    Response model for model version information.
    """
    version: str = Field(..., description="Version identifier")
    model_id: str = Field(..., description="ID of the parent model")
    created_at: datetime = Field(..., description="When this version was created")
    is_active: bool = Field(..., description="Whether this version is currently active")
    metrics: Dict[str, float] = Field(..., description="Performance metrics for this version")
    training_data_size: int = Field(..., description="Number of samples used for training")
    features: List[str] = Field(..., description="Features used in this version")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters/hyperparameters")
    artifact_uri: Optional[str] = Field(None, description="URI to model artifacts in MLflow")

class ModelEvaluationRequest(BaseModel):
    """
    Request model for evaluating a model on user data.
    """
    model_id: str = Field(..., description="ID of the model to evaluate")
    cycle_data: CycleData = Field(..., description="User cycle data to evaluate against")
    version: Optional[str] = Field(None, description="Specific version to evaluate (defaults to current active)")

class ModelEvaluationResponse(BaseModel):
    """
    Response model for model evaluation results.
    """
    model_id: str = Field(..., description="ID of the evaluated model")
    version: str = Field(..., description="Version that was evaluated")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    average_error_days: float = Field(..., description="Average error in days")
    is_recommended: bool = Field(..., description="Whether this model is recommended for this user")
    evaluation_timestamp: datetime = Field(default_factory=datetime.now, description="When evaluation was performed")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional evaluation details")
