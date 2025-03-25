# app/schemas/ml.py
"""Schema definitions for machine learning models and training."""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class TrainingParameters(BaseModel):
    """Parameters for training a model."""
    # ARIMA parameters
    p: Optional[int] = Field(None, description="AR order for ARIMA models")
    d: Optional[int] = Field(None, description="Differencing order for ARIMA models")
    q: Optional[int] = Field(None, description="MA order for ARIMA models")
    
    # Tree-based model parameters
    n_estimators: Optional[int] = Field(None, description="Number of estimators for ensemble models")
    max_depth: Optional[int] = Field(None, description="Maximum depth for tree-based models")
    learning_rate: Optional[float] = Field(None, description="Learning rate for boosting models")
    
    # General parameters
    test_size: Optional[float] = Field(0.2, description="Test size for train-test split")
    random_state: Optional[int] = Field(42, description="Random state for reproducibility")
    
    class Config:
        json_schema_extra = {
            "example": {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1
            }
        }

class TrainingRequest(BaseModel):
    """Request to start model training."""
    model_type: str = Field(..., description="Type of model to train (arima, random_forest, gradient_boosting, ensemble)")
    training_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters for training")
    user_id: Optional[int] = Field(None, description="User ID for user-specific models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "random_forest",
                "training_parameters": {
                    "n_estimators": 100,
                    "max_depth": 5
                },
                "user_id": 1
            }
        }

class TrainingResponse(BaseModel):
    """Response from training operations."""
    training_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status (pending, running, completed, failed)")
    progress: int = Field(..., description="Training progress (0-100)")
    message: str = Field(..., description="Status message")
    model_id: Optional[str] = Field(None, description="ID of the trained model (if completed)")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last status update timestamp")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Model performance metrics (if completed)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "training_id": "training_12345678",
                "status": "completed",
                "progress": 100,
                "message": "Model trained successfully",
                "model_id": "model_87654321",
                "created_at": "2023-10-25T14:30:00.000Z",
                "updated_at": "2023-10-25T14:32:00.000Z",
                "metrics": {
                    "mae": 1.2,
                    "rmse": 1.5,
                    "r2": 0.85
                }
            }
        }

class ModelFeatureImportance(BaseModel):
    """Feature importance for a model."""
    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_12345678",
                "model_type": "RandomForestModel",
                "feature_importance": {
                    "previous_cycle_length": 0.6,
                    "user_age": 0.2,
                    "average_cycle_length": 0.15,
                    "stress_level": 0.05
                }
            }
        }

class ModelMetrics(BaseModel):
    """Metrics for a model."""
    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    created_at: str = Field(..., description="Model creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_12345678",
                "model_type": "RandomForestModel",
                "metrics": {
                    "mae": 1.2,
                    "rmse": 1.5,
                    "r2": 0.85,
                    "within_1_day_percent": 78.5,
                    "within_2_days_percent": 92.3
                },
                "created_at": "2023-10-25T14:30:00.000Z"
            }
        }

class ModelInfo(BaseModel):
    """Detailed information about a model."""
    model_id: str = Field(..., description="Model ID")
    model_type: str = Field(..., description="Model type")
    created_at: str = Field(..., description="Creation timestamp")
    last_used: Optional[str] = Field(None, description="Last used timestamp")
    user_id: Optional[int] = Field(None, description="User ID for personalized models")
    active: bool = Field(False, description="Whether model is currently active")
    metrics: Dict[str, Any] = Field({}, description="Performance metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "model_12345678",
                "model_type": "RandomForestModel",
                "created_at": "2023-10-25T14:30:00.000Z",
                "last_used": "2023-10-26T09:15:00.000Z",
                "user_id": 1,
                "active": True,
                "metrics": {
                    "mae": 1.2,
                    "rmse": 1.5,
                    "r2": 0.85
                }
            }
        }

class ModelsList(BaseModel):
    """List of available models."""
    models: List[Dict[str, Any]] = Field(..., description="List of models")
    count: int = Field(..., description="Total number of models")
    global_models: int = Field(..., description="Number of global models")
    user_models: int = Field(..., description="Number of user-specific models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "model_id": "model_12345678",
                        "model_type": "RandomForestModel",
                        "created_at": "2023-10-25T14:30:00.000Z",
                        "active": True,
                        "user_id": 1
                    },
                    {
                        "model_id": "model_87654321",
                        "model_type": "ARIMAModel",
                        "created_at": "2023-10-24T10:15:00.000Z",
                        "active": True,
                        "user_id": None
                    }
                ],
                "count": 2,
                "global_models": 1,
                "user_models": 1
            }
        }