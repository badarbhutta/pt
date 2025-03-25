from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import mlflow
import json

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.prediction import CycleData, SymptomData, LifestyleFactors

logger = get_logger(__name__)

class PredictionResult:
    """
    Container for prediction results.
    """
    def __init__(
        self, 
        predicted_date: date,
        confidence: float,
        prediction_window: List[date],
        features_importance: Optional[Dict[str, float]] = None,
        model_version: str = "latest",
        fertility_window_start: Optional[date] = None,
        fertility_window_end: Optional[date] = None
    ):
        self.predicted_date = predicted_date
        self.confidence = confidence
        self.prediction_window = prediction_window
        self.features_importance = features_importance
        self.model_version = model_version
        self.fertility_window_start = fertility_window_start
        self.fertility_window_end = fertility_window_end

class PredictionService:
    """
    Service for making predictions using ML models.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.model_registry_uri = settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.model_registry_uri)
    
    async def predict_next_period(
        self,
        user_id: int,
        cycle_data: CycleData,
        symptoms: Optional[SymptomData] = None,
        lifestyle_factors: Optional[LifestyleFactors] = None
    ) -> PredictionResult:
        """
        Predict the next period start date using the best available model.
        """
        try:
            # 1. Determine which model to use (user-specific or global)
            model_info = await self._get_best_model_for_user(user_id, "period_prediction")
            model_uri = model_info["model_uri"]
            model_version = model_info["version"]
            
            # 2. Preprocess and prepare features
            features_df = await self._prepare_features(
                user_id, cycle_data, symptoms, lifestyle_factors
            )
            
            # 3. Load model and make prediction
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            prediction = loaded_model.predict(features_df)
            
            # 4. Process prediction results
            predicted_date_idx = int(prediction[0])  # Number of days from last period
            last_period_date = cycle_data.period_start_dates[-1]
            predicted_date = last_period_date + timedelta(days=predicted_date_idx)
            
            # 5. Calculate confidence and window based on model uncertainty
            confidence = self._calculate_confidence(prediction, features_df)
            window_size = max(3, int(10 * (1 - confidence)))  # Wider window for less confident predictions
            prediction_window = [
                predicted_date - timedelta(days=window_size // 2 + i)
                for i in range(window_size)
            ]
            
            # 6. Extract feature importance if available
            features_importance = self._extract_feature_importance(loaded_model, features_df)
            
            return PredictionResult(
                predicted_date=predicted_date,
                confidence=confidence,
                prediction_window=prediction_window,
                features_importance=features_importance,
                model_version=model_version
            )
            
        except Exception as e:
            logger.error(f"Error predicting next period: {str(e)}")
            raise
    
    async def predict_fertility_window(
        self,
        user_id: int,
        cycle_data: CycleData,
        symptoms: Optional[SymptomData] = None,
        lifestyle_factors: Optional[LifestyleFactors] = None
    ) -> PredictionResult:
        """
        Predict the fertility window using the best available model.
        """
        try:
            # Similar to predict_next_period but focuses on fertility window
            # 1. Get next period prediction first
            period_prediction = await self.predict_next_period(
                user_id, cycle_data, symptoms, lifestyle_factors
            )
            
            # 2. Calculate fertility window based on predicted next period
            # Typical fertility window is 12-16 days before next period
            fertility_end = period_prediction.predicted_date - timedelta(days=12)
            fertility_start = period_prediction.predicted_date - timedelta(days=16)
            
            # 3. Adjust based on user's historical cycle data if available
            if len(cycle_data.cycle_lengths) >= 3:
                avg_cycle = sum(cycle_data.cycle_lengths) / len(cycle_data.cycle_lengths)
                # Adjust fertility window based on cycle length
                if avg_cycle < 26:  # Shorter cycles
                    fertility_start = period_prediction.predicted_date - timedelta(days=14)
                    fertility_end = period_prediction.predicted_date - timedelta(days=10)
                elif avg_cycle > 32:  # Longer cycles
                    fertility_start = period_prediction.predicted_date - timedelta(days=18)
                    fertility_end = period_prediction.predicted_date - timedelta(days=14)
            
            # Return with fertility window information added
            period_prediction.fertility_window_start = fertility_start
            period_prediction.fertility_window_end = fertility_end
            
            return period_prediction
            
        except Exception as e:
            logger.error(f"Error predicting fertility window: {str(e)}")
            raise
    
    async def _get_best_model_for_user(self, user_id: int, model_type: str) -> Dict[str, Any]:
        """
        Determine the best model to use for this user (user-specific or global).
        """
        try:
            # Check if user has a personalized model
            user_model_name = f"user_{user_id}_{model_type}"
            user_model_exists = await self._check_model_exists(user_model_name)
            
            if user_model_exists:
                # Get user model metadata
                user_model = await self._get_model_info(user_model_name)
                
                # Check if user model is mature enough (accuracy better than global)
                if user_model.get("is_mature", False):
                    return {
                        "model_uri": f"models:/{user_model_name}/latest",
                        "version": user_model.get("version", "latest"),
                        "is_user_specific": True
                    }
            
            # Fall back to global model
            global_model_name = f"global_{model_type}"
            return {
                "model_uri": f"models:/{global_model_name}/latest",
                "version": "latest",
                "is_user_specific": False
            }
            
        except Exception as e:
            logger.error(f"Error getting best model for user: {str(e)}")
            # Fall back to global model in case of error
            return {
                "model_uri": f"models:/global_{model_type}/latest",
                "version": "latest",
                "is_user_specific": False
            }
    
    async def _check_model_exists(self, model_name: str) -> bool:
        """
        Check if a model exists in the MLflow registry.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_details = client.get_registered_model(model_name)
            return True
        except Exception:
            return False
    
    async def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a model from MLflow.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            model_details = client.get_registered_model(model_name)
            latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
            
            # Get model metadata
            run = client.get_run(latest_version.run_id)
            metrics = run.data.metrics
            tags = run.data.tags
            
            return {
                "name": model_name,
                "version": latest_version.version,
                "accuracy": metrics.get("accuracy", 0.0),
                "mae_days": metrics.get("mean_absolute_error_days", 0.0),
                "is_mature": tags.get("is_mature", "false").lower() == "true"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    async def _prepare_features(
        self,
        user_id: int,
        cycle_data: CycleData,
        symptoms: Optional[SymptomData] = None,
        lifestyle_factors: Optional[LifestyleFactors] = None
    ) -> pd.DataFrame:
        """
        Prepare features dataframe for model input.
        """
        # Core cycle features
        features = {}
        
        # Calculate cycle statistics
        if len(cycle_data.period_start_dates) > 1:
            cycle_lengths = []
            for i in range(1, len(cycle_data.period_start_dates)):
                cycle_length = (cycle_data.period_start_dates[i] - cycle_data.period_start_dates[i-1]).days
                cycle_lengths.append(cycle_length)
                
            features["last_cycle_length"] = cycle_lengths[-1] if cycle_lengths else 28
            features["avg_cycle_length"] = sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else 28
            features["min_cycle_length"] = min(cycle_lengths) if cycle_lengths else 28
            features["max_cycle_length"] = max(cycle_lengths) if cycle_lengths else 28
            features["cycle_length_variance"] = np.var(cycle_lengths) if len(cycle_lengths) > 1 else 0
            
            if len(cycle_lengths) >= 3:
                features["trend"] = cycle_lengths[-1] - cycle_lengths[-3]
            else:
                features["trend"] = 0
        else:
            # Default values if not enough data
            features["last_cycle_length"] = 28
            features["avg_cycle_length"] = 28
            features["min_cycle_length"] = 28
            features["max_cycle_length"] = 28
            features["cycle_length_variance"] = 0
            features["trend"] = 0
        
        # Calculate period length statistics
        if cycle_data.period_end_dates and len(cycle_data.period_end_dates) == len(cycle_data.period_start_dates):
            period_lengths = []
            for i in range(len(cycle_data.period_start_dates)):
                period_length = (cycle_data.period_end_dates[i] - cycle_data.period_start_dates[i]).days + 1
                period_lengths.append(period_length)
                
            features["last_period_length"] = period_lengths[-1] if period_lengths else 5
            features["avg_period_length"] = sum(period_lengths) / len(period_lengths) if period_lengths else 5
        else:
            features["last_period_length"] = 5
            features["avg_period_length"] = 5
        
        # Add symptoms if available
        if symptoms:
            # Process most recent symptoms data
            recent_symptoms = {}
            if symptoms.cramps and any(symptoms.cramps):
                recent_symptoms["avg_cramps"] = sum(filter(None, symptoms.cramps)) / len(
                    [s for s in symptoms.cramps if s is not None]
                ) if any(s is not None for s in symptoms.cramps) else 0
            
            if symptoms.headache and any(symptoms.headache):
                recent_symptoms["avg_headache"] = sum(filter(None, symptoms.headache)) / len(
                    [s for s in symptoms.headache if s is not None]
                ) if any(s is not None for s in symptoms.headache) else 0
            
            features.update(recent_symptoms)
        
        # Add lifestyle factors if available
        if lifestyle_factors:
            recent_lifestyle = {}
            if lifestyle_factors.stress_level and any(lifestyle_factors.stress_level):
                recent_lifestyle["avg_stress"] = sum(filter(None, lifestyle_factors.stress_level)) / len(
                    [s for s in lifestyle_factors.stress_level if s is not None]
                ) if any(s is not None for s in lifestyle_factors.stress_level) else 0
                
            features.update(recent_lifestyle)
        
        return pd.DataFrame([features])
    
    def _calculate_confidence(self, prediction: np.ndarray, features_df: pd.DataFrame) -> float:
        """
        Calculate confidence score for the prediction.
        """
        # Base confidence on model type and data quality
        # This is a simplified implementation - in production this would be more sophisticated
        base_confidence = 0.75  # Default base confidence
        
        # Adjust confidence based on data quality
        if "cycle_length_variance" in features_df.columns:
            variance = features_df["cycle_length_variance"].values[0]
            # Higher variance = lower confidence
            if variance > 10:
                base_confidence *= 0.9
            elif variance > 20:
                base_confidence *= 0.8
        
        # Adjust confidence based on amount of data
        if "avg_cycle_length" in features_df.columns and features_df["avg_cycle_length"].values[0] != 28:
            # If we have real cycle data (not default), increase confidence
            base_confidence *= 1.1
        
        return min(max(base_confidence, 0.3), 0.95)  # Cap between 0.3 and 0.95
    
    def _extract_feature_importance(self, model, features_df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Extract feature importance from the model if available.
        """
        try:
            # This is a simplified implementation - actual extraction depends on model type
            feature_importance = {}
            
            # Check if model has feature_importances_ attribute (like sklearn models)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                for i, col in enumerate(features_df.columns):
                    feature_importance[col] = float(importances[i])
            elif hasattr(model, "get_feature_importance"):
                # For models that have a get_feature_importance method
                importances = model.get_feature_importance()
                for i, col in enumerate(features_df.columns):
                    feature_importance[col] = float(importances[i])
            # If we're using MLflow pyfunc models, extract from metadata
            else:
                model_meta = model.metadata.get_all()
                if "feature_importance" in model_meta:
                    feature_importance = json.loads(model_meta["feature_importance"])
            
            return feature_importance if feature_importance else None
        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            return None
