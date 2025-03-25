from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import mlflow
from mlflow.entities import ViewType
import json

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.model import ModelInfoResponse, ModelVersionResponse, ModelEvaluationResponse
from app.schemas.prediction import CycleData

logger = get_logger(__name__)

class ModelService:
    """
    Service for managing ML models including info, versions, and evaluation.
    """
    def __init__(self, db: AsyncSession):
        self.db = db
        self.mlflow_uri = settings.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    async def get_models_info(self, user_id: int) -> List[ModelInfoResponse]:
        """
        Get information about available models for a user, including global and user-specific models.
        """
        try:
            models_info = []
            
            # Get global models
            global_model_names = await self._get_global_model_names()
            for model_name in global_model_names:
                model_info = await self._get_model_details(model_name)
                models_info.append(ModelInfoResponse(
                    model_id=model_info["model_id"],
                    model_type=model_info["model_type"],
                    features=model_info["features"],
                    target_metric=model_info["target_metric"],
                    current_version=model_info["current_version"],
                    created_at=model_info["created_at"],
                    last_updated=model_info["last_updated"],
                    accuracy=model_info.get("accuracy"),
                    average_error_days=model_info.get("average_error_days"),
                    is_user_specific=False,
                    available_versions=model_info.get("available_versions")
                ))
            
            # Get user-specific models
            user_model_names = await self._get_user_model_names(user_id)
            for model_name in user_model_names:
                model_info = await self._get_model_details(model_name)
                models_info.append(ModelInfoResponse(
                    model_id=model_info["model_id"],
                    model_type=model_info["model_type"],
                    features=model_info["features"],
                    target_metric=model_info["target_metric"],
                    current_version=model_info["current_version"],
                    created_at=model_info["created_at"],
                    last_updated=model_info["last_updated"],
                    accuracy=model_info.get("accuracy"),
                    average_error_days=model_info.get("average_error_days"),
                    is_user_specific=True,
                    available_versions=model_info.get("available_versions")
                ))
            
            return models_info
        
        except Exception as e:
            logger.error(f"Error retrieving models info: {str(e)}")
            raise
    
    async def get_model_versions(self, model_id: str, user_id: int) -> List[ModelVersionResponse]:
        """
        Get versions information for a specific model.
        """
        try:
            # Check if user has access to this model
            if model_id.startswith("user_") and not model_id.startswith(f"user_{user_id}_"):
                logger.warning(f"User {user_id} attempted to access unauthorized model {model_id}")
                return []
            
            # Get model versions from MLflow
            versions_info = []
            versions = self.client.search_model_versions(f"name='{model_id}'")
            
            for version in versions:
                # Get run details for metrics
                try:
                    run = self.client.get_run(version.run_id) if version.run_id else None
                    metrics = run.data.metrics if run else {}
                    tags = run.data.tags if run else {}
                    
                    # Parse parameters if available
                    params = {}
                    if run and run.data.params:
                        params = run.data.params
                    
                    # Parse features list
                    features = []
                    if tags and "features" in tags:
                        features = json.loads(tags["features"])
                    
                    versions_info.append(ModelVersionResponse(
                        version=version.version,
                        model_id=model_id,
                        created_at=datetime.strptime(version.creation_timestamp/1000, "%Y-%m-%d %H:%M:%S") 
                            if version.creation_timestamp else datetime.now(),
                        is_active=version.current_stage == "Production",
                        metrics=metrics,
                        training_data_size=int(tags.get("training_data_size", "0")),
                        features=features,
                        parameters=params,
                        artifact_uri=version.source if hasattr(version, "source") else None
                    ))
                except Exception as e:
                    logger.error(f"Error processing model version {version.version}: {str(e)}")
                    # Skip this version if there's an error
                    continue
            
            # Sort versions by creation time, newest first
            versions_info.sort(key=lambda x: x.created_at, reverse=True)
            return versions_info
        
        except Exception as e:
            logger.error(f"Error retrieving model versions: {str(e)}")
            raise
    
    async def evaluate_model(self, model_id: str, user_id: int, cycle_data: CycleData, version: Optional[str] = None) -> ModelEvaluationResponse:
        """
        Evaluate a model's performance on user data.
        """
        try:
            # If version not specified, use latest production version
            if not version:
                version = await self._get_latest_production_version(model_id)
            
            # Load the model
            model_uri = f"models:/{model_id}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Convert cycle data to features
            features = await self._prepare_evaluation_features(cycle_data)
            
            # Get ground truth data (actual cycle start dates)
            # For evaluation, we'll use a holdout strategy: remove most recent cycle and predict it
            if len(cycle_data.period_start_dates) < 3:
                raise ValueError("Insufficient data for evaluation. Need at least 3 cycles.")
            
            # Evaluate the model (simplified for this implementation)
            last_actual_cycle_length = (cycle_data.period_start_dates[-1] - 
                                      cycle_data.period_start_dates[-2]).days
            
            # Make prediction using the model
            predictions = model.predict(features)
            predicted_cycle_length = predictions[0] if isinstance(predictions, (list, tuple, np.ndarray)) else predictions
            
            # Calculate error
            error_days = abs(predicted_cycle_length - last_actual_cycle_length)
            
            # Calculate additional metrics
            metrics = {
                "absolute_error_days": error_days,
                "squared_error": error_days ** 2
            }
            
            # Determine if this model is recommended for the user
            # A simple heuristic: error less than 3 days is good
            is_recommended = error_days <= 3
            
            return ModelEvaluationResponse(
                model_id=model_id,
                version=version,
                metrics=metrics,
                average_error_days=float(error_days),
                is_recommended=is_recommended,
                evaluation_timestamp=datetime.now(),
                details={
                    "evaluation_method": "holdout",
                    "cycles_used": len(cycle_data.period_start_dates) - 1
                }
            )
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    async def promote_model_version(self, model_id: str, version: str, user_id: int) -> Dict[str, Any]:
        """
        Promote a specific model version to production.
        Only admins can promote global models, users can promote their own models.
        """
        try:
            # Check if user has permission to promote this model
            if model_id.startswith("global_"):
                # Check if user is admin
                is_admin = await self._is_admin(user_id)
                if not is_admin:
                    raise ValueError(f"User {user_id} does not have permission to promote global models")
            elif model_id.startswith(f"user_{user_id}_"):
                # User can promote their own models
                pass
            else:
                # User cannot promote other users' models
                raise ValueError(f"User {user_id} does not have permission to promote model {model_id}")
            
            # Promote the version to Production
            self.client.transition_model_version_stage(
                name=model_id,
                version=version,
                stage="Production",
                archive_existing_versions=True  # Archive existing production versions
            )
            
            # Update model metadata
            timestamp = datetime.now().isoformat()
            self.client.set_model_version_tag(
                name=model_id,
                version=version,
                key="promoted_by",
                value=str(user_id)
            )
            self.client.set_model_version_tag(
                name=model_id,
                version=version,
                key="promoted_at",
                value=timestamp
            )
            
            return {
                "model_id": model_id,
                "version": version,
                "promoted_to": "Production",
                "promoted_by": user_id,
                "promoted_at": timestamp
            }
        
        except Exception as e:
            logger.error(f"Error promoting model version: {str(e)}")
            raise
    
    async def _get_global_model_names(self) -> List[str]:
        """
        Get names of all global models available.
        """
        models = self.client.search_registered_models(filter_string="tags.type = 'global'")
        return [model.name for model in models]
    
    async def _get_user_model_names(self, user_id: int) -> List[str]:
        """
        Get names of all models specific to this user.
        """
        models = self.client.search_registered_models(filter_string=f"tags.user_id = '{user_id}'")
        return [model.name for model in models]
    
    async def _get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        """
        try:
            model = self.client.get_registered_model(model_name)
            
            # Get latest version in Production stage
            latest_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            current_version = latest_versions[0].version if latest_versions else "None"
            
            # Get creation time and last update time
            created_at = datetime.fromtimestamp(model.creation_timestamp / 1000) if hasattr(model, "creation_timestamp") else datetime.now()
            last_updated = datetime.fromtimestamp(model.last_updated_timestamp / 1000) if hasattr(model, "last_updated_timestamp") else created_at
            
            # Get features and target metric from tags
            features = []
            target_metric = "mean_absolute_error"
            if hasattr(model, "tags") and model.tags:
                if "features" in model.tags:
                    features = json.loads(model.tags["features"])
                if "target_metric" in model.tags:
                    target_metric = model.tags["target_metric"]
            
            # Get model type from name
            if "period_prediction" in model_name:
                model_type = "period_prediction"
            elif "fertility_prediction" in model_name:
                model_type = "fertility_prediction"
            elif "symptom_prediction" in model_name:
                model_type = "symptom_prediction"
            else:
                model_type = "other"
            
            # Get performance metrics from latest production version
            accuracy = None
            average_error_days = None
            if latest_versions:
                run = self.client.get_run(latest_versions[0].run_id) if latest_versions[0].run_id else None
                if run and run.data.metrics:
                    accuracy = run.data.metrics.get("accuracy")
                    average_error_days = run.data.metrics.get("mean_absolute_error_days")
            
            # Count available versions
            all_versions = self.client.search_model_versions(f"name='{model_name}'")
            available_versions = len(all_versions)
            
            return {
                "model_id": model_name,
                "model_type": model_type,
                "features": features,
                "target_metric": target_metric,
                "current_version": current_version,
                "created_at": created_at,
                "last_updated": last_updated,
                "accuracy": accuracy,
                "average_error_days": average_error_days,
                "available_versions": available_versions
            }
        
        except Exception as e:
            logger.error(f"Error getting model details for {model_name}: {str(e)}")
            # Return minimal info in case of error
            return {
                "model_id": model_name,
                "model_type": "unknown",
                "features": [],
                "target_metric": "unknown",
                "current_version": "None",
                "created_at": datetime.now(),
                "last_updated": datetime.now()
            }
    
    async def _get_latest_production_version(self, model_name: str) -> str:
        """
        Get the latest production version for a model.
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            if versions:
                return versions[0].version
            else:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                if versions:
                    # If no production version exists, return the latest version
                    versions.sort(key=lambda x: int(x.version), reverse=True)
                    return versions[0].version
                else:
                    raise ValueError(f"No versions found for model {model_name}")
        except Exception as e:
            logger.error(f"Error getting latest production version: {str(e)}")
            raise
    
    async def _prepare_evaluation_features(self, cycle_data: CycleData) -> pd.DataFrame:
        """
        Prepare features for model evaluation from cycle data.
        """
        features = {}
        
        # Calculate cycle statistics excluding most recent cycle (which we'll use for validation)
        if len(cycle_data.period_start_dates) > 2:
            cycle_lengths = []
            for i in range(1, len(cycle_data.period_start_dates) - 1):  # Exclude most recent cycle
                cycle_length = (cycle_data.period_start_dates[i] - cycle_data.period_start_dates[i-1]).days
                cycle_lengths.append(cycle_length)
                
            features["last_cycle_length"] = cycle_lengths[-1] if cycle_lengths else 28
            features["avg_cycle_length"] = sum(cycle_lengths) / len(cycle_lengths) if cycle_lengths else 28
            features["min_cycle_length"] = min(cycle_lengths) if cycle_lengths else 28
            features["max_cycle_length"] = max(cycle_lengths) if cycle_lengths else 28
            features["cycle_length_variance"] = np.var(cycle_lengths) if len(cycle_lengths) > 1 else 0
            features["cycle_count"] = len(cycle_lengths)
        else:
            # Default values if not enough data
            features["last_cycle_length"] = 28
            features["avg_cycle_length"] = 28
            features["min_cycle_length"] = 28
            features["max_cycle_length"] = 28
            features["cycle_length_variance"] = 0
            features["cycle_count"] = 0
        
        return pd.DataFrame([features])
    
    async def _is_admin(self, user_id: int) -> bool:
        """
        Check if a user has admin privileges.
        """
        # This is a simplified implementation - in production this would query a database
        admin_ids = settings.ADMIN_USER_IDS
        return user_id in admin_ids
