from fastapi import APIRouter

from app.api.endpoints import predictions, models, training, health

api_router = APIRouter()
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
