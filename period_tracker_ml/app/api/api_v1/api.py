# app/api/api_v1/api.py
"""API router for Period Tracker ML application."""
from fastapi import APIRouter
from app.api.api_v1.endpoints import predictions, training

api_router = APIRouter()

# Include prediction endpoints
api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["predictions"]
)

# Include training endpoints
api_router.include_router(
    training.router,
    prefix="/training",
    tags=["training"]
)