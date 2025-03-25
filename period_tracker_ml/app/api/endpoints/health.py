from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
from datetime import datetime

from app.core.config import settings
from app.api.deps import get_db
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint that returns basic application information.
    """
    try:
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "app_name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": "development" if settings.API_KEY == "development_api_key" else "production"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service is experiencing issues"
        )

@router.get("/db", response_model=Dict[str, Any])
async def db_health_check(db=Depends(get_db)):
    """
    Database connection health check.
    """
    try:
        # Attempt a simple query to verify database connection
        db_status = await db.execute("SELECT 1")
        return {
            "status": "ok" if db_status else "error",
            "timestamp": datetime.now().isoformat(),
            "database": settings.POSTGRES_DB,
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "database": settings.POSTGRES_DB,
            "message": f"Database connection failed: {str(e)}"
        }
