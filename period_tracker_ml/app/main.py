# app/main.py
"""Main application for Period Tracker ML API."""
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time

from app.core.config import settings
from app.api.api_v1.api import api_router
from app.db.session import create_db_and_tables

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("period_tracker")

# Create application
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="""
    Period Tracker ML API provides machine learning-powered predictions for menstrual cycles,
    fertility windows, and ovulation dates. The API also allows training custom models based
    on user data for increased prediction accuracy.
    """,
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request details."""
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Log request details
    process_time = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Include API router
app.include_router(
    api_router,
    prefix=settings.API_V1_STR
)

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": "1.0.0",
        "description": "Period Tracker ML API powered by machine learning models",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    logger.info("Starting Period Tracker ML API...")
    
    # Create database tables
    try:
        create_db_and_tables()
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
    
    # Create models directory if it doesn't exist
    os.makedirs(settings.MODEL_STORAGE_PATH, exist_ok=True)
    
    logger.info("Period Tracker ML API started successfully")

if __name__ == "__main__":
    # This is for development purposes only
    # In production, use uvicorn command directly
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)