# app/db/base_class.py
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
from sqlalchemy import Column, Integer, DateTime
from datetime import datetime

# Create a declarative base class for all models
Base: DeclarativeMeta = declarative_base()

# Add a common base class with id and creation date
class BaseModel:
    """Base class for all database models."""
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)