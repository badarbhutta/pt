# app/db/session.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from app.core.config import settings
from app.db.base import Base

# Create SQLAlchemy engine
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    connect_args={"check_same_thread": False}  # Only needed for SQLite
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_db_and_tables():
    """Create database and tables."""
    # Create directory for SQLite database if it doesn't exist
    if settings.SQLALCHEMY_DATABASE_URI.startswith("sqlite:///"):
        db_path = settings.SQLALCHEMY_DATABASE_URI.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database and tables created successfully")