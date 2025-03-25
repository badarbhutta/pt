# app/models/prediction.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base, BaseModel

class Prediction(Base, BaseModel):
    """Model for period and fertility predictions."""
    __tablename__ = "predictions"
    
    user_id = Column(Integer, index=True, nullable=False)
    cycle_id = Column(Integer, ForeignKey("cycles.id"), nullable=True)
    
    # Prediction type
    prediction_type = Column(String, nullable=False)  # "period", "fertility", "ovulation"
    
    # Prediction values
    predicted_date = Column(DateTime, nullable=False)
    predicted_end_date = Column(DateTime, nullable=True)  # For fertility window end
    predicted_cycle_length = Column(Integer, nullable=True)
    predicted_period_length = Column(Integer, nullable=True)
    
    # Confidence
    confidence = Column(Float, nullable=False)
    
    # Models used
    models_used = Column(String, nullable=True)  # JSON string of model IDs
    model_contributions = Column(String, nullable=True)  # JSON string of model contributions
    
    # Status
    is_active = Column(Boolean, default=True)  # If this is the latest prediction
    was_accurate = Column(Boolean, nullable=True)  # Set after feedback or actual data
    
    # Relationships
    cycle = relationship("Cycle", back_populates="predictions")
    feedback = relationship("Feedback", back_populates="prediction", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Prediction(user_id={self.user_id}, type={self.prediction_type}, date={self.predicted_date})>"