# app/models/feedback.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base, BaseModel

class Feedback(Base, BaseModel):
    """Model for user feedback on predictions."""
    __tablename__ = "feedback"
    
    user_id = Column(Integer, index=True, nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True, index=True)
    
    # Feedback data
    prediction_type = Column(String, nullable=False)  # "period", "fertility", "ovulation"
    predicted_date = Column(DateTime, nullable=False)
    actual_date = Column(DateTime, nullable=False)
    accuracy_rating = Column(Integer, nullable=False)  # 1-5 scale
    
    # Additional information
    days_difference = Column(Integer, nullable=True)  # Days off from prediction
    comments = Column(String, nullable=True)
    
    # Relationships
    prediction = relationship("Prediction", back_populates="feedback")
    
    def __repr__(self):
        return f"<Feedback(user_id={self.user_id}, type={self.prediction_type}, rating={self.accuracy_rating})>"