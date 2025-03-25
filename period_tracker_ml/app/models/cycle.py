# app/models/cycle.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base, BaseModel

class Cycle(Base, BaseModel):
    """Model for menstrual cycle data."""
    __tablename__ = "cycles"
    
    user_id = Column(Integer, index=True, nullable=False)
    start_date = Column(DateTime, nullable=False, index=True)
    end_date = Column(DateTime, nullable=True)  # Null until period ends
    
    # Cycle attributes
    cycle_length = Column(Integer, nullable=True)  # Length of the full cycle in days
    period_length = Column(Integer, nullable=True)  # Length of the period in days
    
    # Symptoms and notes
    symptoms = Column(String, nullable=True)  # JSON string of symptoms
    flow_level = Column(Integer, nullable=True)  # 1-5 scale
    notes = Column(String, nullable=True)
    
    # Flags
    is_predicted = Column(Boolean, default=False)  # True if this was a predicted cycle
    was_regular = Column(Boolean, default=True)  # False if cycle was irregular
    
    # Relationships
    predictions = relationship("Prediction", back_populates="cycle", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Cycle(user_id={self.user_id}, start_date={self.start_date})>"