# app/db/base.py
# Import all models here to ensure they are registered with the Base metadata
from app.db.base_class import Base
from app.models.cycle import Cycle
from app.models.prediction import Prediction
from app.models.feedback import Feedback

# All models should be imported so they're registered with SQLAlchemy
__all__ = ["Base", "Cycle", "Prediction", "Feedback"]