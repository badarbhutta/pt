# Period Tracker ML Backend

A machine learning backend service for predicting menstrual cycles, fertility windows, and ovulation dates. This service is designed to work with a Laravel frontend application, providing API endpoints for predictions and model training.

## Features

- **Period Predictions**: Predict next period start dates and cycle lengths
- **Fertility Window Predictions**: Identify fertile days for users
- **Ovulation Predictions**: Predict ovulation days
- **Continuous Learning**: Incorporate user feedback to improve prediction accuracy
- **Multiple ML Models**: Ensemble of ARIMA, Random Forest, and Gradient Boosting models
- **User-Specific Models**: Personalized predictions for individual users
- **API-First Design**: RESTful API endpoints for easy integration

## Architecture

The system is composed of several key components:

- **ML Models**: Core prediction algorithms for different aspects of menstrual cycles
- **API Endpoints**: FastAPI routes for predictions and model training
- **Data Storage**: SQLAlchemy ORM for storing user cycles, predictions, and feedback
- **Training Pipeline**: Background tasks for continuous model improvement

## API Endpoints

### Prediction Endpoints

- `GET /api/v1/predictions/`: Information about available prediction endpoints
- `POST /api/v1/predictions/period`: Predict next period date and cycle length
- `POST /api/v1/predictions/fertility`: Predict fertility window
- `POST /api/v1/predictions/ovulation`: Predict ovulation date
- `POST /api/v1/predictions/feedback`: Submit feedback on prediction accuracy
- `GET /api/v1/predictions/confidence`: Get confidence scores for predictions
- `GET /api/v1/predictions/history`: Get historical predictions
- `GET /api/v1/predictions/accuracy`: Get prediction accuracy statistics

### Training Endpoints

- `GET /api/v1/training/`: Information about training endpoints
- `POST /api/v1/training/trigger`: Trigger model training
- `GET /api/v1/training/status/{training_id}`: Check training job status
- `GET /api/v1/training/models`: List available models
- `GET /api/v1/training/models/{model_id}`: Get model details
- `GET /api/v1/training/models/{model_id}/metrics`: Get model metrics
- `POST /api/v1/training/models/{model_id}/activate`: Activate a model for predictions
- `GET /api/v1/training/feature-importance/{model_id}`: Get feature importance

## ML Models

### ARIMA Models
Time series models for predicting cycle length based on historical patterns.

### Random Forest Models
Regression models using multiple features like previous cycle lengths and symptoms.

### Gradient Boosting Models
Advanced regression models for fertility and ovulation prediction.

### Ensemble Integration
Combines multiple models for more accurate predictions with confidence scores.

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- SQLAlchemy
- scikit-learn
- statsmodels
- pandas
- numpy

### Installation

1. Clone the repository:


2. Install dependencies:


3. Run the application:


4. Access the API documentation: