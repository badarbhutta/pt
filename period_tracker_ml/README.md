# Period Tracker ML API

A machine learning-powered API for predicting menstrual cycles, fertility windows, and ovulation dates. This system uses multiple machine learning models including ARIMA, Random Forest, and Gradient Boosting to provide accurate predictions based on user cycle history and other factors.

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Environment Setup](#environment-setup)
5. [Configuration](#configuration)
6. [Database Setup](#database-setup)
7. [Running the API](#running-the-api)
8. [API Documentation](#api-documentation)
9. [ML Models](#ml-models)
10. [Deployment Options](#deployment-options)
11. [Monitoring and Maintenance](#monitoring-and-maintenance)
12. [Troubleshooting](#troubleshooting)

## Features

- **Period Prediction**: Predict the start date of the next menstrual period
- **Fertility Window**: Calculate the fertility window based on cycle history
- **Ovulation Prediction**: Estimate ovulation dates
- **Multiple ML Models**: Uses ensemble of different models for more accurate predictions
- **Continuous Learning**: Improves with user feedback and additional data
- **Model Training API**: Endpoints for training and managing ML models
- **User Data Management**: Store and retrieve user cycle data securely

## System Architecture

The system consists of the following components:

- **FastAPI Backend**: Provides RESTful API endpoints
- **ML Module**: Contains machine learning models for predictions
- **Database**: Stores user data, cycle information, and model metadata
- **Training Service**: Handles model training in the background

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- PostgreSQL database (recommended for production)
- Virtual environment (recommended)

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/period-tracker-ml.git
   cd period-tracker-ml
   ```

2. Create and activate a virtual environment:
   ```bash
   # For Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

Create a `.env` file in the root directory with the following variables:

```
# API settings
PROJECT_NAME="Period Tracker ML API"
API_V1_STR="/api/v1"
SECRET_KEY="your-secret-key-here"

# Database connection
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=period_tracker

# Model storage
MODEL_STORAGE_PATH=./models
```

## Configuration

The application configuration is managed in `app/core/config.py`. You can modify settings directly in this file or override them using environment variables.

Key configuration options include:

- **API Settings**: Endpoints, versioning, and security configurations
- **Database Settings**: Connection parameters for PostgreSQL
- **Model Settings**: Storage paths and training parameters
- **Logging Settings**: Log levels and output destinations

## Database Setup

### Local Development Database

1. Install PostgreSQL on your system if not already installed

2. Create a new database:
   ```bash
   # Connect to PostgreSQL
   psql -U postgres
   
   # Create database
   CREATE DATABASE period_tracker;
   
   # Exit psql
   \q
   ```

3. The application will automatically create the required tables on first run using SQLAlchemy migrations

### Using SQLite for Development (Alternative)

For simple development or testing, you can use SQLite instead:

1. Modify your .env file:
   ```
   SQLALCHEMY_DATABASE_URI=sqlite:///./period_tracker.db
   ```

## Running the API

### Development Server

To start the development server:

```bash
# From the root directory
python run.py
```

Alternatively, you can use Uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

### Testing the Installation

Open your browser and navigate to http://localhost:8000/docs to access the Swagger UI documentation.

## API Documentation

### Interactive Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key API Endpoints

#### Predictions API

- `POST /api/v1/predictions/period`: Predict next period date
- `POST /api/v1/predictions/fertility`: Calculate fertility window
- `POST /api/v1/predictions/ovulation`: Predict ovulation date
- `POST /api/v1/predictions/feedback`: Submit feedback on prediction accuracy

#### Training API

- `POST /api/v1/training/start`: Start a new model training job
- `GET /api/v1/training/status/{training_id}`: Check training status
- `GET /api/v1/training/models`: List available models
- `GET /api/v1/training/models/{model_id}`: Get model details
- `POST /api/v1/training/models/{model_id}/activate`: Activate a specific model

### Sample API Requests

#### Predict Next Period

```bash
curl -X POST "http://localhost:8000/api/v1/predictions/period" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "cycle_lengths": [28, 30, 29, 31],
    "last_period_date": "2023-03-01"
  }'
```

#### Start Model Training

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "arima",
    "training_params": {"p": 1, "d": 1, "q": 0}
  }'
```

## ML Models

The system uses several machine learning models for predictions:

### Available Models

1. **ARIMA (Time Series Analysis)**
   - Best for: Users with regular cycles and sufficient historical data
   - Configuration parameters: p, d, q values

2. **Random Forest**
   - Best for: Considering multiple factors beyond just cycle history
   - Configuration parameters: n_estimators, max_depth

3. **Gradient Boosting**
   - Best for: Improved accuracy with complex patterns
   - Configuration parameters: n_estimators, learning_rate, max_depth

4. **Ensemble Model**
   - Combines outputs from multiple models using weighted averaging
   - Currently uses confidence scores to weight individual model predictions

### Custom Models

To add a new custom model:

1. Extend the BaseModel class in `app/ml/model_implementations.py`
2. Implement required methods (train, predict, save, load)
3. Update the ModelFactory in `app/ml/models.py` to include your new model

## Deployment Options

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t period-tracker-ml .
   ```

2. Run the container:
   ```bash
   docker run -d \
     --name period-tracker-api \
     -p 8000:8000 \
     -e POSTGRES_SERVER=host.docker.internal \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=your_password \
     -e POSTGRES_DB=period_tracker \
     -e SECRET_KEY=your-secret-key \
     period-tracker-ml
   ```

### Cloud Deployment

#### AWS Elastic Beanstalk

1. Install the EB CLI:
   ```bash
   pip install awsebcli
   ```

2. Initialize EB application:
   ```bash
   eb init -p python-3.8 period-tracker-ml
   ```

3. Create an environment and deploy:
   ```bash
   eb create period-tracker-ml-env
   ```

4. Configure environment variables through the AWS console or using:
   ```bash
   eb setenv POSTGRES_SERVER=your-rds-instance.amazonaws.com POSTGRES_USER=postgres ...
   ```

#### Heroku

1. Install Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create period-tracker-ml
   ```

3. Add PostgreSQL addon:
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```

4. Set environment variables:
   ```bash
   heroku config:set SECRET_KEY=your-secret-key
   ```

5. Deploy the application:
   ```bash
   git push heroku main
   ```

### Production Considerations

1. **Database**: Use a managed PostgreSQL service (AWS RDS, Azure Database, GCP Cloud SQL)
2. **Scaling**: Consider using Kubernetes for container orchestration
3. **Security**: Implement proper authentication and authorization
4. **Storage**: Use cloud storage (S3, Azure Blob) for model persistence
5. **HTTPS**: Configure SSL/TLS for encrypted connections

## Monitoring and Maintenance

### Logging

Logs are configured in `app/core/logging_config.py`. By default, logs are written to both console and file.

To view logs in production:

```bash
# Docker logs
docker logs -f period-tracker-api

# Heroku logs
heroku logs --tail

# AWS Elastic Beanstalk logs
eb logs
```

### Health Checks

The API provides a health check endpoint at `/health` which returns the current status of the service.

### Backup and Recovery

1. **Database Backups**: Configure regular PostgreSQL backups
2. **Model Backups**: Periodically backup the models directory

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check database credentials in .env file
   - Verify that the PostgreSQL server is running
   - Check network connectivity to the database server

2. **Model Training Failures**
   - Check logs for specific error messages
   - Verify that required dependencies are installed
   - Ensure sufficient disk space for model storage

3. **API Startup Issues**
   - Confirm all required environment variables are set
   - Check for port conflicts
   - Verify Python version compatibility

### Getting Help

If you encounter issues not covered in this documentation, please:

1. Check the logs for detailed error messages
2. Open an issue on the GitHub repository with details about the problem
3. Include relevant logs and environment information

## License

This project is licensed under the MIT License - see the LICENSE file for details.
