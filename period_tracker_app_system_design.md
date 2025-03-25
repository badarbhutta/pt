# Period Tracker App - System Architecture Design

## Table of Contents

1. [Implementation Approach](#implementation-approach)
2. [System Architecture Overview](#system-architecture-overview)
3. [Data Structures and Interfaces](#data-structures-and-interfaces)
4. [Communication Protocols and Data Flow](#communication-protocols-and-data-flow)
5. [Security Architecture](#security-architecture)
6. [API Design](#api-design)
7. [ML Pipeline Architecture](#ml-pipeline-architecture)
8. [Caching and Performance Optimization](#caching-and-performance-optimization)
9. [Scaling Considerations](#scaling-considerations)

## Implementation Approach

### Technology Stack Selection

Based on the PRD requirements, we will implement a dual-component architecture:

1. **Laravel Backend (PHP)**: 
   - Handles user authentication, data management, API endpoints, and business logic
   - Manages PostgreSQL database interactions and data validation
   - Provides admin interfaces and monitoring capabilities

2. **FastAPI ML Engine (Python)**:
   - Implements the machine learning models (ARIMA, Random Forest, Gradient Boosting)
   - Handles model training, validation, and serving predictions
   - Manages model versioning and performance metrics

### Key Implementation Challenges

1. **Data Privacy and Security**: 
   - Implementation of end-to-end encryption for sensitive health data
   - Anonymization procedures for ML training data
   - Compliance with healthcare data regulations

2. **Model Accuracy and Personalization**:
   - Balancing global model performance with personalized user models
   - Managing cold-start problems for new users
   - Ensuring continuous model improvement without degradation

3. **System Integration**:
   - Seamless communication between Laravel and FastAPI components
   - Efficient data transfer while maintaining performance
   - Consistent API versioning across components

4. **Scalability**:
   - Designing for potential high user growth
   - Ensuring ML training processes scale efficiently
   - Managing database growth and query performance

### Open-Source Libraries

**Laravel Backend**:
- Laravel Sanctum for authentication
- Laravel Horizon for queue management
- Laravel Telescope for debugging
- Spatie Permission for role management
- Laravel Encryption for data protection

**FastAPI ML Engine**:
- Pandas and NumPy for data manipulation
- Statsmodels for ARIMA implementation
- Scikit-learn for Random Forest and Gradient Boosting models
- MLflow for model tracking and versioning
- FastAPI for API implementation
- Pydantic for data validation

## System Architecture Overview

The period tracker application follows a microservices-inspired architecture with two main components that communicate via RESTful APIs. The system integrates with a PostgreSQL database for data storage and Redis for caching and queue management.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Client Applications                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Gateway / Load Balancer                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                 ┌─────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
┌────────────────────────────────┐      ┌─────────────────────────────────┐
│      Laravel API Backend       │      │       FastAPI ML Engine         │
│                                │◄────►│                                 │
│  ┌─────────────┐ ┌──────────┐  │      │  ┌─────────┐  ┌─────────────┐  │
│  │User & Data  │ │Admin     │  │      │  │Model    │  │Prediction   │  │
│  │Management   │ │Dashboard │  │      │  │Training │  │Service      │  │
│  └─────────────┘ └──────────┘  │      │  └─────────┘  └─────────────┘  │
└───────────────┬────────────────┘      └──────────────┬────────────────┘
                │                                       │
                ▼                                       ▼
┌────────────────────────────────┐      ┌─────────────────────────────────┐
│      PostgreSQL Database       │      │           ML Artifacts           │
│  ┌─────────────┐ ┌──────────┐  │      │  ┌─────────┐  ┌─────────────┐  │
│  │User Data    │ │Cycle &   │  │      │  │Model    │  │Feature      │  │
│  │             │ │Health Data│  │      │  │Versions │  │Importance   │  │
│  └─────────────┘ └──────────┘  │      │  └─────────┘  └─────────────┘  │
└────────────────────────────────┘      └─────────────────────────────────┘
                │                                       │
                └───────────────┬───────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Redis Cache & Queue Manager                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Structures and Interfaces

### Database Schema Design

The PostgreSQL database schema is designed to efficiently store user data, period tracking information, and support the ML pipeline.

#### Core Tables

1. **users**
   - Stores user account information and preferences
   - Contains privacy settings and data sharing preferences

2. **cycles**
   - Records individual menstrual cycle data
   - Links to users and contains period start/end dates

3. **symptoms**
   - Catalog of trackable symptoms categorized by type
   - Includes symptom metadata and display information

4. **symptom_logs**
   - User-logged symptom instances with severity and notes
   - Timestamps for correlation with cycle phases

5. **biometric_logs**
   - Stores basal body temperature, weight, and other measurements
   - Timestamped for correlation analysis

6. **fertility_logs**
   - Specialized tracking for fertility indicators
   - Includes cervical fluid observations, ovulation tests results

7. **lifestyle_logs**
   - Records sleep, exercise, stress, and other lifestyle factors
   - Used for correlation with cycle and symptom data

8. **predictions**
   - Stores generated predictions for cycles, fertility, and ovulation
   - Includes confidence scores and generated timestamps

9. **prediction_feedback**
   - User feedback on prediction accuracy
   - Used to improve model performance

10. **ml_models**
    - Tracks deployed model versions and their performance metrics
    - Links global and user-specific models

11. **model_metrics**
    - Detailed performance metrics for each model version
    - Used for monitoring and evaluation

### Key Relationships

- Each user has many cycles, symptom_logs, biometric_logs, etc.
- Each cycle may have many associated logs of different types
- Each prediction is linked to a user and specific model version
- Feedback is linked to specific predictions for performance tracking

## Communication Protocols and Data Flow

### API Communication

1. **Client to Laravel API**
   - RESTful HTTP/HTTPS endpoints
   - JWT authentication for secure access
   - JSON request/response format

2. **Laravel to FastAPI Communication**
   - Internal RESTful API calls
   - API key authentication
   - Batch data processing for training
   - Real-time requests for predictions

3. **Database Access**
   - Laravel ORM for database operations
   - Connection pooling for performance
   - Read/write separation for scaling

### Data Flow Scenarios

#### User Data Logging Flow

1. Client sends period or symptom data to Laravel API
2. Laravel validates and stores data in PostgreSQL
3. Laravel queues a job to update predictions
4. Queue worker sends prediction request to FastAPI
5. FastAPI generates new predictions using latest models
6. Predictions are stored in the database
7. Notification sent to client about updated predictions

#### ML Training Flow

1. Scheduled job or manual trigger initiates training
2. Laravel prepares anonymized training dataset
3. Training job is queued with prioritization
4. FastAPI processes the training job
5. New model is validated against historical data
6. If performance improves, model version is updated
7. Model metrics are stored for monitoring
8. Laravel admin dashboard is updated with new metrics

## Security Architecture

### Data Protection

1. **Encryption**
   - Transport Layer Security (TLS) for all API communications
   - AES-256 encryption for sensitive data at rest
   - Key management system with rotation policies

2. **Data Anonymization**
   - Removal of personally identifiable information for training data
   - Aggregation techniques for reporting and analytics
   - Differential privacy implementation for sensitive analytics

3. **Access Controls**
   - Role-based access control (RBAC) for system functions
   - Fine-grained permissions for admin functions
   - Audit logging for all sensitive operations

### Authentication and Authorization

1. **User Authentication**
   - JWT-based authentication with refresh tokens
   - Multi-factor authentication option for added security
   - Password policies and secure reset procedures

2. **API Security**
   - Rate limiting to prevent abuse
   - CORS policies for frontend access
   - API versioning for compatibility

3. **Internal Service Authentication**
   - Service-to-service authentication using API keys
   - IP whitelisting for internal services
   - Mutual TLS for sensitive communications

### Compliance Considerations

- GDPR compliance for European users
- HIPAA considerations for health data protection
- Data residency controls for international deployments
- Consent management for data usage

## API Design

### API Design Principles

1. **RESTful Design**
   - Resource-oriented endpoints
   - Standard HTTP methods (GET, POST, PUT, DELETE)
   - Consistent URL patterns

2. **Versioning Strategy**
   - URL-based versioning (e.g., /api/v1/)
   - Gradual deprecation of older versions
   - Version documentation in OpenAPI/Swagger

3. **Response Formats**
   - Consistent JSON structure
   - Standard error handling format
   - Pagination for collection endpoints

### Core API Endpoints

#### User Management

```
/api/v1/auth
  POST /register       # Create new user account
  POST /login          # Authenticate user
  POST /refresh-token  # Refresh authentication token
  POST /forgot-password # Initiate password reset
  POST /reset-password # Complete password reset

/api/v1/users
  GET /profile         # Get user profile information
  PUT /profile         # Update profile information
  PUT /settings        # Update application settings
  GET /preferences     # Get user preferences
  PUT /preferences     # Update user preferences
```

#### Cycle Management

```
/api/v1/cycles
  GET /                # List user's cycles
  POST /               # Record new cycle
  GET /{id}            # Get specific cycle details
  PUT /{id}            # Update cycle information
  DELETE /{id}         # Delete cycle record
  GET /statistics      # Get cycle statistics
  GET /current         # Get current cycle status
```

#### Symptom Tracking

```
/api/v1/symptoms
  GET /categories      # List symptom categories
  GET /               # List all available symptoms
  POST /log           # Log new symptom occurrence
  PUT /log/{id}       # Update symptom log
  DELETE /log/{id}    # Delete symptom log
  GET /logs           # Get user's symptom history
  GET /correlations   # Get symptom-cycle correlations
```

#### Prediction Endpoints

```
/api/v1/predictions
  GET /period         # Get period predictions
  GET /fertility      # Get fertility window predictions
  GET /ovulation      # Get ovulation day predictions
  POST /feedback      # Submit feedback on predictions
  GET /history        # Get historical predictions
  GET /accuracy       # Get prediction accuracy stats
```

#### Admin and Monitoring

```
/api/v1/admin
  GET /models              # List ML model versions
  GET /models/{id}         # Get specific model details
  GET /models/{id}/metrics # Get model performance metrics
  POST /models/train       # Trigger model training
  PUT /models/{id}/activate # Activate specific model
  GET /users/statistics    # Get anonymized user statistics
  GET /system/health       # Get system health status
```

## ML Pipeline Architecture

### Model Components

1. **ARIMA Time Series Model**
   - Purpose: Baseline period prediction based on historical cycle lengths
   - Implementation: Using statsmodels ARIMA implementation
   - Input features: Historical cycle lengths, seasonal patterns

2. **Random Forest Regressor**
   - Purpose: Evaluate symptom impact on cycle timing
   - Implementation: scikit-learn RandomForestRegressor
   - Input features: Symptoms, lifestyle factors, historical patterns

3. **Gradient Boosting Regressor**
   - Purpose: Fertility window and ovulation timing prediction
   - Implementation: scikit-learn GradientBoostingRegressor
   - Input features: Temperature data, fertility indicators, symptoms

4. **Ensemble Integrator**
   - Purpose: Combine predictions from individual models
   - Implementation: Weighted averaging with confidence scoring
   - Logic: Adjust weights based on model performance for each user

### Training Pipeline

1. **Data Preparation**
   - Extract anonymized training data from PostgreSQL
   - Preprocess data (normalization, imputation, feature engineering)
   - Split into training and validation sets

2. **Model Training**
   - Global model training on all anonymized data
   - User-specific model fine-tuning for users with sufficient data
   - Hyperparameter optimization for each model component

3. **Model Validation**
   - Evaluate on holdout validation data
   - Compare with previous model versions
   - Calculate key metrics (MAE, RMSE, accuracy)

4. **Deployment Process**
   - MLflow model registry for versioning
   - A/B testing for new model versions
   - Automated rollback if performance degrades

### Continuous Learning

1. **Feedback Integration**
   - Collect user corrections on predictions
   - Weight feedback based on recency and consistency
   - Incorporate into retraining process

2. **Monitoring and Alerts**
   - Track prediction accuracy in production
   - Detect model drift or degradation
   - Alert on significant accuracy drops

3. **Scheduled Retraining**
   - Weekly retraining of global models
   - Monthly retraining of user-specific models
   - On-demand retraining triggered by significant new data

## Caching and Performance Optimization

### Caching Strategy

1. **Redis Cache Implementation**
   - Cache prediction results for quick access
   - Cache frequently accessed user data
   - Store session data for authentication

2. **Cache Invalidation**
   - Time-based expiration for predictions
   - Event-based invalidation when new data is logged
   - Selective invalidation for specific data types

3. **Cache Hierarchy**
   - Frontend application cache (service worker)
   - API response cache (Redis)
   - Database query cache

### Performance Optimizations

1. **Database Optimization**
   - Indexing of frequently queried fields
   - Partitioning of historical data
   - Query optimization for complex joins

2. **API Performance**
   - Response compression
   - Pagination of large result sets
   - Asynchronous processing for non-critical operations

3. **ML Performance**
   - Model quantization for faster inference
   - Batch prediction processing
   - Caching of intermediate calculations

4. **Background Processing**
   - Queue non-urgent tasks (Laravel Horizon)
   - Prioritize critical user-facing operations
   - Scheduled tasks during off-peak hours

## Scaling Considerations

### Horizontal Scaling

1. **Laravel API Scaling**
   - Stateless design for horizontal scaling
   - Load balancing across multiple instances
   - Session management via Redis

2. **FastAPI ML Engine Scaling**
   - Multiple instances for prediction serving
   - Kubernetes-based orchestration
   - Autoscaling based on request load

3. **Database Scaling**
   - Read replicas for query distribution
   - Connection pooling
   - Sharding strategy for future growth

### Vertical Scaling

1. **Resource Optimization**
   - CPU optimization for ML workloads
   - Memory allocation for database operations
   - Disk I/O optimization for logging and storage

2. **Instance Sizing**
   - Tiered instance types based on workload
   - Reserved instances for baseline capacity
   - Burst capacity for peak loads

### Growth Planning

1. **Monitoring and Capacity Planning**
   - Track resource utilization trends
   - Forecast growth based on user acquisition
   - Implement alerts for capacity thresholds

2. **Geographic Distribution**
   - Regional deployments for global user base
   - Content delivery network for static assets
   - Data residency considerations

3. **Disaster Recovery**
   - Regular database backups
   - Multi-region failover capability
   - Recovery time objective (RTO) and recovery point objective (RPO) planning
