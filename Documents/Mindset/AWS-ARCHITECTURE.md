# MINDSET AWS Architecture

## Overview

MINDSET is a cloud-native, AI-driven news recommendation platform that provides transparency metrics for news articles. This document describes the AWS architecture used to deploy and run MINDSET.

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                            AWS Cloud                                                   │
│                                                                                                        │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐      │
│  │   Amazon S3      │     │  Amazon ECR      │     │   Amazon ECS      │     │   CloudWatch     │      │
│  │                  │     │                  │     │                   │     │                  │      │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │     │ ┌───────────────┐ │     │ ┌──────────────┐ │      │
│  │ │  Data Layers  │ │     │ │  Container   │ │     │ │ Frontend Task │ │     │ │  Dashboard   │ │      │
│  │ │              │ │     │ │  Registry    │ │     │ │               │ │     │ │              │ │      │
│  │ │  - Raw       │ │     │ │              │ │     │ └───────────────┘ │     │ └──────────────┘ │      │
│  │ │  - Bronze    │ │     │ │ - Backend    │ │     │ ┌───────────────┐ │     │ ┌──────────────┐ │      │
│  │ │  - Silver    │ │     │ │ - Frontend   │ │     │ │ Backend Task  │ │     │ │    Alarms    │ │      │
│  │ │  - Gold      │ │     │ │ - ML         │ │     │ │               │ │     │ │              │ │      │
│  │ └──────────────┘ │     │ └──────────────┘ │     │ └───────────────┘ │     │ └──────────────┘ │      │
│  └──────────────────┘     └──────────────────┘     │ ┌───────────────┐ │     └──────────────────┘      │
│           │                        │               │ │   ML Task     │ │              ▲                 │
│           ▼                        ▼               │ │               │ │              │                 │
│  ┌──────────────────┐     ┌──────────────────┐     │ └───────────────┘ │     ┌──────────────────┐      │
│  │  Parameter Store │     │  Load Balancer   │     └──────────────────┘     │   CloudTrail     │      │
│  │                  │     │                  │              │                │                  │      │
│  │ ┌──────────────┐ │     │ ┌──────────────┐ │              ▼                │ ┌──────────────┐ │      │
│  │ │   Secrets    │ │     │ │   Target     │ │     ┌──────────────────┐     │ │    Audit     │ │      │
│  │ │              │ │     │ │   Groups     │ │     │    Auto Scaling  │     │ │    Logs      │ │      │
│  │ │ - NewsAPI Key│ │     │ │              │ │     │                  │     │ │              │ │      │
│  │ │ - DB Creds   │ │     │ └──────────────┘ │     │  ┌────────────┐  │     │ └──────────────┘ │      │
│  │ └──────────────┘ │     └──────────────────┘     │  │  Scaling   │  │     └──────────────────┘      │
│  └──────────────────┘              │               │  │  Policies  │  │                               │
│           │                        │               │  └────────────┘  │                               │
│           │                        ▼               └──────────────────┘                               │
│           │               ┌──────────────────┐                                                         │
│           │               │    VPC & Subnets │                                                         │
│           ▼               │                  │                                                         │
│  ┌──────────────────┐     │ ┌──────────────┐ │                                                         │
│  │   IAM            │     │ │  Security    │ │                                                         │
│  │                  │     │ │  Groups      │ │                                                         │
│  │ ┌──────────────┐ │     │ │              │ │                                                         │
│  │ │   Roles &    │ │     │ └──────────────┘ │                                                         │
│  │ │   Policies   │ │     └──────────────────┘                                                         │
│  │ └──────────────┘ │                                                                                  │
│  └──────────────────┘                                                                                  │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Storage (S3)

Amazon S3 is used to store data in the medallion architecture pattern:

- **Raw Layer**: Original MINDLarge dataset and NewsAPI data
- **Bronze Layer**: Cleaned and standardized data
- **Silver Layer**: Feature-engineered data ready for ML processing
- **Gold Layer**: Processed data with transparency metrics

### 2. Compute Services (ECS)

Amazon ECS with Fargate is used to run containerized services:

- **Frontend Service**: React/Next.js application for user interface
- **Backend Service**: FastAPI application providing REST API endpoints
- **ML Service**: Machine learning service for article analysis and metrics calculation

### 3. Container Registry (ECR)

Amazon ECR stores Docker images for all components:

- Backend image
- Frontend image
- ML model image

### 4. Security & Configuration

- **Parameter Store**: Securely stores configuration and secrets
- **IAM Roles**: Provides least-privilege access to AWS resources
- **Security Groups**: Controls network traffic

### 5. Monitoring & Logging

- **CloudWatch**: Metrics, logs, and dashboards
- **CloudTrail**: Audit and compliance

## Data Flow

1. **Data Ingestion**:
   - MINDLarge dataset is uploaded to S3 raw layer
   - NewsAPI data is fetched in real-time and stored in the raw layer

2. **Data Processing Pipeline**:
   - Raw → Bronze: Data cleaning and standardization
   - Bronze → Silver: Feature engineering tailored for metrics
   - Silver → Gold: ML processing and metrics calculation

3. **Application Flow**:
   - Frontend requests article data from backend
   - Backend retrieves processed articles with metrics from S3 gold layer
   - Real-time articles are processed through the ML pipeline for metrics

## ML Components

### Silicon Layer

The Silicon Layer is a specialized processing layer that powers the advanced ML capabilities:

1. **Ensemble Models**: Combines multiple algorithms for improved accuracy
   - Bagging: Random Forest-based models
   - Boosting: XGBoost and LightGBM
   - Stacking: Meta-learning approach

2. **Explainable AI (XAI)**: Provides transparency into model decisions
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)

3. **Online Learning**: Enables real-time model updates with new data
   - Incremental learning algorithms
   - Feedback integration

4. **Feature Store**: Efficiently manages and reuses ML features
   - Feature versioning
   - Feature serving

5. **Drift Detection**: Monitors and adapts to changes in data patterns
   - Statistical drift detection
   - Automated retraining triggers

6. **Rust Integration**: High-performance components for critical processing
   - Native extensions via PyO3
   - Parallel processing for metrics calculation

## Deployment Architecture

The application is deployed using AWS CloudFormation for infrastructure as code with the following key resources:

- **VPC**: Isolated network with public and private subnets
- **Load Balancer**: Distributes traffic to services
- **ECS Cluster**: Runs containerized services with auto-scaling
- **Security Groups**: Controls network access
- **IAM Roles**: Provides necessary permissions
- **S3 Buckets**: Stores data in the medallion pattern
- **CloudWatch**: Monitors performance and health

## Security Considerations

1. **Data Protection**:
   - Encryption at rest using SSE-S3
   - Encryption in transit using HTTPS

2. **Access Control**:
   - Least privilege IAM policies
   - Role-based access control
   - Secure parameter storage

3. **Network Security**:
   - Security groups for traffic control
   - Private subnets for sensitive components

## Monitoring and Observability

1. **Performance Monitoring**:
   - CPU and memory utilization
   - Request latency and throughput

2. **Application Health**:
   - Service availability
   - Error rates and status codes

3. **Custom Metrics**:
   - ML model performance
   - Data processing pipeline metrics

4. **Alerting**:
   - CloudWatch alarms for critical metrics
   - Automated notification system

## Cost Optimization

1. **Fargate Spot** for non-critical workloads
2. **Auto-scaling** based on demand
3. **S3 Lifecycle Policies** for older data
4. **Reserved Instances** for predictable workloads

## Conclusion

The MINDSET AWS architecture provides a scalable, secure, and efficient platform for delivering AI-driven news transparency metrics. The system leverages AWS managed services to reduce operational overhead while maintaining high availability and performance.