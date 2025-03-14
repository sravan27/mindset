# MINDSET AWS Deployment Guide

## Overview

MINDSET is an AI-driven news recommendation system with transparency metrics that help users understand the nature of news content. The system provides three key transparency metrics:

1. **Political Influence Level**: Measures the political bias of an article on a 0-10 scale
2. **Rhetoric Intensity Scale**: Quantifies the emotional language used in an article on a 0-10 scale
3. **Information Depth Score**: Categorizes article depth as Shallow, Moderate, or Deep with a 0-10 score

This guide covers the AWS deployment of MINDSET using the provided scripts and CloudFormation templates.

## Architecture

MINDSET uses a medallion architecture with a unique Silicon Layer for ML operations:

- **Raw**: Original source data (MINDLarge dataset, NewsAPI)
- **Bronze**: Validated and cleaned data
- **Silver**: Feature-engineered data ready for ML
- **Silicon**: Advanced ML layer that adds transparency metrics using ensemble learning, XAI, and drift detection
- **Gold**: Final data products including recommendations and visualizations

The AWS deployment uses the following services:

- **S3**: For data storage across all medallion layers
- **ECS/Fargate**: For containerized application services
- **ECR**: For Docker container registry
- **CloudWatch**: For monitoring and logging
- **SageMaker**: For ML model development and training
- **Application Load Balancer**: For API access

## Prerequisites

- AWS CLI installed and configured
- Docker installed (for local testing)
- The MINDLarge dataset directories in your project folder:
  - `MINDlarge_train/`
  - `MINDlarge_dev/`
  - `MINDlarge_test/`

## Deployment Steps

### 1. Set Up Data Buckets and Upload Data

```bash
# Make scripts executable
chmod +x deploy-mindset-aws-data.sh
chmod +x deploy-aws-master.sh

# Deploy data first (creates S3 buckets and uploads MINDLarge dataset)
./deploy-mindset-aws-data.sh dev
```

This script:
- Creates S3 buckets for Raw, Bronze, Silver, Gold, and Models data
- Validates the MINDLarge dataset structure
- Uploads the dataset to the Raw S3 bucket
- Creates the folder structure in all buckets

### 2. Deploy the Full Infrastructure

```bash
# Deploy the full infrastructure using CloudFormation
./deploy-aws-master.sh dev
```

This master script:
1. Deploys the CloudFormation stack with all AWS resources
2. Retrieves the S3 bucket names and other resources from CloudFormation outputs
3. Uploads the MINDLarge dataset to the Raw bucket (if not already done)
4. Builds and pushes the Docker image to ECR
5. Runs the data processing pipeline
6. Deploys the ECS service for the API

## AWS Resources Created

- **VPC with public and private subnets**
- **S3 Buckets**:
  - `mindset-raw-{account-id}-{env}`: Raw data
  - `mindset-bronze-{account-id}-{env}`: Bronze data
  - `mindset-silver-{account-id}-{env}`: Silver data  
  - `mindset-gold-{account-id}-{env}`: Gold data
  - `mindset-models-{account-id}-{env}`: ML models
- **ECR Repository**: `mindset-{env}`
- **ECS Cluster and Service**: Running the MINDSET application
- **Application Load Balancer**: Exposing the API
- **SageMaker Notebook**: For ML development and experimentation

## Testing the Deployment

After deployment completes, you can test the API:

```bash
# Get the ALB DNS name
ALB_DNS=$(aws cloudformation describe-stacks \
    --stack-name mindset-dev \
    --query "Stacks[0].Outputs[?OutputKey=='ALBDNSName'].OutputValue" \
    --output text)

# Test the health endpoint
curl http://$ALB_DNS/health

# Get recommendations
curl http://$ALB_DNS/recommendations

# Get metrics summary
curl http://$ALB_DNS/metrics/summary
```

## The Silicon Layer

The Silicon Layer is a critical component of MINDSET that provides advanced ML capabilities:

1. **Feature Store**: Efficient management and versioning of ML features
2. **Ensemble Learning**: Combining multiple models for better predictions
3. **Explainable AI (XAI)**: Making model predictions transparent
4. **Drift Detection**: Monitoring for changes in data distributions
5. **Online Learning**: Real-time model adaptation

This layer transforms standard ML-ready data into enhanced data with rich transparency metrics.

## Rust Metrics Engine

MINDSET uses a high-performance Rust-based metrics calculation engine for computing the transparency metrics. The engine provides:

1. Parallelized processing for large volumes of news articles
2. Efficient text analysis algorithms
3. Python integration through PyO3 bindings

If the Rust engine is not available, the system falls back to pure Python implementations.

## Local Development

You can run MINDSET locally for development:

```bash
# Create directory structure
mkdir -p infrastructure/aws/cloudformation
mkdir -p src/ml
mkdir -p src/api
mkdir -p src/rust/metrics_engine
mkdir -p docker

# Build the Docker image locally
docker build -t mindset:local -f docker/Dockerfile .

# Run the container locally
docker run -p 8080:80 \
  -e S3_RAW_BUCKET=mindset-raw-{account-id}-dev \
  -e S3_BRONZE_BUCKET=mindset-bronze-{account-id}-dev \
  -e S3_SILVER_BUCKET=mindset-silver-{account-id}-dev \
  -e S3_GOLD_BUCKET=mindset-gold-{account-id}-dev \
  -e S3_MODELS_BUCKET=mindset-models-{account-id}-dev \
  -e ENVIRONMENT=dev \
  mindset:local
```

## Cleaning Up

To delete all AWS resources:

```bash
# Delete the CloudFormation stack
aws cloudformation delete-stack --stack-name mindset-dev

# Empty and delete S3 buckets
# Replace {account-id} with your AWS account ID
aws s3 rb s3://mindset-raw-{account-id}-dev --force
aws s3 rb s3://mindset-bronze-{account-id}-dev --force
aws s3 rb s3://mindset-silver-{account-id}-dev --force
aws s3 rb s3://mindset-gold-{account-id}-dev --force
aws s3 rb s3://mindset-models-{account-id}-dev --force
```

## Troubleshooting

- **CloudFormation stack creation fails**: Check the CloudFormation events for error details
- **Data pipeline fails**: Check CloudWatch logs for the ECS task
- **API not accessible**: Verify the security groups allow traffic to the ALB

## Next Steps

- Implement the frontend application using React/Next.js with Tailwind CSS
- Set up CI/CD pipelines with AWS CodePipeline
- Enhance the Rust metrics engine with more sophisticated algorithms
- Implement advanced features of the Silicon Layer (online learning, full XAI)