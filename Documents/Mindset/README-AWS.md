# MINDSET on AWS

This guide will help you deploy and run MINDSET on AWS.

## Prerequisites

1. AWS CLI installed and configured with your credentials
2. Python 3.8+ with pip
3. NewsAPI.org API key

## Deployment Steps

### 1. Deploy Infrastructure to AWS

```bash
./deploy-mindset-aws-simple.sh
```

This script will:
- Configure AWS CLI if not already done
- Create S3 buckets for data layers (raw, bronze, silver, gold)
- Upload MINDLarge datasets to S3
- Store configuration in AWS Parameter Store
- Create a local .env.aws file with your configuration

### 2. Run the Data Pipeline

```bash
python run_pipeline_aws.py
```

This will:
- Load data from the S3 raw layer (MINDLarge datasets)
- Process it through the medallion architecture:
  - Raw to Bronze: Data cleaning
  - Bronze to Silver: Feature engineering
  - Silver to Gold: Add transparency metrics
- Store the processed data back in S3

### 3. Deploy the Web Application

```bash
python deploy_app.py
```

This will:
- Create a Streamlit application for visualizing the data
- Install required dependencies
- Start the web server locally

You'll be able to access the web application at http://localhost:8501

## Application Features

MINDSET provides:

1. **Dashboard** - Overview of news articles and metrics
2. **News Articles** - Browse and search articles with transparency metrics
3. **Analytics** - Detailed analysis of metrics and categories
4. **About** - Information about the platform

## Transparency Metrics

MINDSET analyzes news articles and provides three key transparency metrics:

1. **Political Influence Level (0-100)** - Measures how politically charged the content is
2. **Rhetoric Intensity Scale (0-100)** - Quantifies the use of rhetorical devices and emotional language
3. **Information Depth Score (0-100)** - Evaluates the depth and substance of the information provided

## Architecture

MINDSET follows a medallion architecture with AWS services:

- **Storage**: AWS S3 for all data layers
- **Security**: AWS Parameter Store for secrets
- **Processing**: Python data pipeline
- **Visualization**: Streamlit web application

## Troubleshooting

If you encounter issues:

1. Check that your AWS credentials are correctly configured
2. Ensure your S3 bucket names are correctly set in .env.aws
3. Verify that the MINDLarge datasets were uploaded correctly
4. Check for any error messages in the command output

For more detailed assistance, contact the MINDSET team.