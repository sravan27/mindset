#!/bin/bash
# MINDSET AWS Deployment Script - Simplified Version

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Welcome banner
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  MINDSET - AWS Deployment (Simplified)          ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Starting deployment to AWS...${NC}"

# Check required commands
echo -e "${YELLOW}Checking required commands...${NC}"
REQUIRED_COMMANDS=("aws" "python3" "pip")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd command is required but not found${NC}"
        echo -e "${YELLOW}Please install $cmd and try again${NC}"
        exit 1
    fi
done
echo -e "${GREEN}All required commands are available${NC}"

# Get AWS credentials if not already configured
echo -e "${YELLOW}Checking AWS credentials...${NC}"
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${YELLOW}Please configure AWS credentials:${NC}"
    echo -e "${YELLOW}AWS Access Key ID:${NC}"
    read -p "> " AWS_ACCESS_KEY_ID
    echo -e "${YELLOW}AWS Secret Access Key:${NC}"
    read -p "> " AWS_SECRET_ACCESS_KEY
    echo -e "${YELLOW}Default region name [us-east-1]:${NC}"
    read -p "> " AWS_REGION
    AWS_REGION=${AWS_REGION:-us-east-1}
    
    # Configure AWS CLI
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set region "$AWS_REGION"
else
    # Get current region
    AWS_REGION=$(aws configure get region)
    echo -e "${GREEN}AWS credentials already configured. Using region: ${AWS_REGION}${NC}"
fi

# Get NewsAPI key
echo -e "${YELLOW}Please provide your NewsAPI.org API key:${NC}"
read -p "> " NEWSAPI_KEY
export NEWSAPI_KEY="$NEWSAPI_KEY"

# Create a unique identifier for resources
TIMESTAMP=$(date +%Y%m%d%H%M%S)
RANDOM_STRING=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | fold -w 8 | head -n 1)
RESOURCE_PREFIX="mindset-${RANDOM_STRING}"

# Step 1: Create S3 bucket for data
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 1: Creating S3 bucket${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create bucket with unique name
S3_BUCKET_DATA="${RESOURCE_PREFIX}-data"
echo -e "${YELLOW}Creating S3 bucket: ${S3_BUCKET_DATA}${NC}"
aws s3api create-bucket \
    --bucket ${S3_BUCKET_DATA} \
    --region ${AWS_REGION} \
    $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=${AWS_REGION}"; fi)

# Create directories for data layers within the bucket
for layer in raw bronze silver gold datasets; do
    echo -e "${YELLOW}Creating S3 directory: ${layer}${NC}"
    aws s3api put-object --bucket ${S3_BUCKET_DATA} --key ${layer}/
done

# Step 2: Upload MINDLarge datasets to S3
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 2: Uploading datasets to S3${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Upload MINDLarge datasets
for dir in MINDlarge_train MINDlarge_dev MINDlarge_test; do
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Uploading $dir...${NC}"
        aws s3 cp $dir s3://${S3_BUCKET_DATA}/datasets/$dir/ --recursive
    fi
done

# Step 3: Store configuration
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 3: Storing configuration${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Store NewsAPI key securely
echo -e "${YELLOW}Storing NewsAPI key in Parameter Store...${NC}"
aws ssm put-parameter \
    --name /${RESOURCE_PREFIX}/newsapi-key \
    --value ${NEWSAPI_KEY} \
    --type SecureString \
    --overwrite

# Store S3 bucket name
aws ssm put-parameter \
    --name /${RESOURCE_PREFIX}/s3-bucket \
    --value ${S3_BUCKET_DATA} \
    --type String \
    --overwrite

# Create a file with configuration information
cat > .env.aws << EOF
# MINDSET AWS Configuration

# Resource Prefix
RESOURCE_PREFIX=${RESOURCE_PREFIX}

# S3 Bucket
S3_BUCKET_DATA=${S3_BUCKET_DATA}

# AWS Region
AWS_REGION=${AWS_REGION}

# NewsAPI Key
NEWSAPI_KEY=${NEWSAPI_KEY}
EOF

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}     DEPLOYMENT COMPLETE!            ${NC}"
echo -e "${GREEN}=====================================${NC}"

echo -e "${YELLOW}MINDSET data layer has been deployed to AWS!${NC}"
echo -e ""
echo -e "${YELLOW}S3 Data Bucket:${NC}"
echo -e "${GREEN}${S3_BUCKET_DATA}${NC}"
echo -e ""
echo -e "${YELLOW}Configuration saved to:${NC}"
echo -e "${GREEN}.env.aws${NC}"
echo -e ""
echo -e "${YELLOW}To view your data in AWS S3:${NC}"
echo -e "${GREEN}https://s3.console.aws.amazon.com/s3/buckets/${S3_BUCKET_DATA}${NC}"
echo -e ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. ${GREEN}Run the data pipeline: python run_pipeline.py${NC}"
echo -e "2. ${GREEN}Deploy the web application with: python deploy_app.py${NC}"
echo -e ""
echo -e "${BLUE}Thank you for using MINDSET on AWS!${NC}"