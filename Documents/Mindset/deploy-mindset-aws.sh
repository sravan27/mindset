#!/bin/bash
# MINDSET AWS Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Welcome banner
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  MINDSET - AWS Deployment                        ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Starting deployment to AWS...${NC}"

# Check required commands
echo -e "${YELLOW}Checking required commands...${NC}"
REQUIRED_COMMANDS=("aws" "docker" "python" "pip")
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
RANDOM_STRING=$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
RESOURCE_PREFIX="mindset-${RANDOM_STRING}"

# Step 1: Set up environment
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 1: Setting up environment${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create Python virtual environment if needed
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-aws.txt
else
    echo -e "${YELLOW}Using existing virtual environment...${NC}"
    source venv/bin/activate
    # Ensure requirements are up to date
    pip install -r requirements-aws.txt
fi

# Step 2: Create S3 buckets for data layers
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 2: Creating S3 buckets${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create buckets with unique names
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

# Step 3: Upload MINDLarge datasets to S3
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 3: Uploading datasets to S3${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Upload MINDLarge datasets
for dir in MINDlarge_train MINDlarge_dev MINDlarge_test; do
    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Uploading $dir...${NC}"
        aws s3 cp $dir s3://${S3_BUCKET_DATA}/datasets/$dir/ --recursive
    fi
done

# Step 4: Set up ECR for Docker images
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 4: Setting up ECR repositories${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create ECR repositories for our images
for repo in mindset-backend mindset-frontend mindset-ml; do
    echo -e "${YELLOW}Creating ECR repository: ${repo}${NC}"
    aws ecr describe-repositories --repository-names ${repo} &> /dev/null || \
    aws ecr create-repository --repository-name ${repo}
done

# Get the ECR registry URL
ECR_REGISTRY=$(aws ecr describe-repositories --repository-names mindset-backend --query 'repositories[0].repositoryUri' --output text | sed 's|/.*||')

# Step 5: Build and push Docker images
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 5: Building and pushing Docker images${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Login to ECR
echo -e "${YELLOW}Logging in to ECR...${NC}"
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Build and push backend image
echo -e "${YELLOW}Building and pushing backend image...${NC}"
docker build -t ${ECR_REGISTRY}/mindset-backend:latest -f infrastructure/docker/backend.Dockerfile .
docker push ${ECR_REGISTRY}/mindset-backend:latest

# Build and push frontend image
echo -e "${YELLOW}Building and pushing frontend image...${NC}"
docker build -t ${ECR_REGISTRY}/mindset-frontend:latest -f infrastructure/docker/frontend.Dockerfile .
docker push ${ECR_REGISTRY}/mindset-frontend:latest

# Build and push ML image
echo -e "${YELLOW}Building and pushing ML image...${NC}"
docker build -t ${ECR_REGISTRY}/mindset-ml:latest -f infrastructure/docker/ml.Dockerfile .
docker push ${ECR_REGISTRY}/mindset-ml:latest

# Step 6: Create ECS cluster and service
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 6: Creating ECS resources${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create ECS cluster
echo -e "${YELLOW}Creating ECS cluster: ${RESOURCE_PREFIX}-cluster${NC}"
aws ecs create-cluster --cluster-name ${RESOURCE_PREFIX}-cluster

# Create task execution role and policy
echo -e "${YELLOW}Creating IAM roles for ECS...${NC}"
aws iam create-role \
    --role-name ${RESOURCE_PREFIX}-ecs-execution-role \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam attach-role-policy \
    --role-name ${RESOURCE_PREFIX}-ecs-execution-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create task role for S3 access
aws iam create-role \
    --role-name ${RESOURCE_PREFIX}-ecs-task-role \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

aws iam put-role-policy \
    --role-name ${RESOURCE_PREFIX}-ecs-task-role \
    --policy-name S3Access \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::'${S3_BUCKET_DATA}'",
                    "arn:aws:s3:::'${S3_BUCKET_DATA}'/*"
                ]
            }
        ]
    }'

# Create security group for the services
echo -e "${YELLOW}Creating security group for ECS services...${NC}"
VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 create-security-group \
    --group-name ${RESOURCE_PREFIX}-sg \
    --description "Security group for MINDSET ECS services" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' --output text)

# Allow HTTP and HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

# Step 7: Create and store configuration in Parameter Store
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 7: Storing configuration${NC}"
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

# ECR Repository
ECR_REGISTRY=${ECR_REGISTRY}

# ECS Cluster
ECS_CLUSTER=${RESOURCE_PREFIX}-cluster

# IAM Roles
ECS_EXECUTION_ROLE=${RESOURCE_PREFIX}-ecs-execution-role
ECS_TASK_ROLE=${RESOURCE_PREFIX}-ecs-task-role

# Security Group
SECURITY_GROUP_ID=${SG_ID}
EOF

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}     DEPLOYMENT COMPLETE!            ${NC}"
echo -e "${GREEN}=====================================${NC}"

echo -e "${YELLOW}MINDSET has been deployed to AWS!${NC}"
echo -e ""
echo -e "${YELLOW}S3 Data Bucket:${NC}"
echo -e "${GREEN}${S3_BUCKET_DATA}${NC}"
echo -e ""
echo -e "${YELLOW}ECS Cluster:${NC}"
echo -e "${GREEN}${RESOURCE_PREFIX}-cluster${NC}"
echo -e ""
echo -e "${YELLOW}Configuration saved to:${NC}"
echo -e "${GREEN}.env.aws${NC}"
echo -e ""
echo -e "${YELLOW}To view your data in AWS S3:${NC}"
echo -e "${GREEN}https://s3.console.aws.amazon.com/s3/buckets/${S3_BUCKET_DATA}${NC}"
echo -e ""
echo -e "${YELLOW}To access ECS:${NC}"
echo -e "${GREEN}https://console.aws.amazon.com/ecs/home?region=${AWS_REGION}#/clusters/${RESOURCE_PREFIX}-cluster${NC}"
echo -e ""
echo -e "${BLUE}Thank you for using MINDSET on AWS!${NC}"