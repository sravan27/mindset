#!/bin/bash
set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV=${1:-dev}
STACK_NAME="mindset-$ENV"
AWS_REGION=$(aws configure get region)
[ -z "$AWS_REGION" ] && AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
ECR_REPOSITORY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mindset-$ENV"

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRASTRUCTURE_DIR="$SCRIPT_DIR/infrastructure/aws"
SRC_DIR="$SCRIPT_DIR/src"
CFN_TEMPLATE="$INFRASTRUCTURE_DIR/cloudformation/mindset-infrastructure.yaml"
DOCKER_DIR="$SCRIPT_DIR/docker"

echo -e "${BLUE}=== MINDSET AWS Deployment ====${NC}"
echo -e "${BLUE}Environment: ${GREEN}$ENV${NC}"
echo -e "${BLUE}Region: ${GREEN}$AWS_REGION${NC}"
echo -e "${BLUE}Account ID: ${GREEN}$AWS_ACCOUNT_ID${NC}"

# Create directories if they don't exist
mkdir -p "$INFRASTRUCTURE_DIR/cloudformation"
mkdir -p "$SRC_DIR"
mkdir -p "$DOCKER_DIR"

# Check if AWS CLI is installed and configured
echo -e "${BLUE}Checking AWS CLI configuration...${NC}"
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}AWS CLI not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}AWS CLI is properly configured.${NC}"

# Deploy CloudFormation stack for infrastructure
deploy_infrastructure() {
    echo -e "${BLUE}Deploying infrastructure with CloudFormation...${NC}"
    
    if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" &> /dev/null; then
        echo -e "${YELLOW}Stack does not exist. Creating new stack...${NC}"
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME" \
            --template-body "file://$CFN_TEMPLATE" \
            --capabilities CAPABILITY_IAM \
            --parameters ParameterKey=EnvironmentName,ParameterValue="$ENV"
        
        echo -e "${BLUE}Waiting for stack creation to complete...${NC}"
        aws cloudformation wait stack-create-complete --stack-name "$STACK_NAME"
    else
        echo -e "${YELLOW}Stack exists. Updating stack...${NC}"
        
        # Using change-set to update the stack
        CHANGE_SET_NAME="$STACK_NAME-$(date +%Y%m%d%H%M%S)"
        
        aws cloudformation create-change-set \
            --stack-name "$STACK_NAME" \
            --change-set-name "$CHANGE_SET_NAME" \
            --template-body "file://$CFN_TEMPLATE" \
            --capabilities CAPABILITY_IAM \
            --parameters ParameterKey=EnvironmentName,ParameterValue="$ENV"
            
        # Wait for change-set creation
        aws cloudformation wait change-set-create-complete \
            --stack-name "$STACK_NAME" \
            --change-set-name "$CHANGE_SET_NAME"
            
        # Execute change-set
        aws cloudformation execute-change-set \
            --stack-name "$STACK_NAME" \
            --change-set-name "$CHANGE_SET_NAME"
            
        # Wait for stack update to complete
        aws cloudformation wait stack-update-complete --stack-name "$STACK_NAME"
    fi
    
    echo -e "${GREEN}Infrastructure deployment completed.${NC}"
}

# Get stack outputs as environment variables
get_stack_outputs() {
    echo -e "${BLUE}Getting CloudFormation stack outputs...${NC}"
    
    # Get the bucket names from CloudFormation outputs
    RAW_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='RawBucketName'].OutputValue" \
        --output text)
    
    BRONZE_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='BronzeBucketName'].OutputValue" \
        --output text)
    
    SILVER_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='SilverBucketName'].OutputValue" \
        --output text)
    
    GOLD_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='GoldBucketName'].OutputValue" \
        --output text)
    
    MODELS_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='ModelsBucketName'].OutputValue" \
        --output text)
    
    ECR_REPO=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='ECRRepository'].OutputValue" \
        --output text)
    
    ALB_DNS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --query "Stacks[0].Outputs[?OutputKey=='ALBDNSName'].OutputValue" \
        --output text)
    
    echo -e "${GREEN}Successfully retrieved stack outputs.${NC}"
}

# Upload MINDLarge dataset to S3
upload_data() {
    echo -e "${BLUE}Uploading MINDLarge dataset to S3...${NC}"
    
    # Check if the RAW_BUCKET is defined
    if [ -z "$RAW_BUCKET" ]; then
        echo -e "${RED}RAW_BUCKET is not defined. Please run get_stack_outputs first.${NC}"
        return 1
    fi
    
    # Upload train data
    if [ -d "$SCRIPT_DIR/MINDlarge_train" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_train dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_train" "s3://$RAW_BUCKET/MINDlarge_train" --recursive
    else
        echo -e "${RED}MINDlarge_train directory not found. Skipping...${NC}"
    fi
    
    # Upload dev data
    if [ -d "$SCRIPT_DIR/MINDlarge_dev" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_dev dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_dev" "s3://$RAW_BUCKET/MINDlarge_dev" --recursive
    else
        echo -e "${RED}MINDlarge_dev directory not found. Skipping...${NC}"
    fi
    
    # Upload test data
    if [ -d "$SCRIPT_DIR/MINDlarge_test" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_test dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_test" "s3://$RAW_BUCKET/MINDlarge_test" --recursive
    else
        echo -e "${RED}MINDlarge_test directory not found. Skipping...${NC}"
    fi
    
    echo -e "${GREEN}Data upload completed.${NC}"
}

# Build and push Docker image
build_and_push_docker() {
    echo -e "${BLUE}Building and pushing Docker image...${NC}"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Docker not found. Please install Docker first.${NC}"
        return 1
    fi
    
    # Check if we can authenticate to ECR
    echo -e "${YELLOW}Authenticating to ECR...${NC}"
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Check if the Dockerfile exists
    if [ ! -f "$DOCKER_DIR/Dockerfile" ]; then
        echo -e "${RED}Dockerfile not found at $DOCKER_DIR/Dockerfile${NC}"
        return 1
    fi
    
    # Build the Docker image
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t "mindset:latest" -f "$DOCKER_DIR/Dockerfile" "$SCRIPT_DIR"
    
    # Tag the image for ECR
    echo -e "${YELLOW}Tagging Docker image...${NC}"
    docker tag "mindset:latest" "$ECR_REPO:latest"
    
    # Push the image to ECR
    echo -e "${YELLOW}Pushing Docker image to ECR...${NC}"
    docker push "$ECR_REPO:latest"
    
    echo -e "${GREEN}Docker image built and pushed successfully.${NC}"
}

# Run the data processing pipeline
run_pipeline() {
    echo -e "${BLUE}Running data processing pipeline...${NC}"
    
    # Check if the run_pipeline_aws.py script exists
    if [ ! -f "$SRC_DIR/run_pipeline_aws.py" ]; then
        echo -e "${RED}Pipeline script not found at $SRC_DIR/run_pipeline_aws.py${NC}"
        return 1
    fi
    
    # Set environment variables for the pipeline
    export S3_RAW_BUCKET="$RAW_BUCKET"
    export S3_BRONZE_BUCKET="$BRONZE_BUCKET"
    export S3_SILVER_BUCKET="$SILVER_BUCKET"
    export S3_GOLD_BUCKET="$GOLD_BUCKET"
    export S3_MODELS_BUCKET="$MODELS_BUCKET"
    export ENVIRONMENT="$ENV"
    
    # Run the pipeline
    echo -e "${YELLOW}Executing pipeline script...${NC}"
    python "$SRC_DIR/run_pipeline_aws.py"
    
    echo -e "${GREEN}Pipeline execution completed.${NC}"
}

# Deploy the ECS service
deploy_service() {
    echo -e "${BLUE}Deploying ECS service...${NC}"
    
    # Get cluster name
    CLUSTER_NAME="$ENV-mindset-cluster"
    SERVICE_NAME="$ENV-mindset-service"
    
    # Check if service exists and update
    if aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$SERVICE_NAME" --query "services[?status!='INACTIVE'] | [0]" --output text &> /dev/null; then
        echo -e "${YELLOW}Updating ECS service...${NC}"
        aws ecs update-service --cluster "$CLUSTER_NAME" --service "$SERVICE_NAME" --force-new-deployment
    else
        echo -e "${YELLOW}Service not found or inactive. CloudFormation should have created it.${NC}"
    fi
    
    echo -e "${GREEN}ECS service deployment triggered.${NC}"
}

# Main deployment flow
main() {
    echo -e "${BLUE}Starting MINDSET deployment process for $ENV environment...${NC}"
    
    deploy_infrastructure
    get_stack_outputs
    upload_data
    
    # Create local Docker directory and Dockerfile if they don't exist
    if [ ! -f "$DOCKER_DIR/Dockerfile" ]; then
        echo -e "${YELLOW}Creating Dockerfile...${NC}"
        mkdir -p "$DOCKER_DIR"
        cat > "$DOCKER_DIR/Dockerfile" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ /app/src/
COPY docker/entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app

# Expose port for the API
EXPOSE 80

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
EOF
    fi
    
    # Create entrypoint script if it doesn't exist
    if [ ! -f "$DOCKER_DIR/entrypoint.sh" ]; then
        echo -e "${YELLOW}Creating Docker entrypoint script...${NC}"
        mkdir -p "$DOCKER_DIR"
        cat > "$DOCKER_DIR/entrypoint.sh" << 'EOF'
#!/bin/bash
set -e

# Start the API server by default
if [ "$1" = "api" ] || [ -z "$1" ]; then
    echo "Starting API server..."
    uvicorn src.api.main:app --host 0.0.0.0 --port 80
    
# Run the pipeline if requested
elif [ "$1" = "pipeline" ]; then
    echo "Running full data pipeline..."
    python src/run_pipeline_aws.py
    
# Run silico layer processing only
elif [ "$1" = "silicon" ]; then
    echo "Running Silicon Layer processing..."
    python src/ml/silicon_layer.py

# Run a custom command
else
    exec "$@"
fi
EOF
        chmod +x "$DOCKER_DIR/entrypoint.sh"
    fi
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        echo -e "${YELLOW}Creating requirements.txt...${NC}"
        cat > "$SCRIPT_DIR/requirements.txt" << 'EOF'
# API
fastapi==0.95.0
uvicorn==0.21.1
pydantic==1.10.7

# AWS
boto3==1.26.115
awswrangler==3.1.1

# Data Processing
pandas==2.0.0
numpy==1.24.2
pyarrow==12.0.0
duckdb==0.7.1
dask==2023.3.2
distributed==2023.3.2

# ML
scikit-learn==1.2.2
xgboost==1.7.5
lightgbm==3.3.5
shap==0.41.0
lime==0.2.0.1
mlflow==2.3.0
river==0.14.0
feature-engine==1.5.2

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Utilities
tqdm==4.65.0
python-dotenv==1.0.0
pytest==7.3.1
maturin==0.14.17
EOF
    fi
    
    build_and_push_docker
    run_pipeline
    deploy_service
    
    echo -e "${GREEN}===== MINDSET deployment completed successfully! =====\
${NC}"
    echo -e "${BLUE}Access the application at:${GREEN} http://$ALB_DNS${NC}"
    echo -e "${BLUE}Raw data bucket:${GREEN} $RAW_BUCKET${NC}"
    echo -e "${BLUE}Bronze data bucket:${GREEN} $BRONZE_BUCKET${NC}"
    echo -e "${BLUE}Silver data bucket:${GREEN} $SILVER_BUCKET${NC}"
    echo -e "${BLUE}Gold data bucket:${GREEN} $GOLD_BUCKET${NC}"
    echo -e "${BLUE}ML models bucket:${GREEN} $MODELS_BUCKET${NC}"
}

# Execute main function
main