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
AWS_REGION=$(aws configure get region)
[ -z "$AWS_REGION" ] && AWS_REGION="us-east-1"
STACK_NAME="mindset-$ENV"

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}=== MINDSET AWS Data Deployment ====${NC}"
echo -e "${BLUE}Environment: ${GREEN}$ENV${NC}"
echo -e "${BLUE}Region: ${GREEN}$AWS_REGION${NC}"

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

# Check if the CloudFormation stack exists
if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" &> /dev/null; then
    echo -e "${YELLOW}CloudFormation stack $STACK_NAME does not exist.${NC}"
    echo -e "${YELLOW}Creating S3 buckets directly...${NC}"
    
    # Generate bucket names based on account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
    RAW_BUCKET="mindset-raw-$AWS_ACCOUNT_ID-$ENV"
    BRONZE_BUCKET="mindset-bronze-$AWS_ACCOUNT_ID-$ENV"
    SILVER_BUCKET="mindset-silver-$AWS_ACCOUNT_ID-$ENV"
    GOLD_BUCKET="mindset-gold-$AWS_ACCOUNT_ID-$ENV"
    MODELS_BUCKET="mindset-models-$AWS_ACCOUNT_ID-$ENV"
    
    # Create the buckets
    echo -e "${BLUE}Creating S3 buckets...${NC}"
    aws s3api create-bucket --bucket "$RAW_BUCKET" --region "$AWS_REGION" $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$AWS_REGION"; fi)
    aws s3api create-bucket --bucket "$BRONZE_BUCKET" --region "$AWS_REGION" $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$AWS_REGION"; fi)
    aws s3api create-bucket --bucket "$SILVER_BUCKET" --region "$AWS_REGION" $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$AWS_REGION"; fi)
    aws s3api create-bucket --bucket "$GOLD_BUCKET" --region "$AWS_REGION" $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$AWS_REGION"; fi)
    aws s3api create-bucket --bucket "$MODELS_BUCKET" --region "$AWS_REGION" $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=$AWS_REGION"; fi)
    
    # Enable versioning
    echo -e "${BLUE}Enabling versioning for all buckets...${NC}"
    aws s3api put-bucket-versioning --bucket "$RAW_BUCKET" --versioning-configuration Status=Enabled
    aws s3api put-bucket-versioning --bucket "$BRONZE_BUCKET" --versioning-configuration Status=Enabled
    aws s3api put-bucket-versioning --bucket "$SILVER_BUCKET" --versioning-configuration Status=Enabled
    aws s3api put-bucket-versioning --bucket "$GOLD_BUCKET" --versioning-configuration Status=Enabled
    aws s3api put-bucket-versioning --bucket "$MODELS_BUCKET" --versioning-configuration Status=Enabled
else
    echo -e "${GREEN}CloudFormation stack $STACK_NAME exists.${NC}"
    
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
fi

echo -e "${BLUE}Using the following S3 buckets:${NC}"
echo -e "Raw bucket: ${GREEN}$RAW_BUCKET${NC}"
echo -e "Bronze bucket: ${GREEN}$BRONZE_BUCKET${NC}"
echo -e "Silver bucket: ${GREEN}$SILVER_BUCKET${NC}"
echo -e "Gold bucket: ${GREEN}$GOLD_BUCKET${NC}"
echo -e "Models bucket: ${GREEN}$MODELS_BUCKET${NC}"

# Upload MINDLarge dataset to S3
upload_mindlarge_data() {
    echo -e "${BLUE}Uploading MINDLarge dataset to S3...${NC}"
    
    # Upload train data
    if [ -d "$SCRIPT_DIR/MINDlarge_train" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_train dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_train" "s3://$RAW_BUCKET/MINDlarge_train" --recursive
        echo -e "${GREEN}MINDlarge_train upload complete!${NC}"
    else
        echo -e "${RED}MINDlarge_train directory not found at $SCRIPT_DIR/MINDlarge_train${NC}"
    fi
    
    # Upload dev data
    if [ -d "$SCRIPT_DIR/MINDlarge_dev" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_dev dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_dev" "s3://$RAW_BUCKET/MINDlarge_dev" --recursive
        echo -e "${GREEN}MINDlarge_dev upload complete!${NC}"
    else
        echo -e "${RED}MINDlarge_dev directory not found at $SCRIPT_DIR/MINDlarge_dev${NC}"
    fi
    
    # Upload test data
    if [ -d "$SCRIPT_DIR/MINDlarge_test" ]; then
        echo -e "${YELLOW}Uploading MINDlarge_test dataset...${NC}"
        aws s3 cp "$SCRIPT_DIR/MINDlarge_test" "s3://$RAW_BUCKET/MINDlarge_test" --recursive
        echo -e "${GREEN}MINDlarge_test upload complete!${NC}"
    else
        echo -e "${RED}MINDlarge_test directory not found at $SCRIPT_DIR/MINDlarge_test${NC}"
    fi
}

# Check dataset files
validate_mindlarge_datasets() {
    echo -e "${BLUE}Validating MINDLarge datasets...${NC}"
    local all_datasets_found=true
    
    for dataset in "MINDlarge_train" "MINDlarge_dev" "MINDlarge_test"; do
        if [ ! -d "$SCRIPT_DIR/$dataset" ]; then
            echo -e "${RED}$dataset directory not found at $SCRIPT_DIR/$dataset${NC}"
            all_datasets_found=false
            continue
        fi
        
        # Check for required files in each dataset
        for file in "behaviors.tsv" "news.tsv"; do
            if [ ! -f "$SCRIPT_DIR/$dataset/$file" ]; then
                echo -e "${RED}$file not found in $dataset${NC}"
                all_datasets_found=false
            fi
        done
    done
    
    if [ "$all_datasets_found" = false ]; then
        echo -e "${RED}Some dataset files are missing. Please ensure all required MINDLarge datasets are available.${NC}"
        echo -e "${YELLOW}You should have the following structure:${NC}"
        echo -e "${YELLOW}MINDlarge_train/behaviors.tsv${NC}"
        echo -e "${YELLOW}MINDlarge_train/news.tsv${NC}"
        echo -e "${YELLOW}MINDlarge_dev/behaviors.tsv${NC}"
        echo -e "${YELLOW}MINDlarge_dev/news.tsv${NC}"
        echo -e "${YELLOW}MINDlarge_test/behaviors.tsv${NC}"
        echo -e "${YELLOW}MINDlarge_test/news.tsv${NC}"
        return 1
    else
        echo -e "${GREEN}All required MINDLarge dataset files found!${NC}"
        return 0
    fi
}

# Create initial folder structure in S3
create_s3_folder_structure() {
    echo -e "${BLUE}Creating folder structure in S3 buckets...${NC}"
    
    # Raw bucket: Keep original structure by dataset
    # Already done by the upload_mindlarge_data function
    
    # Bronze bucket: Structure by data type and partition
    echo -e "${YELLOW}Creating Bronze bucket structure...${NC}"
    aws s3api put-object --bucket "$BRONZE_BUCKET" --key "news/"
    aws s3api put-object --bucket "$BRONZE_BUCKET" --key "behaviors/"
    aws s3api put-object --bucket "$BRONZE_BUCKET" --key "embeddings/"
    
    # Silver bucket: Structure for ML-ready data
    echo -e "${YELLOW}Creating Silver bucket structure...${NC}"
    aws s3api put-object --bucket "$SILVER_BUCKET" --key "features/"
    aws s3api put-object --bucket "$SILVER_BUCKET" --key "prepared/"
    aws s3api put-object --bucket "$SILVER_BUCKET" --key "user_profiles/"
    aws s3api put-object --bucket "$SILVER_BUCKET" --key "news_profiles/"
    
    # Gold bucket: Structure for final output data
    echo -e "${YELLOW}Creating Gold bucket structure...${NC}"
    aws s3api put-object --bucket "$GOLD_BUCKET" --key "recommendations/"
    aws s3api put-object --bucket "$GOLD_BUCKET" --key "metrics/"
    aws s3api put-object --bucket "$GOLD_BUCKET" --key "news_analytics/"
    aws s3api put-object --bucket "$GOLD_BUCKET" --key "user_analytics/"
    
    # Models bucket: Structure for model files
    echo -e "${YELLOW}Creating Models bucket structure...${NC}"
    aws s3api put-object --bucket "$MODELS_BUCKET" --key "recommendation_models/"
    aws s3api put-object --bucket "$MODELS_BUCKET" --key "political_influence_models/"
    aws s3api put-object --bucket "$MODELS_BUCKET" --key "rhetoric_intensity_models/"
    aws s3api put-object --bucket "$MODELS_BUCKET" --key "information_depth_models/"
    aws s3api put-object --bucket "$MODELS_BUCKET" --key "feature_store/"
    
    echo -e "${GREEN}S3 folder structure created successfully!${NC}"
}

# Main function
main() {
    echo -e "${BLUE}Starting MINDSET AWS data deployment...${NC}"
    
    if validate_mindlarge_datasets; then
        upload_mindlarge_data
        create_s3_folder_structure
        
        echo -e "${GREEN}===== MINDSET data deployment completed successfully! =====${NC}"
        echo -e "${BLUE}The following S3 buckets have been set up:${NC}"
        echo -e "Raw bucket: ${GREEN}$RAW_BUCKET${NC}"
        echo -e "Bronze bucket: ${GREEN}$BRONZE_BUCKET${NC}"
        echo -e "Silver bucket: ${GREEN}$SILVER_BUCKET${NC}"
        echo -e "Gold bucket: ${GREEN}$GOLD_BUCKET${NC}"
        echo -e "Models bucket: ${GREEN}$MODELS_BUCKET${NC}"
        
        echo -e "${BLUE}You can now proceed with the pipeline execution:${NC}"
        echo -e "${YELLOW}./deploy-aws-master.sh $ENV${NC}"
    else
        echo -e "${RED}Data deployment failed due to missing dataset files.${NC}"
        exit 1
    fi
}

# Execute main function
main