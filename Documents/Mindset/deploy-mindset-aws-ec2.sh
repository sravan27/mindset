#!/bin/bash
# MINDSET AWS EC2 Deployment Script
# This script creates a single EC2 instance with all MINDSET components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Welcome banner
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}  MINDSET - AWS EC2 Deployment                    ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Starting deployment to AWS EC2...${NC}"

# Check required commands
echo -e "${YELLOW}Checking required commands...${NC}"
REQUIRED_COMMANDS=("aws" "python" "pip")
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

# Step 1: Set up S3 buckets for data
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 1: Creating S3 buckets${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create buckets with unique names
S3_BUCKET_DATA="${RESOURCE_PREFIX}-data"
echo -e "${YELLOW}Creating S3 bucket: ${S3_BUCKET_DATA}${NC}"
aws s3api create-bucket \
    --bucket ${S3_BUCKET_DATA} \
    --region ${AWS_REGION} \
    $(if [ "$AWS_REGION" != "us-east-1" ]; then echo "--create-bucket-configuration LocationConstraint=${AWS_REGION}"; fi)

# Create directories for data layers within the bucket
for layer in raw bronze silver gold silicon datasets; do
    echo -e "${YELLOW}Creating S3 directory: ${layer}${NC}"
    aws s3api put-object --bucket ${S3_BUCKET_DATA} --key ${layer}/
done

# Step 2: Upload MINDLarge datasets to S3
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 2: Uploading datasets to S3${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Check if datasets are already uploaded
DATASETS_CHECK=$(aws s3 ls s3://${S3_BUCKET_DATA}/datasets/ || echo "NOT_FOUND")
if [[ "$DATASETS_CHECK" == *"NOT_FOUND"* ]] || [[ "$DATASETS_CHECK" == "" ]]; then
    # Upload MINDLarge datasets
    for dir in MINDlarge_train MINDlarge_dev MINDlarge_test; do
        if [ -d "$dir" ]; then
            echo -e "${YELLOW}Uploading $dir...${NC}"
            aws s3 cp $dir s3://${S3_BUCKET_DATA}/datasets/$dir/ --recursive
        fi
    done
else
    echo -e "${GREEN}Datasets already uploaded, skipping...${NC}"
fi

# Step 3: Create security group for EC2
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 3: Creating security group${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text)
echo -e "${YELLOW}Using VPC: ${VPC_ID}${NC}"

# Create security group
SECURITY_GROUP_NAME="${RESOURCE_PREFIX}-sg"
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name ${SECURITY_GROUP_NAME} \
    --description "Security group for MINDSET EC2 instance" \
    --vpc-id ${VPC_ID} \
    --output text \
    --query 'GroupId')

echo -e "${YELLOW}Created security group: ${SECURITY_GROUP_ID}${NC}"

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP_ID} \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP_ID} \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP_ID} \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SECURITY_GROUP_ID} \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Step 4: Create EC2 IAM role with S3 access
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 4: Creating IAM role${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create IAM role for EC2
IAM_ROLE_NAME="${RESOURCE_PREFIX}-ec2-role"

# Create trust policy document
cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name ${IAM_ROLE_NAME} \
    --assume-role-policy-document file://trust-policy.json

# Attach S3 access policy
aws iam attach-role-policy \
    --role-name ${IAM_ROLE_NAME} \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile and add role to it
aws iam create-instance-profile \
    --instance-profile-name ${IAM_ROLE_NAME}

aws iam add-role-to-instance-profile \
    --role-name ${IAM_ROLE_NAME} \
    --instance-profile-name ${IAM_ROLE_NAME}

# Wait for the instance profile to be ready
echo -e "${YELLOW}Waiting for instance profile to be ready...${NC}"
sleep 10

# Step 5: Create user data script for EC2 instance
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 5: Creating user data script${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Create user data script
cat > ec2-user-data.sh << 'EOF'
#!/bin/bash
# MINDSET EC2 Setup Script

# Set up logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting MINDSET setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install dependencies
apt-get install -y build-essential python3-pip python3-dev python3-venv \
    nginx nodejs npm git libssl-dev pkg-config awscli

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Clone MINDSET repository
cd /home/ubuntu
git clone https://github.com/sravan27/mindset.git Mindset
cd Mindset

# Set up Python environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install boto3 dask[complete] pyarrow duckdb

# Build Rust metrics engine
cd src/rust/metrics_engine
python build.py
cd ~/Mindset

# Set up environment variables
cat > .env << ENV_EOF
# AWS Settings
S3_BUCKET=%%S3_BUCKET%%
AWS_REGION=%%AWS_REGION%%
# API Keys
NEWSAPI_KEY=%%NEWSAPI_KEY%%
# Features
ENABLE_SILICON_LAYER=true
USE_RUST_ENGINE=true
ENV_EOF

# Replace placeholders with actual values
sed -i "s|%%S3_BUCKET%%|${S3_BUCKET}|g" .env
sed -i "s|%%AWS_REGION%%|${AWS_REGION}|g" .env
sed -i "s|%%NEWSAPI_KEY%%|${NEWSAPI_KEY}|g" .env

# Set up directories
mkdir -p ~/mindset_data/{raw,bronze,silver,silicon,gold}

# Install and configure Nginx
cat > /etc/nginx/sites-available/mindset << 'NGINX_EOF'
server {
    listen 80;
    server_name _;

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
NGINX_EOF

ln -s /etc/nginx/sites-available/mindset /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
systemctl restart nginx

# Set up the frontend
cd ~/Mindset/src/frontend
npm install
npm run build

# Set up systemd services
# 1. API Service
cat > /etc/systemd/system/mindset-api.service << 'API_EOF'
[Unit]
Description=MINDSET API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset
ExecStart=/home/ubuntu/Mindset/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000
Restart=always
Environment="PYTHONPATH=/home/ubuntu/Mindset"
EnvironmentFile=/home/ubuntu/Mindset/.env

[Install]
WantedBy=multi-user.target
API_EOF

# 2. Frontend Service
cat > /etc/systemd/system/mindset-frontend.service << 'FRONTEND_EOF'
[Unit]
Description=MINDSET Frontend Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset/src/frontend
ExecStart=/usr/bin/npm run start
Restart=always
Environment="NEXT_PUBLIC_API_URL=http://localhost/api"

[Install]
WantedBy=multi-user.target
FRONTEND_EOF

# 3. Data Pipeline Service
cat > /etc/systemd/system/mindset-pipeline.service << 'PIPELINE_EOF'
[Unit]
Description=MINDSET Data Pipeline
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Mindset
ExecStart=/home/ubuntu/Mindset/venv/bin/python run_pipeline_aws.py
Environment="PYTHONPATH=/home/ubuntu/Mindset"
EnvironmentFile=/home/ubuntu/Mindset/.env

[Install]
WantedBy=multi-user.target
PIPELINE_EOF

# 4. Data Pipeline Timer
cat > /etc/systemd/system/mindset-pipeline.timer << 'TIMER_EOF'
[Unit]
Description=Run MINDSET Pipeline hourly

[Timer]
OnBootSec=15min
OnUnitActiveSec=1h
Unit=mindset-pipeline.service

[Install]
WantedBy=timers.target
TIMER_EOF

# Enable and start services
systemctl daemon-reload
systemctl enable mindset-api.service
systemctl enable mindset-frontend.service
systemctl enable mindset-pipeline.timer

systemctl start mindset-api.service
systemctl start mindset-frontend.service
systemctl start mindset-pipeline.timer

echo "MINDSET setup complete!"
EOF

# Replace placeholders in user data
sed -i '' "s|%%S3_BUCKET%%|${S3_BUCKET_DATA}|g" ec2-user-data.sh
sed -i '' "s|%%AWS_REGION%%|${AWS_REGION}|g" ec2-user-data.sh
sed -i '' "s|%%NEWSAPI_KEY%%|${NEWSAPI_KEY}|g" ec2-user-data.sh

# Encode user data
USER_DATA=$(cat ec2-user-data.sh | base64)

# Step 6: Launch EC2 instance
echo -e "${YELLOW}=====================================${NC}"
echo -e "${YELLOW}Step 6: Launching EC2 instance${NC}"
echo -e "${YELLOW}=====================================${NC}"

# Get latest Ubuntu AMI
AMI_ID=$(aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
                "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

echo -e "${YELLOW}Using AMI: ${AMI_ID}${NC}"

# Create key pair
KEY_NAME="${RESOURCE_PREFIX}-key"
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
chmod 400 ${KEY_NAME}.pem
echo -e "${YELLOW}Created key pair: ${KEY_NAME}${NC}"

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ${AMI_ID} \
    --instance-type t3.xlarge \
    --key-name ${KEY_NAME} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --user-data file://ec2-user-data.sh \
    --iam-instance-profile Name=${IAM_ROLE_NAME} \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${RESOURCE_PREFIX}}]" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":40,\"VolumeType\":\"gp3\"}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo -e "${YELLOW}Created EC2 instance: ${INSTANCE_ID}${NC}"

# Wait for instance to be running
echo -e "${YELLOW}Waiting for instance to start...${NC}"
aws ec2 wait instance-running --instance-ids ${INSTANCE_ID}

# Get instance public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids ${INSTANCE_ID} \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo -e "${GREEN}Instance is running with IP: ${PUBLIC_IP}${NC}"

# Step 7: Create DNS record (optional)
# This step is commented out as it requires a hosted zone
# If you have a domain, you can uncomment and configure this part
#echo -e "${YELLOW}=====================================${NC}"
#echo -e "${YELLOW}Step 7: Creating DNS record${NC}"
#echo -e "${YELLOW}=====================================${NC}"
#
#HOSTED_ZONE_ID="your-hosted-zone-id"
#DOMAIN_NAME="mindset.yourdomain.com"
#
#aws route53 change-resource-record-sets \
#    --hosted-zone-id ${HOSTED_ZONE_ID} \
#    --change-batch '{
#        "Changes": [
#            {
#                "Action": "UPSERT",
#                "ResourceRecordSet": {
#                    "Name": "'${DOMAIN_NAME}'",
#                    "Type": "A",
#                    "TTL": 300,
#                    "ResourceRecords": [
#                        {
#                            "Value": "'${PUBLIC_IP}'"
#                        }
#                    ]
#                }
#            }
#        ]
#    }'
#echo -e "${YELLOW}Created DNS record: ${DOMAIN_NAME} -> ${PUBLIC_IP}${NC}"

# Create a file with configuration information
cat > .env.aws-ec2 << EOF
# MINDSET AWS EC2 Configuration

# Resource Prefix
RESOURCE_PREFIX=${RESOURCE_PREFIX}

# S3 Bucket
S3_BUCKET_DATA=${S3_BUCKET_DATA}

# EC2 Instance
INSTANCE_ID=${INSTANCE_ID}
PUBLIC_IP=${PUBLIC_IP}
KEY_FILE=${KEY_NAME}.pem

# Security Group
SECURITY_GROUP_ID=${SECURITY_GROUP_ID}
EOF

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}     DEPLOYMENT COMPLETE!            ${NC}"
echo -e "${GREEN}=====================================${NC}"

echo -e "${YELLOW}MINDSET has been deployed to AWS EC2!${NC}"
echo -e ""
echo -e "${YELLOW}S3 Data Bucket:${NC}"
echo -e "${GREEN}${S3_BUCKET_DATA}${NC}"
echo -e ""
echo -e "${YELLOW}EC2 Instance:${NC}"
echo -e "${GREEN}${INSTANCE_ID} (${PUBLIC_IP})${NC}"
echo -e ""
echo -e "${YELLOW}Access MINDSET:${NC}"
echo -e "${GREEN}http://${PUBLIC_IP}/${NC}"
echo -e ""
echo -e "${YELLOW}SSH Access:${NC}"
echo -e "${GREEN}ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_IP}${NC}"
echo -e ""
echo -e "${YELLOW}Configuration saved to:${NC}"
echo -e "${GREEN}.env.aws-ec2${NC}"
echo -e ""
echo -e "${YELLOW}Check instance status:${NC}"
echo -e "${GREEN}tail -f /var/log/user-data.log${NC}"
echo -e ""
echo -e "${YELLOW}Note: It may take 5-10 minutes for the instance to complete setup${NC}"
echo -e "${YELLOW}You can monitor the setup process by SSH into the instance and running:${NC}"
echo -e "${GREEN}sudo tail -f /var/log/user-data.log${NC}"
echo -e ""
echo -e "${BLUE}Thank you for using MINDSET on AWS!${NC}"