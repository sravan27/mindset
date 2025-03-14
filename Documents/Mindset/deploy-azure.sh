#!/bin/bash
# Master script to deploy MINDSET on Azure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}       MINDSET AZURE DEPLOYMENT SCRIPT           ${NC}"
echo -e "${YELLOW}=================================================${NC}"

# Check if NewsAPI key is provided
NEWSAPI_KEY=${NEWSAPI_KEY:-}
if [ -z "$NEWSAPI_KEY" ]; then
    echo -e "${YELLOW}Please provide your NewsAPI.org API key:${NC}"
    read -p "> " NEWSAPI_KEY
    export NEWSAPI_KEY="$NEWSAPI_KEY"
fi

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Azure CLI not found. Please install it first.${NC}"
    echo -e "${YELLOW}Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install it first.${NC}"
    echo -e "${YELLOW}Visit: https://docs.docker.com/get-docker/${NC}"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl not found. Please install it first.${NC}"
    echo -e "${YELLOW}Visit: https://kubernetes.io/docs/tasks/tools/install-kubectl/${NC}"
    exit 1
fi

# Step 1: Deploy Azure infrastructure
echo -e "${YELLOW}Step 1: Deploying Azure infrastructure...${NC}"
cd infrastructure/azure
chmod +x deploy-infrastructure.sh
./deploy-infrastructure.sh
cd ../..

# Step 2: Upload datasets to Azure Storage
echo -e "${YELLOW}Step 2: Uploading datasets to Azure Storage...${NC}"
cd infrastructure/azure
chmod +x upload-datasets.sh
./upload-datasets.sh
cd ../..

# Step 3: Build and push Docker images to ACR
echo -e "${YELLOW}Step 3: Building and pushing Docker images to ACR...${NC}"
cd infrastructure/azure
chmod +x build-push.sh
./build-push.sh
cd ../..

# Step 4: Deploy Kubernetes manifests
echo -e "${YELLOW}Step 4: Deploying Kubernetes manifests...${NC}"
cd infrastructure/azure
chmod +x deploy-k8s.sh
./deploy-k8s.sh
cd ../..

# Step 5: Wait for all pods to be ready
echo -e "${YELLOW}Step 5: Waiting for all pods to be ready...${NC}"
kubectl wait --namespace mindset \
  --for=condition=ready pod \
  --selector=app=mindset \
  --timeout=300s

# Step 6: Get the application URL
echo -e "${YELLOW}Step 6: Getting application URL...${NC}"
INGRESS_IP=$(kubectl get service -n ingress-nginx ingress-nginx-controller -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}       MINDSET DEPLOYMENT COMPLETED!              ${NC}"
echo -e "${GREEN}=================================================${NC}"
echo -e "${YELLOW}Application URL: http://mindset.example.com${NC}"
echo -e "${YELLOW}(Add this to your hosts file: ${INGRESS_IP} mindset.example.com)${NC}"
echo -e "${YELLOW}Kubernetes Dashboard: Run 'kubectl proxy' and visit:${NC}"
echo -e "${YELLOW}http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/${NC}"