#!/bin/bash

# Complete MLOps Stack Deployment Script
# This script deploys the complete churn prediction stack to EKS

set -e

# Configuration
AWS_REGION="ap-south-1"
ECR_REPOSITORY_API="churn-prediction-api"
ECR_REPOSITORY_UI="churn-prediction-ui"
EKS_CLUSTER_NAME="churnmodel"
K8S_NAMESPACE="default"
AWS_ACCOUNT_ID="911167906047"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please configure AWS credentials."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

setup_ecr_repositories() {
    log_info "Setting up ECR repositories..."
    
    # Create API repository
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY_API --region $AWS_REGION &> /dev/null; then
        log_info "ECR repository $ECR_REPOSITORY_API already exists"
    else
        log_info "Creating ECR repository $ECR_REPOSITORY_API..."
        aws ecr create-repository \
            --repository-name $ECR_REPOSITORY_API \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        log_success "ECR repository $ECR_REPOSITORY_API created"
    fi
    
    # Create UI repository
    if aws ecr describe-repositories --repository-names $ECR_REPOSITORY_UI --region $AWS_REGION &> /dev/null; then
        log_info "ECR repository $ECR_REPOSITORY_UI already exists"
    else
        log_info "Creating ECR repository $ECR_REPOSITORY_UI..."
        aws ecr create-repository \
            --repository-name $ECR_REPOSITORY_UI \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        log_success "ECR repository $ECR_REPOSITORY_UI created"
    fi
    
    # Login to ECR
    log_info "Logging into ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
    log_success "ECR login successful"
}

build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Generate unique tag based on timestamp
    IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
    
    # Build FastAPI image
    log_info "Building FastAPI image..."
    cd fastapi-deployment
    docker build -t $ECR_REPOSITORY_API:$IMAGE_TAG .
    docker tag $ECR_REPOSITORY_API:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:$IMAGE_TAG
    docker tag $ECR_REPOSITORY_API:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:latest
    
    # Push FastAPI image
    log_info "Pushing FastAPI image..."
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:$IMAGE_TAG
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:latest
    log_success "FastAPI image pushed"
    
    cd ..
    
    # Get node IP for UI configuration
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
    if [ -z "$NODE_IP" ]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    fi
    
    # Build UI image
    log_info "Building UI image..."
    cd ui-deployment
    
    # Update API endpoint in script.js
    API_ENDPOINT="http://$NODE_IP:30080"
    log_info "Configuring UI for API endpoint: $API_ENDPOINT"
    sed -i "s|const API_BASE_URL = .*|const API_BASE_URL = '$API_ENDPOINT';|g" script.js
    
    docker build -t $ECR_REPOSITORY_UI:$IMAGE_TAG .
    docker tag $ECR_REPOSITORY_UI:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:$IMAGE_TAG
    docker tag $ECR_REPOSITORY_UI:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:latest
    
    # Push UI image
    log_info "Pushing UI image..."
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:$IMAGE_TAG
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:latest
    log_success "UI image pushed"
    
    cd ..
    
    # Export image tag for use in deployment
    export IMAGE_TAG=$IMAGE_TAG
    log_success "All images built and pushed with tag: $IMAGE_TAG"
}

setup_kubernetes() {
    log_info "Setting up Kubernetes cluster connection..."
    
    # Update kubeconfig
    aws eks update-kubeconfig --region $AWS_REGION --name $EKS_CLUSTER_NAME
    
    # Verify cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Failed to connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Kubernetes cluster connection established"
    
    # Create AWS credentials secret
    log_info "Creating AWS credentials secret..."
    kubectl create secret generic aws-credentials \
        --from-literal=aws-access-key-id=$AWS_ACCESS_KEY_ID \
        --from-literal=aws-secret-access-key=$AWS_SECRET_ACCESS_KEY \
        --namespace=$K8S_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -
    log_success "AWS credentials secret created/updated"
}

deploy_fastapi() {
    log_info "Deploying FastAPI backend..."
    
    cd fastapi-deployment
    
    # Update image in deployment manifest
    sed -i "s|image: .*|image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:$IMAGE_TAG|g" k8s-deployment.yaml
    
    # Add deployment timestamp
    sed -i "/metadata:/a\\  annotations:\n    deployment.kubernetes.io/revision: \"$(date +%s)\"" k8s-deployment.yaml
    
    # Apply deployment
    kubectl apply -f k8s-deployment.yaml --namespace=$K8S_NAMESPACE
    
    # Wait for rollout
    log_info "Waiting for FastAPI deployment to complete..."
    kubectl rollout status deployment/churn-prediction-api --namespace=$K8S_NAMESPACE --timeout=300s
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=churn-prediction-api --namespace=$K8S_NAMESPACE --timeout=120s
    
    log_success "FastAPI deployment completed"
    cd ..
}

deploy_ui() {
    log_info "Deploying UI frontend..."
    
    cd ui-deployment
    
    # Update image in deployment manifest
    sed -i "s|image: .*|image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:$IMAGE_TAG|g" k8s-deployment.yaml
    
    # Add deployment timestamp
    sed -i "/metadata:/a\\  annotations:\n    deployment.kubernetes.io/revision: \"$(date +%s)\"" k8s-deployment.yaml
    
    # Apply deployment
    kubectl apply -f k8s-deployment.yaml --namespace=$K8S_NAMESPACE
    
    # Wait for rollout
    log_info "Waiting for UI deployment to complete..."
    kubectl rollout status deployment/churn-prediction-ui --namespace=$K8S_NAMESPACE --timeout=300s
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=churn-prediction-ui --namespace=$K8S_NAMESPACE --timeout=120s
    
    log_success "UI deployment completed"
    cd ..
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    if [ -f "monitoring.yaml" ]; then
        kubectl apply -f monitoring.yaml --namespace=$K8S_NAMESPACE
        log_success "Monitoring stack deployed"
    else
        log_warning "monitoring.yaml not found, skipping monitoring deployment"
    fi
}

verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    echo "Checking pods..."
    kubectl get pods -l app=churn-prediction-api --namespace=$K8S_NAMESPACE
    kubectl get pods -l app=churn-prediction-ui --namespace=$K8S_NAMESPACE
    
    # Check services
    echo "Checking services..."
    kubectl get services --namespace=$K8S_NAMESPACE
    
    # Get node IP
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
    if [ -z "$NODE_IP" ]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Test API health
    log_info "Testing API health..."
    for i in {1..5}; do
        if curl -f -s "http://$NODE_IP:30080/health" > /dev/null; then
            log_success "API health check passed!"
            break
        else
            log_warning "API health check attempt $i/5 failed, retrying..."
            sleep 10
        fi
    done
    
    # Test UI health
    log_info "Testing UI health..."
    for i in {1..5}; do
        if curl -f -s "http://$NODE_IP:30081" > /dev/null; then
            log_success "UI health check passed!"
            break
        else
            log_warning "UI health check attempt $i/5 failed, retrying..."
            sleep 10
        fi
    done
    
    log_success "Deployment verification completed"
}

print_summary() {
    # Get node IP
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}')
    if [ -z "$NODE_IP" ]; then
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    fi
    
    echo ""
    echo "================================="
    echo "DEPLOYMENT COMPLETED SUCCESSFULLY"
    echo "================================="
    echo ""
    echo "Access URLs:"
    echo "   UI Application: http://$NODE_IP:30081"
    echo "   API Backend: http://$NODE_IP:30080"
    echo "   API Documentation: http://$NODE_IP:30080/docs"
    echo "   API Health: http://$NODE_IP:30080/health"
    echo "   Model Info: http://$NODE_IP:30080/model-info"
    echo ""
    echo "Monitoring URLs:"
    echo "   Prometheus: http://$NODE_IP:30090"
    echo "   Grafana: http://$NODE_IP:30030 (admin/admin123)"
    echo "   AlertManager: http://$NODE_IP:30093"
    echo ""
    echo "Docker Images:"
    echo "   FastAPI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_API:$IMAGE_TAG"
    echo "   UI: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_UI:$IMAGE_TAG"
    echo ""
    echo "Management Commands:"
    echo "   kubectl get pods,services -n $K8S_NAMESPACE"
    echo "   kubectl logs -l app=churn-prediction-api -n $K8S_NAMESPACE"
    echo "   kubectl logs -l app=churn-prediction-ui -n $K8S_NAMESPACE"
    echo ""
    echo "To redeploy:"
    echo "   ./deploy-stack.sh"
    echo ""
}

# Main execution
main() {
    echo "Starting MLOps Stack Deployment..."
    echo "======================================"
    
    # Check if required environment variables are set
    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        log_error "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set"
        exit 1
    fi
    
    check_prerequisites
    setup_ecr_repositories
    setup_kubernetes
    build_and_push_images
    deploy_fastapi
    deploy_ui
    deploy_monitoring
    verify_deployment
    print_summary
    
    log_success "Complete MLOps stack deployment finished!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-monitoring)
            SKIP_MONITORING=true
            shift
            ;;
        --skip-ui)
            SKIP_UI=true
            shift
            ;;
        --skip-api)
            SKIP_API=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-monitoring    Skip monitoring stack deployment"
            echo "  --skip-ui           Skip UI deployment"
            echo "  --skip-api          Skip API deployment"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main 