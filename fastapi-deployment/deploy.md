# FastAPI Deployment to EKS

## Overview
This directory contains the FastAPI application that loads the latest registered model from SageMaker Registry and provides prediction endpoints.

## Prerequisites
- EKS cluster "churnmodel" is running
- kubectl configured to connect to the cluster
- Docker installed and configured
- AWS credentials with SageMaker access

## Step 1: Create AWS Credentials Secret

```bash
# Create AWS credentials secret for the pods
kubectl create secret generic aws-credentials \
  --from-literal=aws-access-key-id=YOUR_ACCESS_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET_KEY
```

## Step 2: Build and Push Docker Image

```bash
# Navigate to FastAPI deployment directory
cd fastapi-deployment/

# Build Docker image
docker build -t churn-prediction-api:latest .

# Tag for registry (replace with your registry)
docker tag churn-prediction-api:latest your-registry/churn-prediction-api:latest

# Push to registry
docker push your-registry/churn-prediction-api:latest
```

**Alternative: Use ECR**
```bash
# Get ECR login token
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com

# Create ECR repository
aws ecr create-repository --repository-name churn-prediction-api --region ap-south-1

# Tag and push
docker tag churn-prediction-api:latest YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-api:latest
```

## Step 3: Update Kubernetes Manifest

Edit `k8s-deployment.yaml` and update the image name:

```yaml
containers:
- name: fastapi
  image: your-registry/churn-prediction-api:latest  # Update this line
```

## Step 4: Deploy to EKS

```bash
# Set context to churnmodel cluster
kubectl config use-context arn:aws:eks:ap-south-1:ACCOUNT:cluster/churnmodel

# Apply the deployment
kubectl apply -f k8s-deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods
kubectl get services
```

## Step 5: Test the Deployment

```bash
# Check pod logs
kubectl logs -l app=churn-prediction-api

# Port forward for testing (optional)
kubectl port-forward service/churn-prediction-api-service 8000:80

# Test health endpoint
curl http://localhost:8000/health

# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "gender": "Female",
       "SeniorCitizen": 0,
       "Partner": "Yes",
       "Dependents": "No",
       "tenure": 12,
       "PhoneService": "Yes",
       "MultipleLines": "No",
       "InternetService": "DSL",
       "OnlineSecurity": "Yes",
       "OnlineBackup": "No",
       "DeviceProtection": "No",
       "TechSupport": "No",
       "StreamingTV": "No",
       "StreamingMovies": "No",
       "Contract": "Month-to-month",
       "PaperlessBilling": "Yes",
       "PaymentMethod": "Electronic check",
       "MonthlyCharges": 50.0,
       "TotalCharges": "600.0"
     }'
```

## Step 6: External Access

**Option A: NodePort (Already configured)**
```bash
# Get node external IP
kubectl get nodes -o wide

# Access via NodePort
curl http://NODE_EXTERNAL_IP:30080/health
```

**Option B: Load Balancer**
```bash
# Create LoadBalancer service
kubectl expose deployment churn-prediction-api --type=LoadBalancer --port=80 --target-port=8000 --name=churn-api-lb

# Get external IP
kubectl get service churn-api-lb
```

## API Endpoints

- **Health Check**: `GET /health`
- **Prediction**: `POST /predict`
- **Model Info**: `GET /model-info`
- **Reload Model**: `POST /reload-model`
- **API Docs**: `GET /docs` (Swagger UI)

## Features

- ✅ Loads latest approved model from SageMaker Registry
- ✅ Fallback to S3 if no approved model
- ✅ Real-time model reloading
- ✅ Health checks and monitoring
- ✅ CORS enabled for UI integration
- ✅ Detailed logging and error handling

## Troubleshooting

**Model not loading:**
```bash
# Check if model is approved in SageMaker Registry
aws sagemaker list-model-packages --model-package-group-name ChurnModelPackageGroup

# Manually reload model
curl -X POST http://NODE_IP:30080/reload-model
```

**Pod not starting:**
```bash
# Check pod events
kubectl describe pod POD_NAME

# Check logs
kubectl logs POD_NAME
```

**AWS credentials:**
```bash
# Verify secret exists
kubectl get secret aws-credentials -o yaml
``` 