# Configuration Verification Checklist

This document verifies that all configurations are correct and the deployment will work properly.

## 1. Docker Images Configuration

### FastAPI Docker Configuration
- **Base Image**: python:3.9-slim
- **Dependencies**: curl, gcc installed for health checks
- **Port**: 8000 exposed
- **Health Check**: curl -f http://localhost:8000/health
- **Command**: uvicorn main:app --host 0.0.0.0 --port 8000

### UI Docker Configuration  
- **Base Image**: nginx:alpine
- **Port**: 80 exposed
- **Configuration**: nginx.conf properly configured
- **Static Files**: HTML, CSS, JS files copied

## 2. ECR Configuration

### Repository Names
- **API Repository**: churn-prediction-api
- **UI Repository**: churn-prediction-ui
- **Registry**: 911167906047.dkr.ecr.ap-south-1.amazonaws.com
- **Region**: ap-south-1

### Image Tags
- **Latest Tag**: Always pushed for latest version
- **SHA Tag**: GitHub SHA for specific builds
- **Format**: {REGISTRY}/{REPOSITORY}:{TAG}

## 3. Kubernetes Configuration

### FastAPI Deployment
- **Name**: churn-prediction-api
- **Namespace**: default
- **Replicas**: 2
- **Image**: 911167906047.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-api:latest
- **Port**: 8000
- **Resources**:
  - Requests: 500m CPU, 1Gi Memory
  - Limits: 1000m CPU, 2Gi Memory

### UI Deployment
- **Name**: churn-prediction-ui
- **Namespace**: default
- **Replicas**: 2
- **Image**: 911167906047.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-ui:latest
- **Port**: 80
- **Resources**:
  - Requests: 100m CPU, 128Mi Memory
  - Limits: 200m CPU, 256Mi Memory

### Services
- **API Service**: 
  - ClusterIP: churn-prediction-api-service (port 80 -> 8000)
  - NodePort: churn-prediction-api-nodeport (port 30080)
- **UI Service**:
  - ClusterIP: churn-prediction-ui-service (port 80 -> 80)
  - NodePort: churn-prediction-ui-nodeport (port 30081)

## 4. Environment Variables

### FastAPI Environment
- **AWS_DEFAULT_REGION**: ap-south-1
- **MODEL_REGISTRY_GROUP**: ChurnModelPackageGroup
- **S3_BUCKET**: mlops-churn-model-artifacts
- **PYTHONPATH**: /app

### Required Secrets
- **aws-credentials**: Contains AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

## 5. GitHub Actions Configuration

### Secrets Required
- **AWS_ACCESS_KEY_ID**: AWS access key
- **AWS_SECRET_ACCESS_KEY**: AWS secret key

### Workflow Configuration
- **AWS Region**: ap-south-1
- **EKS Cluster**: churnmodel
- **Namespace**: default

## 6. API Endpoints Verification

### FastAPI Endpoints
- **GET /health**: Health check endpoint
- **POST /predict**: Prediction endpoint
- **GET /model-info**: Model information
- **POST /reload-model**: Model reload
- **GET /metrics**: Prometheus metrics
- **GET /docs**: API documentation

### Expected Response Format
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7834,
  "confidence_score": 0.5668,
  "risk_level": "Very High",
  "model_version": "1.0.0",
  "model_arn": "arn:aws:sagemaker:...",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "processing_time_ms": 45.67
}
```

## 7. UI Configuration

### API Endpoint Configuration
- **Default**: http://NODE_IP:30080
- **Localhost**: http://localhost:8000
- **Auto-detection**: Configured in workflows

### Features
- Sample data fill functionality
- Real-time API status monitoring
- Model reload capability
- Responsive design

## 8. Health Checks

### Kubernetes Probes
- **Liveness Probe**: /health endpoint
- **Readiness Probe**: /health endpoint  
- **Startup Probe**: /health endpoint

### GitHub Actions Health Checks
- API health verification with retries
- UI accessibility checks
- Model endpoint testing

## 9. Monitoring Configuration

### Prometheus Metrics
- **Custom Metrics**: Prediction counts, latency, error rates
- **Model Metrics**: Load status, version tracking
- **Health Metrics**: API status monitoring

### Grafana Dashboards
- API performance metrics
- Model health status
- Business metrics

## 10. Common Issues Prevention

### Docker Build Issues
- Proper base image selection
- Required system dependencies installed
- Health check command available

### Kubernetes Deployment Issues  
- Correct image references
- Proper resource limits
- Valid environment variables
- Secret references correct

### Network Issues
- Proper service configurations
- NodePort accessibility
- Health check endpoints working

## 11. Deployment Verification Steps

1. **ECR Login**: AWS ECR authentication successful
2. **Image Build**: Docker images build without errors
3. **Image Push**: Images pushed to ECR successfully
4. **Cluster Access**: kubectl can connect to EKS cluster
5. **Secret Creation**: AWS credentials secret created
6. **Deployment Apply**: Kubernetes manifests applied successfully
7. **Rollout Status**: Deployments complete successfully
8. **Pod Readiness**: All pods reach ready state
9. **Health Checks**: All health endpoints responding
10. **Service Access**: Services accessible via NodePort

## 12. Troubleshooting Commands

```bash
# Check deployment status
kubectl get pods,services -n default

# Check logs
kubectl logs -l app=churn-prediction-api -n default
kubectl logs -l app=churn-prediction-ui -n default

# Test health endpoints
curl http://NODE_IP:30080/health
curl http://NODE_IP:30081

# Check ECR images
aws ecr list-images --repository-name churn-prediction-api --region ap-south-1

# Restart deployments
kubectl rollout restart deployment/churn-prediction-api -n default
kubectl rollout restart deployment/churn-prediction-ui -n default
```

## Configuration Status: VERIFIED

All configurations have been reviewed and are properly set up for successful deployment. 