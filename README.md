# MLOps Churn Prediction Project

A complete MLOps solution for customer churn prediction using SageMaker, FastAPI, and Kubernetes, with comprehensive monitoring and automated deployment pipelines.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SageMaker     │    │   FastAPI       │    │   React UI      │
│   Model         │───▶│   Backend       │◀───│   Frontend      │
│   Registry      │    │   (EKS)         │    │   (EKS)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   Prometheus    │    │   Grafana       │
│   Tracking      │    │   Monitoring    │    │   Dashboards    │
│   (EKS)         │    │   (EKS)         │    │   (EKS)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Project Structure

```
mlops-churn-project/
├── ML Pipeline
│   ├── churn_pipeline.py          # SageMaker training pipeline
│   ├── preprocessing.py           # Data preprocessing
│   ├── train.py                   # XGBoost training with MLflow
│   ├── evaluate.py                # Model evaluation
│   └── inference.py               # SageMaker inference
│
├── FastAPI Backend
│   ├── fastapi-deployment/
│   │   ├── main.py               # Enhanced FastAPI app with metrics
│   │   ├── requirements.txt      # Python dependencies
│   │   ├── Dockerfile           # Container definition
│   │   └── k8s-deployment.yaml  # Kubernetes manifests
│
├── UI Frontend
│   ├── ui-deployment/
│   │   ├── index.html           # Modern responsive UI
│   │   ├── script.js            # API communication
│   │   ├── style.css            # Beautiful styling
│   │   ├── Dockerfile           # Container definition
│   │   └── k8s-deployment.yaml  # Kubernetes manifests
│
├── Deployment
│   ├── .github/workflows/       # GitHub Actions workflows
│   │   ├── deploy-fastapi.yml   # FastAPI deployment
│   │   ├── deploy-ui.yml        # UI deployment
│   │   └── deploy-complete-stack.yml # Complete stack
│   ├── deploy-stack.sh          # Manual deployment script
│   ├── monitoring.yaml          # Prometheus & Grafana setup
│   └── DEPLOYMENT_GUIDE.md      # Detailed deployment guide
│
└── Monitoring
    └── monitoring.yaml           # Complete monitoring stack
```

## Quick Start

### Option 1: GitHub Actions (Recommended)

1. **Fork this repository** to your GitHub account

2. **Add GitHub Secrets**:
   - Go to `Settings > Secrets and variables > Actions`
   - Add these secrets:
     ```
     AWS_ACCESS_KEY_ID: your_access_key
     AWS_SECRET_ACCESS_KEY: your_secret_key
     ```

3. **Deploy the stack**:
   - Go to `Actions` tab
   - Select "Deploy Complete MLOps Stack"
   - Click "Run workflow"
   - Choose environment and options
   - Click "Run workflow"

### Option 2: Manual Shell Script

1. **Prerequisites**:
   ```bash
   # Install required tools
   aws configure  # Configure AWS credentials
   kubectl config use-context your-eks-cluster
   
   # Export required environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   ```

2. **Deploy**:
   ```bash
   chmod +x deploy-stack.sh
   ./deploy-stack.sh
   ```

### Option 3: Individual Components

Deploy components individually using GitHub Actions:

- **FastAPI Backend**: Run "Deploy FastAPI Backend to EKS" workflow
- **UI Frontend**: Run "Deploy UI Frontend to EKS" workflow

## Access URLs

After deployment, access your services:

| Service | URL | Purpose |
|---------|-----|---------|
| **UI Application** | `http://NODE_IP:30081` | Customer churn prediction interface |
| **API Backend** | `http://NODE_IP:30080` | FastAPI backend service |
| **API Documentation** | `http://NODE_IP:30080/docs` | Swagger UI documentation |
| **Health Check** | `http://NODE_IP:30080/health` | API health status |
| **Metrics** | `http://NODE_IP:30080/metrics` | Prometheus metrics |
| **Model Info** | `http://NODE_IP:30080/model-info` | Model version and status |
| **Prometheus** | `http://NODE_IP:30090` | Metrics collection |
| **Grafana** | `http://NODE_IP:30030` | Monitoring dashboards |
| **AlertManager** | `http://NODE_IP:30093` | Alert management |

*Replace `NODE_IP` with your EKS cluster node's external IP*

## Features

### ML Pipeline
- **SageMaker Integration**: Automated model training and registration
- **MLflow Tracking**: Experiment tracking and model versioning
- **Model Registry**: Centralized model management
- **Automated Preprocessing**: Data cleaning and feature engineering

### FastAPI Backend
- **Latest Model Loading**: Automatically loads approved models from SageMaker Registry
- **Fallback Strategy**: S3 backup if registry is unavailable
- **Async Operations**: Non-blocking model loading
- **Thread Safety**: Concurrent request handling
- **Comprehensive Monitoring**: Prometheus metrics integration
- **Error Handling**: Retry logic with exponential backoff
- **Health Checks**: Kubernetes liveness/readiness probes

### UI Frontend
- **Modern Design**: Responsive gradient interface
- **Real-time Predictions**: Interactive churn prediction
- **API Status Monitoring**: Connection health display
- **Sample Data**: Quick testing capabilities
- **Risk Visualization**: Color-coded risk levels
- **Mobile Friendly**: Works on all devices

### Deployment & DevOps
- **GitHub Actions**: Automated CI/CD pipelines
- **Docker Containerization**: ECR registry integration
- **Kubernetes Orchestration**: EKS deployment with auto-scaling
- **Blue-Green Deployment**: Zero-downtime updates
- **Health Monitoring**: Comprehensive health checks
- **Resource Management**: CPU/memory limits and requests

### Monitoring & Observability
- **Prometheus Metrics**: Custom business metrics
- **Grafana Dashboards**: Visual monitoring
- **AlertManager**: Automated alerting
- **Logging**: Structured application logs
- **Tracing**: Request tracing and performance monitoring

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_DEFAULT_REGION` | AWS region | `ap-south-1` |
| `MODEL_REGISTRY_GROUP` | SageMaker model package group | `ChurnModelPackageGroup` |
| `S3_BUCKET` | S3 bucket for model artifacts | `mlops-churn-model-artifacts` |

### Kubernetes Resources

| Component | CPU Request | Memory Request | CPU Limit | Memory Limit |
|-----------|-------------|----------------|-----------|--------------|
| **FastAPI** | 500m | 1Gi | 1000m | 2Gi |
| **UI** | 100m | 128Mi | 200m | 256Mi |
| **Prometheus** | 100m | 256Mi | 200m | 512Mi |
| **Grafana** | 100m | 256Mi | 200m | 512Mi |

## API Endpoints

### Core Endpoints

- **POST /predict** - Make churn prediction
- **GET /health** - Health check
- **GET /model-info** - Model information
- **POST /reload-model** - Reload latest model
- **GET /metrics** - Prometheus metrics
- **GET /docs** - API documentation

### Example Prediction Request

```bash
curl -X POST "http://NODE_IP:30080/predict" \
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

### Example Response

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

## Management Commands

### Deployment Status
```bash
# Check all pods and services
kubectl get pods,services -n default

# Check specific deployments
kubectl get deployment churn-prediction-api -n default
kubectl get deployment churn-prediction-ui -n default

# Check HPA status
kubectl get hpa -n default
```

### Logs and Troubleshooting
```bash
# View API logs
kubectl logs -l app=churn-prediction-api -n default --tail=100

# View UI logs
kubectl logs -l app=churn-prediction-ui -n default --tail=100

# Follow logs in real-time
kubectl logs -f deployment/churn-prediction-api -n default
```

### Scaling
```bash
# Manual scaling
kubectl scale deployment churn-prediction-api --replicas=5 -n default

# Check auto-scaling
kubectl describe hpa churn-prediction-api-hpa -n default
```

### Updates
```bash
# Restart deployments
kubectl rollout restart deployment/churn-prediction-api -n default
kubectl rollout restart deployment/churn-prediction-ui -n default

# Check rollout status
kubectl rollout status deployment/churn-prediction-api -n default
```

## Monitoring Queries

### Prometheus Queries

```prometheus
# API Health
up{job="churn-prediction-api"}

# Request Rate
rate(http_requests_total{job="churn-prediction-api"}[5m])

# Error Rate
rate(http_requests_total{job="churn-prediction-api",status=~"5.."}[5m])

# Response Time (95th percentile)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Model Load Status
model_loaded{job="churn-prediction-api"}

# Prediction Count
rate(churn_predictions_total[5m])
```

### Grafana Dashboards

- **API Performance**: Request rate, latency, error rate
- **Model Health**: Model load status, version, errors
- **Business Metrics**: Prediction distribution, risk levels
- **Infrastructure**: Pod resources, scaling metrics

## Alerting Rules

### Critical Alerts
- **API Down**: Service unavailable for >1 minute
- **Model Not Loaded**: ML model not available
- **High Error Rate**: >10% error rate for >2 minutes
- **Pod Crash Looping**: >5 restarts in 1 hour

### Warning Alerts
- **High Latency**: >2s response time for >5 minutes
- **High Resource Usage**: >90% CPU/memory for >10 minutes
- **Model Load Failures**: Registry connection issues

## Troubleshooting

### Common Issues

#### 1. Model Not Loading
```bash
# Check SageMaker registry
aws sagemaker list-model-packages --model-package-group-name ChurnModelPackageGroup

# Check S3 bucket
aws s3 ls s3://mlops-churn-model-artifacts/output/

# Reload model manually
curl -X POST http://NODE_IP:30080/reload-model
```

#### 2. API Connection Issues
```bash
# Check API pod status
kubectl get pods -l app=churn-prediction-api -n default

# Check service endpoints
kubectl get endpoints churn-prediction-api-service -n default

# Test API health
curl http://NODE_IP:30080/health
```

#### 3. UI Not Loading
```bash
# Check UI pod status
kubectl get pods -l app=churn-prediction-ui -n default

# Check nginx configuration
kubectl logs -l app=churn-prediction-ui -n default

# Test UI directly
curl http://NODE_IP:30081
```

#### 4. ECR Issues
```bash
# Login to ECR
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 911167906047.dkr.ecr.ap-south-1.amazonaws.com

# List repositories
aws ecr describe-repositories --region ap-south-1

# Check image tags
aws ecr list-images --repository-name churn-prediction-api --region ap-south-1
```

### Performance Tuning

#### Resource Optimization
```yaml
# Increase API resources for high load
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

#### Auto-scaling Configuration
```yaml
# Adjust HPA thresholds
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 50  # Lower for more aggressive scaling
```

## CI/CD Pipeline

### GitHub Actions Workflows

1. **deploy-fastapi.yml**: Deploys FastAPI backend
   - Builds Docker image
   - Pushes to ECR
   - Deploys to EKS
   - Runs health checks

2. **deploy-ui.yml**: Deploys UI frontend
   - Configures API endpoint
   - Builds Docker image
   - Pushes to ECR
   - Deploys to EKS

3. **deploy-complete-stack.yml**: Deploys entire stack
   - Orchestrates both deployments
   - Runs comprehensive tests
   - Generates deployment summary

### Deployment Strategies

- **Rolling Update**: Zero-downtime deployments
- **Blue-Green**: Full environment switching
- **Canary**: Gradual traffic shifting

## Additional Resources

- [SageMaker Model Registry Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section above
- Review logs: `kubectl logs -l app=churn-prediction-api -n default`
- Test connectivity: `curl http://NODE_IP:30080/health`
- Check deployment status: `kubectl get pods,services -n default`

---

**You now have a complete MLOps churn prediction system running on EKS with comprehensive monitoring, automated deployment, and production-ready features!**

