# 🚀 Complete MLOps Churn Prediction Deployment Guide

## 📁 Project Structure

```
mlops-churn-project/
├── 🤖 SageMaker Training Pipeline
│   ├── churn_pipeline.py          # Main SageMaker pipeline
│   ├── preprocessing.py           # Data preprocessing
│   ├── train.py                   # Model training with MLflow
│   ├── evaluate.py                # Model evaluation
│   ├── inference.py               # SageMaker inference handlers
│   └── requirements.txt           # Python dependencies
│
├── 🔧 FastAPI Backend Deployment
│   ├── fastapi-deployment/
│   │   ├── main.py               # FastAPI app with model loading
│   │   ├── Dockerfile            # Container definition
│   │   ├── requirements.txt      # API dependencies
│   │   ├── k8s-deployment.yaml   # Kubernetes manifests
│   │   └── deploy.md             # Deployment instructions
│
└── 🎨 UI Frontend Deployment
    ├── ui-deployment/
    │   ├── index.html            # Modern responsive UI
    │   ├── style.css             # Beautiful styling
    │   ├── script.js             # API communication
    │   ├── nginx.conf            # Web server config
    │   ├── Dockerfile            # Container definition
    │   ├── k8s-deployment.yaml   # Kubernetes manifests
    │   └── deploy.md             # Deployment instructions
```

## 🔄 Complete Workflow

### Phase 1: ML Pipeline (SageMaker)
1. **Data Processing**: Automated preprocessing of customer data
2. **Model Training**: XGBoost with MLflow experiment tracking
3. **Model Registry**: Registered models for production deployment
4. **MLflow Experiments**: Track at http://3.110.135.31:30418/

### Phase 2: API Backend (EKS)
1. **Model Loading**: Automatically loads latest approved model from SageMaker Registry
2. **Inference API**: FastAPI with health checks and monitoring
3. **Auto-scaling**: Kubernetes deployment with resource management
4. **Access**: NodePort (30080) or LoadBalancer

### Phase 3: UI Frontend (EKS)
1. **Modern Interface**: Responsive design with real-time predictions
2. **API Integration**: Communicates with FastAPI backend
3. **User Experience**: Visual risk indicators and recommendations
4. **Access**: NodePort (30081) or LoadBalancer

## 🚀 Deployment Options

### Option 1: GitHub Actions (Recommended)
**Automated deployment using GitHub Actions workflows**

#### Prerequisites
- ✅ GitHub repository with this code
- ✅ EKS cluster "churnmodel" running
- ✅ AWS credentials added as GitHub secrets

#### Setup GitHub Secrets
```bash
# Add these secrets to your GitHub repository:
# Settings > Secrets and variables > Actions > New repository secret

AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
```

#### Available Workflows
1. **🔧 Deploy FastAPI to EKS** - Deploy backend API only
2. **🎨 Deploy UI to EKS** - Deploy frontend UI only  
3. **🚀 Deploy Complete Stack** - Deploy both FastAPI + UI together

#### How to Use
1. Go to **GitHub Actions** tab in your repository
2. Select the workflow you want to run
3. Click **"Run workflow"**
4. Choose environment and options
5. Click **"Run workflow"** to start deployment

### Option 2: Manual Deployment

#### Prerequisites
```bash
# Ensure you have:
# ✅ EKS cluster "churnmodel" running
# ✅ SageMaker model in registry (approved)
# ✅ kubectl configured
# ✅ Docker installed
# ✅ AWS credentials configured
```

### Step 1: Deploy FastAPI Backend
```bash
cd fastapi-deployment/

# 1. Create AWS credentials secret
kubectl create secret generic aws-credentials \
  --from-literal=aws-access-key-id=YOUR_ACCESS_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET_KEY

# 2. Build and push image
docker build -t churn-prediction-api:latest .
# Push to your registry (ECR/DockerHub)

# 3. Update image in k8s-deployment.yaml
# 4. Deploy
kubectl apply -f k8s-deployment.yaml

# 5. Get API URL
kubectl get nodes -o wide
# API will be available at: http://NODE_IP:30080
```

### Step 2: Deploy UI Frontend
```bash
cd ui-deployment/

# 1. Update API URL in script.js
# Replace: http://YOUR_NODE_IP:30080

# 2. Build and push image
docker build -t churn-prediction-ui:latest .
# Push to your registry

# 3. Update image in k8s-deployment.yaml
# 4. Deploy
kubectl apply -f k8s-deployment.yaml

# 5. Access UI
kubectl get nodes -o wide
# UI will be available at: http://NODE_IP:30081
```

### Step 3: Verify Deployment
```bash
# Check all services
kubectl get pods,services

# Test API
curl http://NODE_IP:30080/health

# Test UI
curl http://NODE_IP:30081/health

# Check logs
kubectl logs -l app=churn-prediction-api
kubectl logs -l app=churn-prediction-ui
```

## 🔗 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **SageMaker Pipeline** | SageMaker Console | Model training & registry |
| **MLflow Tracking** | http://3.110.135.31:30418/ | Experiment tracking |
| **FastAPI Backend** | http://NODE_IP:30080 | Prediction API |
| **UI Frontend** | http://NODE_IP:30081 | User interface |
| **API Docs** | http://NODE_IP:30080/docs | Swagger documentation |

## 📊 API Endpoints

### FastAPI Backend
- `GET /health` - Health check
- `POST /predict` - Make churn prediction
- `GET /model-info` - Model version and status
- `POST /reload-model` - Reload latest model
- `GET /docs` - Interactive API documentation

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

## 🎯 Features

### ML Pipeline
- ✅ Automated data preprocessing
- ✅ XGBoost model training
- ✅ MLflow experiment tracking
- ✅ SageMaker model registry
- ✅ GitHub Actions integration

### API Backend
- ✅ Loads latest approved models
- ✅ S3 fallback for model loading
- ✅ Health checks and monitoring
- ✅ CORS enabled
- ✅ Kubernetes auto-scaling

### UI Frontend
- ✅ Modern responsive design
- ✅ Real-time API status monitoring
- ✅ Visual risk indicators
- ✅ Intelligent recommendations
- ✅ Sample data for testing
- ✅ Mobile-friendly interface

## 🔧 Troubleshooting

### Common Issues

**1. Model not loading in API:**
```bash
# Check if model is approved
aws sagemaker list-model-packages --model-package-group-name ChurnModelPackageGroup

# Manually reload
curl -X POST http://NODE_IP:30080/reload-model
```

**2. UI can't connect to API:**
```bash
# Verify API is running
kubectl get pods -l app=churn-prediction-api

# Check API health
curl http://NODE_IP:30080/health

# Update script.js with correct API URL
```

**3. Pods not starting:**
```bash
# Check pod events
kubectl describe pods -l app=churn-prediction-api

# Check logs
kubectl logs -l app=churn-prediction-api

# Verify secrets exist
kubectl get secret aws-credentials
```

## 🔄 Model Updates

### Automated Updates
1. New model trained via SageMaker pipeline
2. Model automatically registered in registry
3. Approve model in SageMaker console
4. API automatically loads latest approved model

### Manual Updates
```bash
# Reload model via API
curl -X POST http://NODE_IP:30080/reload-model

# Or use UI "Reload Model" button
```

## 📈 Monitoring & Observability

### Health Checks
- API: `/health` endpoint
- UI: `/health` endpoint  
- Kubernetes: Liveness/readiness probes

### Logs
```bash
# API logs
kubectl logs -f -l app=churn-prediction-api

# UI logs
kubectl logs -f -l app=churn-prediction-ui

# MLflow experiments
# Visit: http://3.110.135.31:30418/
```

### Metrics
- Model performance in MLflow
- API response times
- Kubernetes resource usage
- Prediction accuracy tracking

## 🚀 Production Considerations

### Security
- AWS credentials via Kubernetes secrets
- CORS configuration
- Content Security Policy headers
- Input validation

### Scalability
- Horizontal pod autoscaling
- Resource limits and requests
- Load balancing
- CDN for static assets

### High Availability
- Multiple replicas
- Health checks
- Graceful shutdowns
- Rolling updates

## 📞 Support

For issues or questions:
1. Check logs: `kubectl logs -l app=churn-prediction-api`
2. Verify connectivity: `curl http://NODE_IP:30080/health`
3. Review deployment guides in respective folders
4. Check MLflow experiments for model issues

---

**🎉 Congratulations! You now have a complete MLOps churn prediction system running on EKS with:**
- ✅ Automated ML pipeline
- ✅ Production-ready API
- ✅ Beautiful user interface
- ✅ Model monitoring & tracking
- ✅ Kubernetes orchestration 