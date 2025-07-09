# UI Deployment to EKS

## Overview
This directory contains the modern, responsive UI for Customer Churn Prediction that communicates with the FastAPI backend.

## Features
- 🎨 Modern, responsive design with gradient backgrounds
- 📊 Real-time prediction results with visual risk indicators
- 🔄 Automatic API health monitoring
- 💡 Intelligent recommendations based on risk levels
- 📱 Mobile-friendly interface
- ⚡ Fast, lightweight nginx-based deployment

## Prerequisites
- EKS cluster "churnmodel" is running
- kubectl configured to connect to the cluster
- Docker installed and configured
- FastAPI backend deployed and accessible

## Step 1: Update API Configuration

Before building, update the API URL in `script.js`:

```javascript
// Update this line in script.js
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : 'http://YOUR_NODE_IP:30080'; // Replace with actual node IP or LoadBalancer URL
```

**Get your FastAPI service URL:**
```bash
# If using NodePort
kubectl get nodes -o wide
# Use: http://NODE_EXTERNAL_IP:30080

# If using LoadBalancer
kubectl get service churn-api-lb
# Use the EXTERNAL-IP
```

## Step 2: Build and Push Docker Image

```bash
# Navigate to UI deployment directory
cd ui-deployment/

# Build Docker image
docker build -t churn-prediction-ui:latest .

# Tag for registry (replace with your registry)
docker tag churn-prediction-ui:latest your-registry/churn-prediction-ui:latest

# Push to registry
docker push your-registry/churn-prediction-ui:latest
```

**Alternative: Use ECR**
```bash
# Create ECR repository
aws ecr create-repository --repository-name churn-prediction-ui --region ap-south-1

# Get ECR login token
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com

# Tag and push
docker tag churn-prediction-ui:latest YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-ui:latest
docker push YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/churn-prediction-ui:latest
```

## Step 3: Update Kubernetes Manifest

Edit `k8s-deployment.yaml` and update the image name:

```yaml
containers:
- name: nginx
  image: your-registry/churn-prediction-ui:latest  # Update this line
```

## Step 4: Deploy to EKS

```bash
# Set context to churnmodel cluster
kubectl config use-context arn:aws:eks:ap-south-1:ACCOUNT:cluster/churnmodel

# Apply the deployment
kubectl apply -f k8s-deployment.yaml

# Verify deployment
kubectl get deployments
kubectl get pods -l app=churn-prediction-ui
kubectl get services
```

## Step 5: Test the Deployment

```bash
# Check pod logs
kubectl logs -l app=churn-prediction-ui

# Port forward for testing (optional)
kubectl port-forward service/churn-prediction-ui-service 3000:80

# Test locally
curl http://localhost:3000/health
```

## Step 6: Access the UI

**Option A: NodePort (Default - Port 30081)**
```bash
# Get node external IP
kubectl get nodes -o wide

# Access UI
http://NODE_EXTERNAL_IP:30081
```

**Option B: LoadBalancer**
```bash
# Create LoadBalancer service
kubectl expose deployment churn-prediction-ui --type=LoadBalancer --port=80 --target-port=80 --name=churn-ui-lb

# Get external IP
kubectl get service churn-ui-lb

# Access UI
http://EXTERNAL_IP
```

**Option C: Ingress (Domain-based)**
```bash
# Install nginx ingress controller (if not installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.5.1/deploy/static/provider/cloud/deploy.yaml

# Apply ingress
kubectl apply -f k8s-deployment.yaml

# Add to /etc/hosts (for local testing)
echo "NODE_IP churn-ui.local" >> /etc/hosts

# Access UI
http://churn-ui.local
```

## Step 7: Verify Full Stack

1. **Check FastAPI Status**: UI should show "✅ API Online" in header
2. **Test Prediction**: Fill form and click "Predict Churn Risk"
3. **Check Model Info**: Right panel should show model version and status

## UI Endpoints

- **Main Interface**: `/`
- **Health Check**: `/health` (returns "healthy")

## UI Features

### Form Sections
- **Demographics**: Gender, age, family status
- **Services**: Phone, internet, streaming services
- **Account**: Contract, billing, payment method

### Results Display
- **Risk Indicator**: Color-coded circular progress (High/Medium/Low)
- **Probability**: Exact churn percentage
- **Recommendations**: Tailored retention strategies
- **Model Info**: Version, update time, registry details

### Interactive Features
- **Sample Data**: Auto-fill button for quick testing
- **Model Reload**: Refresh model from registry
- **Real-time Status**: API health monitoring
- **Responsive Design**: Works on desktop, tablet, mobile

## Troubleshooting

**UI not loading:**
```bash
# Check pod status
kubectl describe pods -l app=churn-prediction-ui

# Check logs
kubectl logs -l app=churn-prediction-ui

# Check service
kubectl get service churn-prediction-ui-nodeport
```

**API connection issues:**
```bash
# Verify FastAPI is running
kubectl get pods -l app=churn-prediction-api

# Check if NodePort is accessible
curl http://NODE_IP:30080/health

# Update script.js with correct API URL
```

**Model not loading in UI:**
```bash
# Check model status via API
curl http://NODE_IP:30080/model-info

# Reload model via UI or API
curl -X POST http://NODE_IP:30080/reload-model
```

## Customization

**Update Branding:**
- Edit `style.css` for colors and styling
- Modify `index.html` for title and content
- Update `script.js` for functionality

**Change API URL:**
- Update `API_BASE_URL` in `script.js`
- Rebuild and redeploy container

## Security Features

- Content Security Policy headers
- XSS protection
- CORS configuration
- Input validation
- No sensitive data exposure

## Performance

- Gzip compression enabled
- Static asset caching (1 year)
- Lightweight nginx container (~20MB)
- CDN-hosted external libraries
- Optimized for mobile networks 