# **MLOps Churn Prediction Project - Beginner's Guide**  
### **Part 1: Setting Up AWS Infrastructure**  
*(Step-by-Step for Beginners)*  

---

## **📌 What Are We Building?**  
We’re creating an **automated system** that:  
1. **Stores** customer data in AWS S3 (like a secure cloud folder).  
2. **Trains** a machine learning model when new data arrives.  
3. **Tracks** experiments using MLflow (like a diary for AI models).  
4. **Hosts** the model on Kubernetes (a system to run apps reliably).  

> **For Beginners:** Think of this as setting up a factory where data comes in, models get trained automatically, and results are stored neatly.  

---

## **🚀 Step 1: Store Data in AWS S3 (Cloud Storage)**  

### **1.1 Download the Dataset**  
Run this in your terminal (Mac/Linux) or Command Prompt (Windows):  
```bash
wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```
*(This downloads a sample customer data file.)*  

### **1.2 Upload to AWS S3**  
```bash
aws s3 mb s3://mlops-churn-raw-data  # Creates a storage "bucket"
aws s3 cp Telco-Customer-Churn.csv s3://mlops-churn-raw-data/raw/  # Uploads file
```
✅ **Check if it worked:**  
```bash
aws s3 ls s3://mlops-churn-raw-data/raw/  # Should show your file
```

---

## **🔐 Step 2: Set Up Permissions (IAM Roles)**  

### **2.1 SageMaker Role (Allows AI Training)**  
1. Go to **AWS IAM Console** → **Roles** → **Create Role**.  
2. Select **SageMaker** as the service.  
3. Attach these **policies**:  
   - `AmazonSageMakerFullAccess`  
   - `AmazonS3FullAccess`  
   - `CloudWatchLogsFullAccess`  
4. Name it **`SageMakerChurnRole`** and create.  

### **2.2 Lambda Role (Automates Training When New Data Comes)**  
1. Same process, but select **Lambda** as the service.  
2. Attach:  
   - `AWSLambdaBasicExecutionRole`  
   - `AmazonS3ReadOnlyAccess`  
   - `AmazonSageMakerFullAccess`  
3. Name it **`LambdaChurnTriggerRole`**.  

---

## **🤖 Step 3: Create a Lambda Function (Automation)**  

### **3.1 Go to AWS Lambda Console**  
1. Click **"Create Function"**.  
2. Name: `TriggerChurnPipeline`.  
3. Runtime: **Python 3.9**.  
4. Role: Select **`LambdaChurnTriggerRole`**.  

### **3.2 Paste This Code**  
```python
import boto3

def lambda_handler(event, context):
    sm_client = boto3.client('sagemaker')
    sm_client.start_pipeline_execution(PipelineName='churn-pipeline')
    return {"status": "Pipeline triggered!"}
```
📌 **Deploy** → **Save**.  

### **3.3 Set Up S3 Trigger**  
1. Go to **S3 Bucket (`mlops-churn-raw-data`)** → **Properties** → **Event Notifications**.  
2. Add:  
   - **Name:** `NewDataTrigger`  
   - **Event:** `PUT` (when files are uploaded)  
   - **Prefix:** `raw/` (only watch this folder)  
   - **Destination:** `TriggerChurnPipeline` (your Lambda)  

✅ Now, uploading a file to `raw/` will **automatically start training!**  

---

## **⚙️ Step 4: Set Up Kubernetes (EKS) for MLflow**  

### **4.1 Install Required Tools**  
*(Run in terminal)*  
```bash
# Install eksctl (Kubernetes tool)
brew install eksctl  # Mac
# OR (Linux)
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install Helm (for MLflow)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### **4.2 Create the Kubernetes Cluster**  
```bash
eksctl create cluster --name churn-mlops --region us-east-1 --nodegroup-name workers --node-type t3.medium --nodes 2 --managed
```
⏳ **Wait ~15 mins** (AWS is setting up servers for you).  

### **4.3 Deploy MLflow (Model Tracking Dashboard)**  
```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install mlflow bitnami/mlflow --namespace mlflow --create-namespace --set service.type=LoadBalancer
```
🔗 **Get MLflow URL:**  
```bash
kubectl get svc -n mlflow
```
👉 Look for `EXTERNAL-IP` (e.g., `http://123.45.67.89:5000`).  

---

## **✅ Final Checks**  

| Task | Command | Expected Output |
|------|---------|-----------------|
| **S3 File Check** | `aws s3 ls s3://mlops-churn-raw-data/raw/` | Shows `Telco-Customer-Churn.csv` |
| **Kubernetes Nodes** | `kubectl get nodes` | 2 nodes in `Ready` state |
| **MLflow Access** | Open `http://<EXTERNAL-IP>:5000` | MLflow dashboard loads |

---

### **💡 Troubleshooting**  
❌ **"Access Denied" errors?** → Check IAM roles.  
❌ **Cluster not creating?** → Ensure AWS limits allow EKS.  
❌ **MLflow not loading?** → Wait 5 mins and check again.  

