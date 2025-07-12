from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os
import tarfile
import logging
from typing import Dict, List, Optional
import time
import uvicorn
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="ML API for Customer Churn Prediction using SageMaker Registered Models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Custom Prometheus metrics
prediction_counter = Counter('churn_predictions_total', 'Total number of churn predictions', ['prediction_result'])
prediction_latency = Histogram('churn_prediction_duration_seconds', 'Time spent on churn predictions')
model_load_counter = Counter('model_loads_total', 'Total number of model loads', ['status'])
model_loaded_gauge = Gauge('model_loaded', 'Whether model is currently loaded')
model_error_counter = Counter('model_errors_total', 'Total number of model errors', ['error_type'])
api_health_gauge = Gauge('api_health_status', 'API health status (1=healthy, 0=unhealthy)')

# Global model state
class ModelState:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_arn = None
        self.last_updated = None
        self.model_hash = None
        self.model_status = "not_loaded"
        self.error_count = 0
        self.last_error = None
        self.loading_lock = threading.Lock()
        
    def update_model(self, model, version, arn, model_hash):
        with self.loading_lock:
            self.model = model
            self.model_version = version
            self.model_arn = arn
            self.model_hash = model_hash
            self.last_updated = datetime.utcnow().isoformat()
            self.model_status = "loaded"
            self.error_count = 0
            self.last_error = None
            # Update metrics
            model_loaded_gauge.set(1)
            model_load_counter.labels(status='success').inc()
            
    def set_error(self, error_msg):
        with self.loading_lock:
            self.error_count += 1
            self.last_error = error_msg
            self.model_status = "error"
            # Update metrics
            model_loaded_gauge.set(0)
            model_error_counter.labels(error_type='model_load').inc()
            
    def is_loaded(self):
        with self.loading_lock:
            return self.model is not None and self.model_status == "loaded"

model_state = ModelState()

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    confidence_score: float
    risk_level: str
    model_version: str
    model_arn: str
    prediction_timestamp: str
    processing_time_ms: float

class ModelInfo(BaseModel):
    model_loaded: bool
    model_version: Optional[str]
    model_arn: Optional[str]
    model_hash: Optional[str]
    last_updated: Optional[str]
    model_status: str
    error_count: int
    last_error: Optional[str]
    registry_group: str
    s3_bucket: str

def get_aws_clients():
    """Get AWS clients with proper error handling"""
    try:
        session = boto3.Session()
        sm_client = session.client('sagemaker')
        s3_client = session.client('s3')
        
        # Test credentials
        sm_client.list_model_packages(MaxResults=1)
        
        return sm_client, s3_client
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        raise Exception("AWS credentials not configured")
    except Exception as e:
        logger.error(f"Error creating AWS clients: {str(e)}")
        raise Exception(f"AWS client error: {str(e)}")

def get_latest_approved_model(max_retries=3):
    """Get the latest approved model from SageMaker Registry with retry logic"""
    for attempt in range(max_retries):
        try:
            sm_client, _ = get_aws_clients()
            
            # Get model package group info
            model_group_name = os.getenv('MODEL_REGISTRY_GROUP', 'ChurnModelPackageGroup')
            
            # List approved model packages
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_group_name,
                ModelApprovalStatus='Approved',
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=1
            )
            
            if not response['ModelPackageSummaryList']:
                logger.warning("No approved models found in registry")
                return None
            
            latest_model = response['ModelPackageSummaryList'][0]
            
            # Get detailed model information
            model_details = sm_client.describe_model_package(
                ModelPackageName=latest_model['ModelPackageArn']
            )
            
            logger.info(f"Found latest approved model: {latest_model['ModelPackageArn']}")
            logger.info(f"Model version: {latest_model.get('ModelPackageVersion', 'unknown')}")
            
            return model_details
            
        except ClientError as e:
            logger.error(f"AWS client error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Unexpected error getting model (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    return None

def download_and_extract_model(s3_uri, local_path="/tmp/model.tar.gz", max_retries=3):
    """Download and extract model from S3 with retry logic"""
    for attempt in range(max_retries):
        try:
            _, s3_client = get_aws_clients()
            
            # Parse S3 URI
            if s3_uri.startswith('s3://'):
                s3_uri = s3_uri[5:]
            
            bucket, key = s3_uri.split('/', 1)
            
            logger.info(f"Downloading model from s3://{bucket}/{key}")
            
            # Download model
            s3_client.download_file(bucket, key, local_path)
            
            # Extract model
            extract_path = "/tmp/model_extract"
            os.makedirs(extract_path, exist_ok=True)
            
            with tarfile.open(local_path, 'r:gz') as tar:
                tar.extractall(extract_path)
            
            # Find model file
            model_files = [f for f in os.listdir(extract_path) if f.startswith('xgboost-model')]
            if not model_files:
                # Try alternative names
                model_files = [f for f in os.listdir(extract_path) if f.endswith('.model') or f.endswith('.xgb')]
                
            if not model_files:
                raise Exception("No XGBoost model file found in archive")
            
            model_file_path = os.path.join(extract_path, model_files[0])
            
            # Calculate model hash for versioning
            with open(model_file_path, 'rb') as f:
                model_hash = hashlib.md5(f.read()).hexdigest()
            
            logger.info(f"Model extracted to: {model_file_path}")
            logger.info(f"Model hash: {model_hash}")
            
            return model_file_path, model_hash
            
        except Exception as e:
            logger.error(f"Error downloading/extracting model (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    return None, None

def download_model_from_s3_fallback(max_retries=3):
    """Fallback: Download model from S3 training output"""
    for attempt in range(max_retries):
        try:
            _, s3_client = get_aws_clients()
            
            bucket = os.getenv('S3_BUCKET', 'mlops-churn-model-artifacts')
            
            # List objects to find latest model
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix='output/',
                Delimiter='/'
            )
            
            if 'Contents' not in response:
                raise Exception("No model artifacts found in S3")
            
            # Find the latest model.tar.gz
            latest_object = max(response['Contents'], key=lambda x: x['LastModified'])
            
            # Download and extract
            model_file_path, model_hash = download_and_extract_model(
                f"s3://{bucket}/{latest_object['Key']}"
            )
            
            # Load XGBoost model
            booster = xgb.Booster()
            booster.load_model(model_file_path)
            
            logger.info("Model loaded from S3 fallback")
            return booster, "s3-fallback", f"arn:aws:s3:::{bucket}/{latest_object['Key']}", model_hash
            
        except Exception as e:
            logger.error(f"S3 fallback failed (attempt {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    
    return None, None, None, None

async def load_model_async():
    """Load model asynchronously to avoid blocking"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, load_model_sync)
            return await future
    except Exception as e:
        logger.error(f"Async model loading failed: {str(e)}")
        model_state.set_error(str(e))
        return False

def load_model_sync():
    """Synchronous model loading with comprehensive error handling"""
    try:
        logger.info("Starting model loading process...")
        
        # Method 1: Try SageMaker Registry
        model_details = get_latest_approved_model()
        
        if model_details:
            try:
                # Get model artifacts URL from registry
                artifacts_uri = model_details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
                logger.info(f"Downloading model from registry: {artifacts_uri}")
                
                # Download and extract model
                model_file_path, model_hash = download_and_extract_model(artifacts_uri)
                
                # Load XGBoost model
                booster = xgb.Booster()
                booster.load_model(model_file_path)
                
                # Update model state
                model_state.update_model(
                    model=booster,
                    version=model_details.get('ModelPackageVersion', 'unknown'),
                    arn=model_details['ModelPackageArn'],
                    model_hash=model_hash
                )
                
                logger.info(f"Model loaded from registry - Version: {model_state.model_version}")
                return True
                
            except Exception as e:
                logger.error(f"Registry model loading failed: {str(e)}")
                model_error_counter.labels(error_type='registry_load_failed').inc()
                # Continue to fallback
        
        # Method 2: Fallback to S3
        logger.info("Using S3 fallback for model loading")
        booster, version, arn, model_hash = download_model_from_s3_fallback()
        
        if booster:
            model_state.update_model(
                model=booster,
                version=version,
                arn=arn,
                model_hash=model_hash
            )
            logger.info("Model loaded from S3 fallback")
            return True
        
        raise Exception("All model loading methods failed")
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        model_state.set_error(error_msg)
        model_load_counter.labels(status='failed').inc()
        return False

def preprocess_features(data: CustomerData) -> pd.DataFrame:
    """Apply the same preprocessing as training with enhanced error handling"""
    try:
        # Convert to dict then DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Handle TotalCharges
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing values with median (fallback to 0)
        median_total_charges = df['TotalCharges'].median()
        if pd.isna(median_total_charges):
            median_total_charges = 0
        df['TotalCharges'].fillna(median_total_charges, inplace=True)
        
        # Encode categorical variables (same mapping as training)
        categorical_mappings = {
            'gender': {'Male': 1, 'Female': 0},
            'Partner': {'Yes': 1, 'No': 0},
            'Dependents': {'Yes': 1, 'No': 0},
            'PhoneService': {'Yes': 1, 'No': 0},
            'MultipleLines': {'Yes': 2, 'No': 1, 'No phone service': 0},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'OnlineSecurity': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'OnlineBackup': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'DeviceProtection': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'TechSupport': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'StreamingTV': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'StreamingMovies': {'Yes': 2, 'No': 1, 'No internet service': 0},
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'PaymentMethod': {
                'Electronic check': 0, 'Mailed check': 1, 
                'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
            }
        }
        
        # Apply mappings with fallback to 0
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)
        
        # Ensure column order matches training
        expected_features = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]
        
        # Ensure all features exist
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Validate data types
        df = df[expected_features].astype(float)
        
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Data preprocessing failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Churn Prediction API...")
    try:
        success = await load_model_async()
        if success:
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning("Model loading failed on startup, will retry on first request")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    is_healthy = model_state.is_loaded()
    api_health_gauge.set(1 if is_healthy else 0)
    
    return {
        "status": "healthy" if is_healthy else "model_not_loaded",
        "model_status": model_state.model_status,
        "model_version": model_state.model_version,
        "last_updated": model_state.last_updated,
        "error_count": model_state.error_count,
        "last_error": model_state.last_error if model_state.error_count > 0 else None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict customer churn with enhanced error handling and monitoring"""
    start_time = time.time()
    
    # Check if model is loaded
    if not model_state.is_loaded():
        logger.warning("Model not loaded, attempting to load...")
        success = await load_model_async()
        if not success:
            model_error_counter.labels(error_type='model_not_loaded').inc()
            raise HTTPException(
                status_code=503, 
                detail=f"Model not available. Error: {model_state.last_error}"
            )
    
    try:
        # Preprocess input
        df_processed = preprocess_features(customer)
        
        # Make prediction
        with model_state.loading_lock:
            dmatrix = xgb.DMatrix(df_processed)
            probability = float(model_state.model.predict(dmatrix)[0])
        
        prediction = 1 if probability > 0.5 else 0
        
        # Calculate confidence score
        confidence_score = abs(probability - 0.5) * 2  # 0-1 scale
        
        # Determine risk level
        if probability > 0.75:
            risk_level = "Very High"
        elif probability > 0.6:
            risk_level = "High"
        elif probability > 0.4:
            risk_level = "Medium"
        elif probability > 0.25:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        prediction_counter.labels(prediction_result=str(prediction)).inc()
        prediction_latency.observe(processing_time / 1000.0)
        
        response = PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            confidence_score=round(confidence_score, 4),
            risk_level=risk_level,
            model_version=str(model_state.model_version),
            model_arn=model_state.model_arn,
            prediction_timestamp=datetime.utcnow().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
        logger.info(f"Prediction completed in {processing_time:.2f}ms: {prediction} ({probability:.4f})")
        return response
        
    except Exception as e:
        model_error_counter.labels(error_type='prediction_error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the latest model from registry"""
    try:
        logger.info("Manual model reload requested")
        success = await load_model_async()
        
        if success:
            return {
                "message": "Model reloaded successfully",
                "model_version": model_state.model_version,
                "model_arn": model_state.model_arn,
                "timestamp": model_state.last_updated
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Model reload failed: {model_state.last_error}"
            )
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get comprehensive model information"""
    return ModelInfo(
        model_loaded=model_state.is_loaded(),
        model_version=model_state.model_version,
        model_arn=model_state.model_arn,
        model_hash=model_state.model_hash,
        last_updated=model_state.last_updated,
        model_status=model_state.model_status,
        error_count=model_state.error_count,
        last_error=model_state.last_error,
        registry_group=os.getenv('MODEL_REGISTRY_GROUP', 'ChurnModelPackageGroup'),
        s3_bucket=os.getenv('S3_BUCKET', 'mlops-churn-model-artifacts')
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model_state.is_loaded(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "reload_model": "/reload-model"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 