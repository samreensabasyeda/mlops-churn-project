from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import boto3
import pandas as pd
import xgboost as xgb
import numpy as np
import json
import os
import tarfile
import logging
from typing import Dict, List
import time
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Global model variable
model = None
model_version = None
last_updated = None

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
    risk_level: str
    model_version: str
    prediction_timestamp: str

def get_latest_approved_model():
    """Get the latest approved model from SageMaker Registry"""
    try:
        sm_client = boto3.client('sagemaker')
        
        # List approved model packages
        response = sm_client.list_model_packages(
            ModelPackageGroupName='ChurnModelPackageGroup',
            ModelApprovalStatus='Approved',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=1
        )
        
        if not response['ModelPackageSummaryList']:
            logger.warning("No approved models found in registry")
            return None
        
        latest_model = response['ModelPackageSummaryList'][0]
        logger.info(f"Found latest approved model: {latest_model['ModelPackageArn']}")
        return latest_model
        
    except Exception as e:
        logger.error(f"Error getting model from registry: {str(e)}")
        return None

def download_model_from_s3_fallback():
    """Fallback: Download model from S3 training output"""
    try:
        s3 = boto3.client('s3')
        
        # List objects to find latest model
        response = s3.list_objects_v2(
            Bucket='mlops-churn-model-artifacts',
            Prefix='output/',
            Delimiter='/'
        )
        
        if 'Contents' not in response:
            raise Exception("No model artifacts found in S3")
        
        # Download model.tar.gz
        s3.download_file(
            'mlops-churn-model-artifacts',
            'output/model.tar.gz',
            '/tmp/model.tar.gz'
        )
        
        # Extract model
        with tarfile.open('/tmp/model.tar.gz', 'r:gz') as tar:
            tar.extractall('/tmp/')
        
        # Load XGBoost model
        booster = xgb.Booster()
        booster.load_model('/tmp/xgboost-model')
        
        logger.info("Model loaded from S3 fallback")
        return booster, "s3-fallback"
        
    except Exception as e:
        logger.error(f"S3 fallback failed: {str(e)}")
        raise

def load_model():
    """Load the latest model - try registry first, then S3 fallback"""
    global model, model_version, last_updated
    
    try:
        # Method 1: Try SageMaker Registry
        latest_model_info = get_latest_approved_model()
        
        if latest_model_info:
            # Get model artifacts URL from registry
            sm_client = boto3.client('sagemaker')
            model_details = sm_client.describe_model_package(
                ModelPackageName=latest_model_info['ModelPackageArn']
            )
            
            # Download from model artifacts S3 location
            artifacts_uri = model_details['InferenceSpecification']['Containers'][0]['ModelDataUrl']
            logger.info(f"Downloading model from registry: {artifacts_uri}")
            
            # Parse S3 URI
            bucket = artifacts_uri.split('/')[2]
            key = '/'.join(artifacts_uri.split('/')[3:])
            
            s3 = boto3.client('s3')
            s3.download_file(bucket, key, '/tmp/model.tar.gz')
            
            # Extract and load
            with tarfile.open('/tmp/model.tar.gz', 'r:gz') as tar:
                tar.extractall('/tmp/')
            
            booster = xgb.Booster()
            booster.load_model('/tmp/xgboost-model')
            
            model = booster
            model_version = latest_model_info['ModelPackageVersion']
            last_updated = time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
            logger.info(f"✅ Model loaded from registry - Version: {model_version}")
            
        else:
            # Method 2: Fallback to S3
            logger.info("No approved model in registry, using S3 fallback")
            model, model_version = download_model_from_s3_fallback()
            last_updated = time.strftime('%Y-%m-%d %H:%M:%S UTC')
            
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise Exception(f"Could not load model: {str(e)}")

def preprocess_features(data: CustomerData) -> pd.DataFrame:
    """Apply the same preprocessing as training"""
    
    # Convert to dict then DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Handle TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
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
    
    # Apply mappings
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
    
    return df[expected_features]

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("🚀 Starting Churn Prediction API...")
    try:
        load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {str(e)}")
        # Don't fail startup, allow manual model reload

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if model is not None else "model_not_loaded"
    return {
        "status": status,
        "model_version": model_version,
        "last_updated": last_updated,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC')
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """Predict customer churn"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input
        df_processed = preprocess_features(customer)
        
        # Make prediction
        dmatrix = xgb.DMatrix(df_processed)
        probability = float(model.predict(dmatrix)[0])
        prediction = 1 if probability > 0.5 else 0
        
        # Determine risk level
        if probability > 0.7:
            risk_level = "High"
        elif probability > 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            model_version=str(model_version),
            prediction_timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the latest model from registry"""
    try:
        load_model()
        return {
            "message": "Model reloaded successfully",
            "model_version": model_version,
            "timestamp": last_updated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get current model information"""
    return {
        "model_loaded": model is not None,
        "model_version": model_version,
        "last_updated": last_updated,
        "registry_group": "ChurnModelPackageGroup"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 