import os
import json
import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the XGBoost model for inference.
    This is called once when the container starts.
    """
    try:
        model_path = os.path.join(model_dir, "xgboost-model")
        booster = xgb.Booster()
        booster.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return booster
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Parse and preprocess the input data.
    Supports both JSON and CSV input formats.
    """
    try:
        if request_content_type == 'application/json':
            # Handle JSON input
            input_data = json.loads(request_body)
            
            if isinstance(input_data, dict):
                # Single prediction
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                # Batch prediction
                df = pd.DataFrame(input_data)
            else:
                raise ValueError("Invalid JSON format")
                
        elif request_content_type == 'text/csv':
            # Handle CSV input
            df = pd.read_csv(StringIO(request_body))
            
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        # Preprocess the data (same as training preprocessing)
        df_processed = preprocess_features(df)
        logger.info(f"Processed {len(df_processed)} rows for prediction")
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error in input processing: {str(e)}")
        raise

def preprocess_features(df):
    """
    Apply the same preprocessing as during training.
    """
    df_processed = df.copy()
    
    # Drop CustomerID if present
    if 'CustomerID' in df_processed.columns:
        df_processed = df_processed.drop(columns=['CustomerID'])
    
    # Handle TotalCharges
    if 'TotalCharges' in df_processed.columns:
        df_processed['TotalCharges'] = df_processed['TotalCharges'].replace(' ', np.nan)
        df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
        df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median(), inplace=True)
    
    # Define categorical columns (same as training)
    categorical_columns = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = df_processed[col].astype('category').cat.codes
    
    # Ensure all columns are numeric
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Expected feature order (excluding target variable 'Churn')
    expected_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    
    # Ensure all expected features are present
    for feature in expected_features:
        if feature not in df_processed.columns:
            df_processed[feature] = 0  # Default value for missing features
    
    # Select and reorder columns
    df_processed = df_processed[expected_features]
    
    return df_processed

def predict_fn(input_data, model):
    """
    Make predictions using the loaded model.
    """
    try:
        # Convert to DMatrix for XGBoost
        dmatrix = xgb.DMatrix(input_data)
        
        # Get predictions (probabilities)
        predictions = model.predict(dmatrix)
        
        # Convert to binary predictions (threshold = 0.5)
        binary_predictions = (predictions > 0.5).astype(int)
        
        logger.info(f"Generated predictions for {len(input_data)} samples")
        
        return {
            'predictions': binary_predictions.tolist(),
            'probabilities': predictions.tolist()
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def output_fn(prediction, content_type):
    """
    Format the prediction output.
    """
    try:
        if content_type == 'application/json':
            return json.dumps(prediction)
        elif content_type == 'text/csv':
            # Return predictions as CSV
            df_output = pd.DataFrame({
                'churn_prediction': prediction['predictions'],
                'churn_probability': prediction['probabilities']
            })
            return df_output.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported output content type: {content_type}")
            
    except Exception as e:
        logger.error(f"Error in output formatting: {str(e)}")
        raise

# Handler for local testing
def lambda_handler(event, context):
    """
    AWS Lambda handler for serverless inference (optional).
    """
    try:
        # Load model (in production, this would be cached)
        model = model_fn("/opt/ml/model")
        
        # Process input
        input_data = input_fn(event['body'], event.get('content-type', 'application/json'))
        
        # Make prediction
        prediction = predict_fn(input_data, model)
        
        # Format output
        output = output_fn(prediction, 'application/json')
        
        return {
            'statusCode': 200,
            'body': output,
            'headers': {
                'Content-Type': 'application/json'
            }
        }
        
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
            'headers': {
                'Content-Type': 'application/json'
            }
        }

# Example usage for local testing
if __name__ == "__main__":
    # Sample customer data for testing
    sample_data = {
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
    }
    
    print("Testing inference with sample data:")
    print(json.dumps(sample_data, indent=2))
