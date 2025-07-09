import pandas as pd
import os
import logging
import sys
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df):
    """Validate the input dataframe"""
    required_columns = {'CustomerID', 'Churn', 'TotalCharges'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    if df.empty:
        raise ValueError("Empty dataframe received")
    
    # Log basic info about the dataset
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Churn distribution: {df['Churn'].value_counts().to_dict()}")
    
    return True

def preprocess():
    try:
        # SageMaker Processing paths (standard)
        input_path = "/opt/ml/processing/input/preprocessed.csv"
        train_path = "/opt/ml/processing/output/train/train.csv"
        val_path = "/opt/ml/processing/output/validation/validation.csv"
        
        # For local testing fallback
        if not os.path.exists(input_path):
            input_path = "data/preprocessed.csv"
            train_path = "data/train.csv"
            val_path = "data/validation.csv"
            logger.info("Using local paths for testing")
        
        # Verify input exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directories
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_path), exist_ok=True)

        logger.info(f"Reading input data from {input_path}")
        
        # Read CSV with robust error handling
        try:
            df = pd.read_csv(input_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1')
        except Exception as e:
            logger.error(f"Failed to read CSV: {str(e)}")
            # Try with different options
            df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
            
        logger.info(f"Successfully loaded {len(df)} rows")
        
        # Validate input
        validate_data(df)
        
        # Data cleaning
        logger.info("Starting data preprocessing...")
        start_time = datetime.now()
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Drop CustomerID (not needed for training)
        if 'CustomerID' in df_processed.columns:
            df_processed = df_processed.drop(columns=['CustomerID'])
            logger.info("Dropped CustomerID column")
        
        # Convert Churn to binary
        if df_processed['Churn'].dtype == 'object':
            df_processed['Churn'] = df_processed['Churn'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
            logger.info("Converted Churn to binary (1/0)")
        
        # Handle TotalCharges - convert to numeric
        if 'TotalCharges' in df_processed.columns:
            # Replace empty strings with NaN
            df_processed['TotalCharges'] = df_processed['TotalCharges'].replace(' ', np.nan)
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            
            # Fill missing values with median
            median_charges = df_processed['TotalCharges'].median()
            missing_count = df_processed['TotalCharges'].isna().sum()
            df_processed['TotalCharges'].fillna(median_charges, inplace=True)
            logger.info(f"Filled {missing_count} missing values in TotalCharges with median: {median_charges:.2f}")

        # Encode categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Churn':  # Skip target variable
                df_processed[col] = df_processed[col].astype('category').cat.codes
                logger.info(f"Encoded categorical column: {col}")

        # Ensure all columns are numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # Manual train/validation split (avoid sklearn dependency)
        # Shuffle the dataframe
        df_shuffled = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split index (80% train, 20% validation)
        train_size = int(0.8 * len(df_shuffled))
        
        # Split data
        train_df = df_shuffled[:train_size]
        val_df = df_shuffled[train_size:]

        # Save processed data
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Preprocessing completed in {duration:.2f} seconds")
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Validation data shape: {val_df.shape}")
        logger.info(f"Train churn distribution: {train_df['Churn'].value_counts().to_dict()}")
        logger.info(f"Validation churn distribution: {val_df['Churn'].value_counts().to_dict()}")
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved validation data to {val_path}")
        
        return train_path, val_path
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess()
    sys.exit(0)