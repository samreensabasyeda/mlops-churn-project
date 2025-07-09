import pandas as pd
import os
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_data(df):
    """Validate the input dataframe"""
    required_columns = {'customerID', 'Churn', 'TotalCharges'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    if df.empty:
        raise ValueError("Empty dataframe received")
    
    return True

def preprocess():
    try:
        # Input/output paths
        input_path = "/opt/ml/processing/input/preprocessed.csv"
        train_path = "/opt/ml/processing/output/train/train.csv"
        val_path = "/opt/ml/processing/output/validation/validation.csv"
        
        # Verify input exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create output directories
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
        #testing to resolve parser csv issue 
        logger.info(f"Reading input data from {input_path}")
        df = pd.read_csv(input_path, sep='|', on_bad_lines='skip', header=None)
        # df = pd.read_csv(input_path, encoding='utf-8')
        
        # Validate input
        validate_data(df)
        
        # Data cleaning
        logger.info("Starting data preprocessing...")
        start_time = datetime.now()
        
        df = df.drop(columns=['customerID'])
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        median_charges = df['TotalCharges'].median()
        df['TotalCharges'].fillna(median_charges, inplace=True)
        logger.info(f"Filled {df['TotalCharges'].isna().sum()} missing values in TotalCharges")

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
            logger.info(f"Encoded categorical column: {col}")

        # Split data (stratified by Churn)
        train_df = df.sample(frac=0.8, random_state=42, weights='Churn')
        val_df = df.drop(train_df.index)

        # Save processed data
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Preprocessing completed in {duration:.2f} seconds")
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Validation data shape: {val_df.shape}")
        logger.info(f"Saved train data to {train_path}")
        logger.info(f"Saved validation data to {val_path}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    preprocess()
    sys.exit(0)