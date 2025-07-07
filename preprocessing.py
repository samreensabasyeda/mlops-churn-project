import pandas as pd
import numpy as np
import os

def preprocess():
    # Input path
    input_path = "/opt/ml/processing/input/preprocessed.csv"
    
    # Output paths
    train_path = "/opt/ml/processing/output/train/train.csv"
    val_path = "/opt/ml/processing/output/validation/validation.csv"
    
    # Create directories
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    
    try:
        # Load data
        df = pd.read_csv(input_path)
        
        # Data cleaning
        df = df.drop(columns=['customerID'])
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # Handle missing values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes

        # Split data (80% train, 20% validation)
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)

        # Save processed data
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print("Preprocessing completed successfully")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess()