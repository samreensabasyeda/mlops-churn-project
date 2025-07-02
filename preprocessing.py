import pandas as pd
import numpy as np

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Example cleaning: drop customerID, convert target to binary
    df = df.drop(columns=['customerID'])
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Fill missing values or transform columns as needed
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical variables (simple example)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes

    df.to_csv(output_path, index=False)
